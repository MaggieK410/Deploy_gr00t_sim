"""
capture_attention.py — runtime attention capture for diffusers-based DiT.

Drop-in for the default ``AttnProcessor2_0``: produces numerically equivalent
outputs (unfused vs fused softmax — FP-precision difference only) and on every
call ALSO stashes the attention probabilities into a shared sink dict keyed by
block index.

Why this exists: modifying the gr00t source (to thread a ``return_attention``
flag through ``BasicTransformerBlock`` / diffusers ``Attention``) is fragile —
it has to be re-applied to every gr00t and every diffusers version. Replacing
``attn.processor`` at runtime achieves the same thing without touching either
library.

Use the same file in your sim fork and at robot deploy time; output of the
forward pass is mathematically identical to the stock processor, so trained
weights apply unchanged and the two run paths are reproducible.

Typical usage::

    from capture_attention import attach

    handle = attach(policy.model.action_head.model)  # DiT or AlternateVLDiT

    handle.reset()
    out = policy.predict(...)                          # one chunk inference
    maps = handle.read()                                # tensor or None
    blocks = handle.block_indices

    handle.detach()  # at shutdown, restores the original processors
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Custom processor                                                            #
# --------------------------------------------------------------------------- #
class _CapturingAttnProcessor:
    """Reimplementation of diffusers ``AttnProcessor2_0`` that ALSO stores the
    attention probabilities into a shared sink dict keyed by block_id.

    Each ``__call__`` appends one (B, H, T_q, T_k) tensor to
    ``sink[block_id]`` (float16, CPU). For a chunk inference with N denoising
    steps this means N entries per cross-attention block.

    Output of ``__call__`` is numerically equivalent to ``AttnProcessor2_0`` up
    to fused vs unfused softmax FP precision — trained weights load unchanged.
    """

    def __init__(self, block_id: int, sink: dict):
        self.block_id = block_id
        self.sink = sink

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B_, C_, H_, W_ = hidden_states.shape
            hidden_states = hidden_states.view(B_, C_, H_ * W_).transpose(1, 2)
        else:
            B_ = C_ = H_ = W_ = None  # placate linters

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        norm_q = getattr(attn, "norm_q", None)
        if norm_q is not None:
            query = norm_q(query)
        norm_k = getattr(attn, "norm_k", None)
        if norm_k is not None:
            key = norm_k(key)

        # Manual unfused attention so we can capture probs. Computing the
        # softmax in float32 keeps numerical parity with PyTorch's fused
        # implementation across precisions.
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if attention_mask is not None:
            scores = scores + attention_mask
        probs_full = F.softmax(scores, dim=-1, dtype=torch.float32)
        probs = probs_full.to(query.dtype)

        # Capture both cross- and self-attention probs. The CaptureHandle
        # tracks which block_ids are cross vs self so they can be stacked into
        # separate tensors (their T_k differs: cross = VLM tokens; self =
        # action tokens, usually 16). is_cross is recorded alongside the
        # tensor so the handle never has to inspect shapes.
        self.sink.setdefault(self.block_id, []).append(
            (probs_full.detach().to(torch.float16).cpu(), is_cross)
        )

        hidden_states = torch.matmul(probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                B_, C_, H_, W_
            )

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual
        rescale = getattr(attn, "rescale_output_factor", 1.0)
        if rescale != 1.0:
            hidden_states = hidden_states / rescale
        return hidden_states


# --------------------------------------------------------------------------- #
# Public handle                                                               #
# --------------------------------------------------------------------------- #
class CaptureHandle:
    """Owns the swapped processors and the per-inference capture sink.

    Lifecycle::

        handle = CaptureHandle().attach(dit_module)
        ...
        handle.reset(); model_forward(...); maps = handle.read()
        ...
        handle.detach()  # restores original processors

    ``read()`` returns a dict with both ``"cross"`` and ``"self"`` tensors;
    they have different T_k so they cannot be stacked into a single array.
    The corresponding block index lists are exposed as
    ``cross_block_indices`` / ``self_block_indices``.
    """

    def __init__(self):
        self.sink: dict = {}
        # block indices, partitioned by whether the block is a cross- or
        # self-attention block. cross/self ordering matches the order in
        # which read() stacks per-type tensors along their block axis.
        self.cross_block_indices: List[int] = []
        self.self_block_indices: List[int] = []
        self._all_block_indices: List[int] = []   # ordered 0..n_blocks-1
        self._originals: list = []
        self._attached: bool = False
        # Captured from AlternateVLDiT.forward(image_mask=...) via a pre-hook
        # so we can later split T_k into image vs text VLM tokens. Updated on
        # every forward; consumers usually read it after the first inference
        # in a session since it's constant for a fixed image-preproc + prompt.
        self.image_mask: Optional[torch.Tensor] = None
        self._dit_pre_hook_handle = None
        # Per-block hidden-state capture. Hidden states are the (B, T, D)
        # tensors flowing between transformer blocks — arbitrary floats, NOT
        # softmax probabilities. They're stored separately from attention
        # probs because their shape and semantics are different.
        # Sink layout: key 'initial' → list of input hidden_states (one per
        # denoise step); key <int block_id> → list of block-output
        # hidden_states (one per denoise step).
        self.hidden_sink: dict = {}
        self._block_post_hooks: list = []

    def attach(self, dit_module) -> "CaptureHandle":
        """Walk ``dit_module.transformer_blocks`` and replace EVERY block's
        ``attn1.processor`` with a capturing processor. Both cross- and
        self-attention blocks are wrapped; the processor records each call
        with an ``is_cross`` tag and the handle partitions them on read.

        Re-entrant: detach()+attach() can be called repeatedly (e.g. after a
        model reload). Per-call state — block index lists, ``_originals``,
        and the capture sink — is reset on every call so block lists don't
        accumulate across reloads."""
        if self._attached:
            raise RuntimeError("CaptureHandle already attached. Call detach() first.")
        if not hasattr(dit_module, "transformer_blocks"):
            raise TypeError(
                f"{type(dit_module).__name__} has no .transformer_blocks; "
                "pass the DiT (action_head.model), not the action head itself."
            )
        # Fresh state — previous attach()'s entries are stale after a reload.
        self.cross_block_indices = []
        self.self_block_indices = []
        self._all_block_indices = []
        self._originals = []
        self.sink.clear()
        self.hidden_sink.clear()
        self.image_mask = None
        # Register a pre-hook on the DiT itself to capture image_mask kwargs
        # AND the initial hidden_states going into the first block.
        # AlternateVLDiT.forward(hidden_states=..., image_mask=...).
        def _dit_pre(module, args_, kwargs_):
            mask = kwargs_.get("image_mask")
            if isinstance(mask, torch.Tensor):
                m = mask.detach()
                if m.dim() > 1:
                    m = m[0]   # take first batch element; B=1 at deploy
                self.image_mask = m.to(torch.bool).cpu()
            # Initial hidden_states (sa_embs) before any block runs. Stored
            # under key 'initial'; one entry per DiT forward (= one per
            # denoise step within a chunk inference).
            hs = args_[0] if args_ else kwargs_.get("hidden_states")
            if isinstance(hs, torch.Tensor):
                self.hidden_sink.setdefault("initial", []).append(
                    hs.detach().to(torch.float16).cpu()
                )
        # Remove any old hook before installing a fresh one.
        if self._dit_pre_hook_handle is not None:
            try:
                self._dit_pre_hook_handle.remove()
            except Exception:
                pass
        self._dit_pre_hook_handle = dit_module.register_forward_pre_hook(
            _dit_pre, with_kwargs=True
        )
        # Clean up any leftover per-block hooks before installing fresh ones.
        for h in self._block_post_hooks:
            try: h.remove()
            except Exception: pass
        self._block_post_hooks = []
        for idx, block in enumerate(dit_module.transformer_blocks):
            attn = getattr(block, "attn1", None)
            if attn is None:
                continue
            self._originals.append((attn, attn.processor))
            attn.processor = _CapturingAttnProcessor(block_id=idx, sink=self.sink)
            # Static block typing from cross_attention_dim. The processor's
            # is_cross tag (based on whether encoder_hidden_states was None)
            # is the source of truth at call time, but this static partition
            # is what we use to lay out the read() output tensors.
            cross_dim = getattr(block, "cross_attention_dim", None)
            if cross_dim is None:
                self.self_block_indices.append(idx)
            else:
                self.cross_block_indices.append(idx)
            self._all_block_indices.append(idx)
            # Forward hook on the block itself captures the block's OUTPUT
            # hidden state (post-attention, post-FF, post-residual). One
            # tensor per denoise step per block.
            def _make_block_hook(block_id):
                def _hook(_module, _args, output):
                    out_tensor = output[0] if isinstance(output, tuple) else output
                    if isinstance(out_tensor, torch.Tensor):
                        self.hidden_sink.setdefault(block_id, []).append(
                            out_tensor.detach().to(torch.float16).cpu()
                        )
                return _hook
            self._block_post_hooks.append(
                block.register_forward_hook(_make_block_hook(idx))
            )
        self._attached = True
        return self

    def detach(self) -> None:
        """Restore the original processors. Call at shutdown."""
        for attn, orig in self._originals:
            attn.processor = orig
        self._originals = []
        self._attached = False
        if self._dit_pre_hook_handle is not None:
            try:
                self._dit_pre_hook_handle.remove()
            except Exception:
                pass
            self._dit_pre_hook_handle = None
        for h in self._block_post_hooks:
            try: h.remove()
            except Exception: pass
        self._block_post_hooks = []

    def reset(self) -> None:
        """Drop any captured tensors. Call before each inference."""
        self.sink.clear()
        self.hidden_sink.clear()

    def _stack_indices(self, indices: List[int]) -> Optional[torch.Tensor]:
        """Stack one block-group's captures into (n_denoise, n_blocks, B, H, Tq, Tk).

        Returns None if any block in ``indices`` produced no captures during
        this inference (e.g. AlternateVLDiT may skip a block on a given run).
        """
        if not indices:
            return None
        per_block = []
        for block_id in indices:
            calls = self.sink.get(block_id)
            if not calls:
                return None
            # Each entry is (probs_tensor, is_cross); drop the tag here.
            probs_only = [c[0] if isinstance(c, tuple) else c for c in calls]
            per_block.append(torch.stack(probs_only, dim=0))  # (n_denoise, B, H, Tq, Tk)
        return torch.stack(per_block, dim=1)  # (n_denoise, n_blocks, B, H, Tq, Tk)

    def _stack_hidden(self) -> Optional[torch.Tensor]:
        """Stack initial + per-block hidden states into a single tensor of
        shape ``(n_denoise, n_blocks + 1, B, T, D)`` float16.

        Block axis layout: ``[initial, block_0_output, block_1_output, ...]``
        — matches the user's fork's ``all_hidden_states`` ordering.
        """
        if not self.hidden_sink:
            return None
        init_list = self.hidden_sink.get("initial")
        if not init_list:
            return None
        try:
            init = torch.stack(init_list, dim=0)        # (n_denoise, B, T, D)
        except Exception:
            return None
        per_block = []
        for block_id in self._all_block_indices:
            calls = self.hidden_sink.get(block_id)
            if not calls:
                return None
            per_block.append(torch.stack(calls, dim=0))  # (n_denoise, B, T, D)
        per_block_t = torch.stack(per_block, dim=1)      # (n_denoise, n_blocks, B, T, D)
        init_t = init.unsqueeze(1)                       # (n_denoise, 1, B, T, D)
        return torch.cat([init_t, per_block_t], dim=1)   # (n_denoise, n_blocks+1, B, T, D)

    def read(self) -> dict:
        """Return captured attention and hidden states.

        Returns a dict with keys:

        * ``"cross"``: tensor of shape
          ``(n_denoise, n_cross_blocks, B, H, T_q, T_k_vlm)`` float16, or None.
          These are softmax PROBABILITIES (rows sum to 1).
        * ``"self"``: tensor of shape
          ``(n_denoise, n_self_blocks, B, H, T_q, T_q)`` float16, or None.
          Also softmax probabilities.
        * ``"hidden_states"``: tensor of shape
          ``(n_denoise, n_blocks + 1, B, T_q, D)`` float16, or None.
          Block-axis layout: ``[initial, after_block_0, after_block_1, ...]``.
          These are ARBITRARY floats, NOT probabilities — the actual hidden-
          state tensors flowing between transformer blocks.
        * ``"cross_block_indices"`` / ``"self_block_indices"`` / ``"all_block_indices"``:
          lists of transformer block indices corresponding to the block axes.

        T_k differs between cross and self (VLM tokens vs action tokens), so
        they're stored separately. Hidden states are a single sequence (T_q),
        independent of cross vs self.
        """
        if not self.sink and not self.hidden_sink:
            return {
                "cross": None, "self": None, "hidden_states": None,
                "cross_block_indices": list(self.cross_block_indices),
                "self_block_indices":  list(self.self_block_indices),
                "all_block_indices":   list(self._all_block_indices),
            }
        return {
            "cross": self._stack_indices(self.cross_block_indices),
            "self":  self._stack_indices(self.self_block_indices),
            "hidden_states": self._stack_hidden(),
            "cross_block_indices": list(self.cross_block_indices),
            "self_block_indices":  list(self.self_block_indices),
            "all_block_indices":   list(self._all_block_indices),
        }

    @property
    def block_indices(self) -> List[int]:
        """Cross-attention block indices, kept for backward compatibility
        with callers that pre-date self-attention capture. Equivalent to
        ``cross_block_indices``."""
        return list(self.cross_block_indices)


def attach(dit_module) -> CaptureHandle:
    """Convenience: create a CaptureHandle and attach it in one call."""
    return CaptureHandle().attach(dit_module)
