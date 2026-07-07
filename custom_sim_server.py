"""Custom GR00T inference server for the robocasa-gr1-tabletop-tasks pipeline.

Bypasses `Gr00tSimPolicyWrapper`. That wrapper's docstring says:

    If you are using other environments, custom robots, or building new
    environments, you should use `Gr00tPolicy` directly and format your
    observations according to its interface.

Which is our case. The robocasa GR1 env at the pinned commit uses LEGACY
modality names that don't line up with modern checkpoints, so instead of
monkey-patching robocasa to rename them, we do the mapping here in Python:

    Env key                                       Model modality key
    -------                                       ------------------
    video.ego_view_pad_res256_freq20         →    video.ego_view
    state.left_arm  / .right_arm  / .waist   →    state.left_arm ...  (1:1)
    state.left_hand / .right_hand            →    state.left_hand / .right_hand
    annotation.human.coarse_action           →    annotation.human.action.task_description
       (prefixed "unlocked_waist: ...")           (raw text — prefix stripped)

Also lets you dictate the task description sent to the VLA independent of
whatever `raw_obs["language"]` gives (`--language-override "..."`).

Usage
-----
    python custom_sim_server.py \\
        --model-path ../checkpoint-60000-red-ball-large-sim/ \\
        --embodiment-tag NEW_EMBODIMENT \\
        --device cuda:0 --port 5555 \\
        --language-override "pick up the red cube"

Notes
-----
- Replaces `run_gr00t_server.py --model-path ... --use-sim-policy-wrapper`
  for this env. Don't run both.
- `patch_robocasa_kwargs.sh` (kwargs filter + camera substitution) is
  still required — that fixes env CONSTRUCTION, not observation naming.
- `patch_robocasa_video_key_alias.sh` becomes redundant when using this
  server; you can revert it with `mv gymnasium_groot.py.bak gymnasium_groot.py`.
"""

from dataclasses import dataclass
from typing import Any

import tyro

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.policy import PolicyWrapper
from gr00t.policy.server_client import PolicyServer


VIDEO_KEY_MAP: dict[str, str] = {
    # Env flat key                              -> model modality key (inside "video")
    "video.ego_view_pad_res256_freq20": "ego_view",
}
STATE_KEY_MAP: dict[str, str] = {
    # Env flat key       -> model modality key (inside "state")
    "state.left_arm":   "left_arm",
    "state.right_arm":  "right_arm",
    "state.left_hand":  "left_hand",
    "state.right_hand": "right_hand",
    "state.waist":      "waist",
}
LANGUAGE_ENV_KEY = "annotation.human.coarse_action"
LANGUAGE_MODEL_KEY = "annotation.human.action.task_description"
LANGUAGE_PREFIXES = ("locked_waist: ", "unlocked_waist: ")


class CustomSimWrapper(PolicyWrapper):
    """Translate flat env obs → nested Gr00tPolicy obs, and back for actions.

    Mirrors the interface of `Gr00tSimPolicyWrapper` (same `PolicyWrapper`
    base, same public methods), but with the naming this specific env
    actually produces.
    """

    def __init__(
        self,
        policy: Gr00tPolicy,
        *,
        strict: bool = True,
        language_override: str | None = None,
    ):
        super().__init__(policy, strict=strict)
        self.policy = policy
        self.language_override = language_override
        self._dumped_shapes = False

        # Log the model's declared state modality keys, so we can see at
        # a glance what the wrapper is expected to feed.
        try:
            state_cfg = policy.modality_configs["state"]
            print(f"[wrapper] model state modality_keys: {list(state_cfg.modality_keys)}")
        except Exception as e:
            print(f"[wrapper] could not introspect model state config: {e}")

    def check_observation(self, observation: dict[str, Any]) -> None:
        for env_key in VIDEO_KEY_MAP:
            assert env_key in observation, (
                f"Env observation missing expected video key '{env_key}'. "
                f"Available keys: {sorted(observation.keys())}"
            )
        for env_key in STATE_KEY_MAP:
            assert env_key in observation, (
                f"Env observation missing expected state key '{env_key}'. "
                f"Available keys: {sorted(observation.keys())}"
            )
        if self.language_override is None:
            assert LANGUAGE_ENV_KEY in observation, (
                f"Env observation missing '{LANGUAGE_ENV_KEY}'. "
                f"Pass --language-override to hardcode a task description instead."
            )

    def _get_action(self, observation: dict[str, Any], options=None):
        nested: dict[str, dict[str, Any]] = {"video": {}, "state": {}, "language": {}}

        for env_key, model_key in VIDEO_KEY_MAP.items():
            nested["video"][model_key] = observation[env_key]
        for env_key, model_key in STATE_KEY_MAP.items():
            nested["state"][model_key] = observation[env_key]

        # One-shot shape dump. Prints ONCE per server run so we can see
        # exactly what each state key looks like when normalization fails.
        if not self._dumped_shapes:
            print("[wrapper] --- first call: shape dump ---")
            for mod in ("video", "state"):
                for k, v in nested[mod].items():
                    shape = getattr(v, "shape", None)
                    dtype = getattr(v, "dtype", None)
                    print(f"[wrapper]   {mod}.{k}: shape={shape}, dtype={dtype}")
            print(f"[wrapper]   language[{LANGUAGE_MODEL_KEY}]: "
                  f"{nested['language'][LANGUAGE_MODEL_KEY]!r}")
            print("[wrapper] --- end shape dump ---")
            self._dumped_shapes = True

        # Language modality: expected shape (B, T=1) as list[list[str]].
        batch_size = self._infer_batch_size(observation)
        if self.language_override is not None:
            texts = [self.language_override] * batch_size
        else:
            raw = observation[LANGUAGE_ENV_KEY]
            if isinstance(raw, str):
                raw = [raw] * batch_size
            texts = [self._strip_prefix(t) for t in raw]
        nested["language"][LANGUAGE_MODEL_KEY] = [[t] for t in texts]

        action, info = self.policy.get_action(nested, options)

        # Env expects flat keys `action.<name>`.
        return {f"action.{k}": v for k, v in action.items()}, info

    def check_action(self, action: dict[str, Any]) -> None:
        # The inner Gr00tPolicy already validated its own nested output;
        # nothing meaningful to add here.
        pass

    @staticmethod
    def _infer_batch_size(observation: dict[str, Any]) -> int:
        for env_key in VIDEO_KEY_MAP:
            v = observation.get(env_key)
            if v is not None:
                return len(v)
        return 1

    @staticmethod
    def _strip_prefix(text):
        if not isinstance(text, str):
            return text
        for prefix in LANGUAGE_PREFIXES:
            if text.startswith(prefix):
                return text[len(prefix):]
        return text


@dataclass
class ServerConfig:
    model_path: str
    """Path to the model checkpoint directory."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag matching the checkpoint's training config."""

    device: str = "cuda:0"
    """Device to run the model on."""

    host: str = "0.0.0.0"
    port: int = 5555

    strict: bool = True
    """Enforce input/output validation."""

    language_override: str | None = None
    """If set, sent to the VLA regardless of what the env's language field is.
    Useful when the env's default task description doesn't match your finetune's."""


def main(cfg: ServerConfig):
    print(f"[server] loading {cfg.model_path} on {cfg.device}")
    policy = Gr00tPolicy(
        embodiment_tag=cfg.embodiment_tag,
        model_path=cfg.model_path,
        device=cfg.device,
        strict=cfg.strict,
    )
    wrapped = CustomSimWrapper(
        policy,
        strict=cfg.strict,
        language_override=cfg.language_override,
    )
    print(
        f"[server] custom wrapper: "
        f"video={list(VIDEO_KEY_MAP)}, state={list(STATE_KEY_MAP)}, "
        f"language_override={cfg.language_override!r}"
    )
    print(f"[server] listening on {cfg.host}:{cfg.port}")
    server = PolicyServer(policy=wrapped, host=cfg.host, port=cfg.port)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[server] shutdown")


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
