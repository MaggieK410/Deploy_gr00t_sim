--- /home/mkulcsar/Isaac-GR00T/external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/utils/gym_utils/gymnasium_basic.py.bak	2026-07-07 11:08:39.319639749 +0200
+++ /home/mkulcsar/Isaac-GR00T/external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/utils/gym_utils/gymnasium_basic.py	2026-07-07 11:08:39.330639775 +0200
@@ -70,7 +70,86 @@
     )
     env_class = REGISTERED_ENVS[env_name]
 
-    env = robosuite.make(**env_kwargs)
+    # ── PATCHED (version drift band-aid) ───────────────────────────────
+    # (1) Filter env_kwargs to what the target env class accepts.
+    #     robocasa was written against a newer robosuite that has extra
+    #     kwargs (seed, translucent_robot, ...); robosuite@v1.5.1 doesn't.
+    # (2) On missing-camera ValueError from robosuite.make, substitute the
+    #     missing camera name with "robot0_robotview" and retry.
+    import inspect as _insp
+    import re as _re
+    from robosuite.environments.base import REGISTERED_ENVS as _REG
+    _en = env_kwargs.get("env_name")
+    if _en in _REG:
+        _cls = _REG[_en]
+        _accepted = set(_insp.signature(_cls.__init__).parameters.keys())
+        env_kwargs = {k: v for k, v in env_kwargs.items()
+                      if k in _accepted or k == "env_name"}
+    _FALLBACK_CAM = "robot0_robotview"
+    _cam_substitutions = {}   # missing_name -> substitute_name
+    for _try in range(8):
+        try:
+            env = robosuite.make(**env_kwargs)
+            break
+        except ValueError as _e:
+            _emsg = str(_e)
+            # robosuite raises this in two forms depending on whether it's
+            # caught & wrapped by Observable._check_sensor_validity:
+            #   direct : 'No "camera" with name <name> exists'
+            #   wrapped: 'Current sensor for observable <name>_image is invalid.'
+            _mcam = (_re.search(r'No "camera" with name (\S+) exists', _emsg)
+                     or _re.search(r'observable (\S+?)_image is invalid', _emsg))
+            if _mcam and "camera_names" in env_kwargs:
+                _bad = _mcam.group(1)
+                _cams = list(env_kwargs["camera_names"])
+                if _bad in _cams:
+                    _idx = _cams.index(_bad)
+                    if _FALLBACK_CAM in _cams:
+                        del _cams[_idx]
+                    else:
+                        _cams[_idx] = _FALLBACK_CAM
+                    _cam_substitutions[_bad] = _FALLBACK_CAM
+                    print(f'[robocasa patch] camera "{_bad}" missing from '
+                          f'scene; substituting -> "{_FALLBACK_CAM}" '
+                          f'(result: {_cams})')
+                    env_kwargs["camera_names"] = _cams
+                    continue
+            raise
+    else:
+        raise RuntimeError('Failed after 8 camera-substitution retries')
+    
+    # (3) Downstream wrappers look up observation keys by the ORIGINAL
+    #     camera name (e.g. `egoview_image`), but robosuite emits keys
+    #     under the SUBSTITUTED name (e.g. `robot0_robotview_image`).
+    #     Wrap step/reset to add the original key as an alias of the
+    #     substituted one. Cheap and non-invasive.
+    if _cam_substitutions:
+        def _alias_obs(_obs, _subs=_cam_substitutions):
+            if not isinstance(_obs, dict):
+                return _obs
+            for _orig, _sub in _subs.items():
+                _src = f'{_sub}_image'
+                _dst = f'{_orig}_image'
+                if _src in _obs and _dst not in _obs:
+                    _obs[_dst] = _obs[_src]
+            return _obs
+        _orig_step = env.step
+        _orig_reset = env.reset
+        def _wrapped_step(action):
+            _r = _orig_step(action)
+            if isinstance(_r, tuple) and len(_r) >= 1:
+                return (_alias_obs(_r[0]),) + tuple(_r[1:])
+            return _alias_obs(_r)
+        def _wrapped_reset(*a, **kw):
+            _r = _orig_reset(*a, **kw)
+            if isinstance(_r, tuple):
+                return (_alias_obs(_r[0]),) + tuple(_r[1:])
+            return _alias_obs(_r)
+        env.step = _wrapped_step
+        env.reset = _wrapped_reset
+        print(f'[robocasa patch] installed obs-key aliases: '
+              f'{[(o, s) for o, s in _cam_substitutions.items()]}')
+    # ── END PATCH ──────────────────────────────────────────────────────
     return env, env_kwargs
 
 
