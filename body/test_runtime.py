"""Tests for the onboard runtime and the ONNX export.

    MUJOCO_GL=osmesa python body/test_runtime.py        # needs models/mid-stand/policy.onnx (package.py)
"""
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import runtime as rtm  # noqa: E402
from mouth import MAX_OPEN, MouthSync, demo_voice  # noqa: E402
from skills_env import SkillEnv  # noqa: E402

STAND = rtm.MODELS / "mid-stand"


class Stub:
    """A fake brain: fixed small action, so tests don't depend on which skills are trained yet."""

    def __init__(self, value=0.05):
        self.value = value

    def __call__(self, obs):
        return np.full(11, self.value) * np.sin(obs[:11] * 10)


def runtime_on(env, skills=("stand",), **kw):
    brains, cfg = rtm.load_brains()
    for s in skills:
        brains.setdefault(s, brains["stand"])
    be = rtm.SimBackend(cfg["servo_names"], model=env.model, data=env.data)
    return rtm.DuckRuntime(be, brains=brains, config=cfg, log=None, **kw)


@unittest.skipUnless((STAND / "policy.onnx").exists(), "run body/package.py --skill stand first")
class TestRuntime(unittest.TestCase):
    def check_obs_match(self, task, command, vel=(0.0, 0.0, 0.0), skills=("stand",), steps=150):
        env = SkillEnv(task=task, randomize=False, model_path=str(rtm.ROBOT_XML))
        env.reset(seed=3)
        rt = runtime_on(env, skills)
        for c in command.split("+"):
            rt.command(c, *vel)
        self.assertEqual(rt.obs_command, env.command)
        env.vel_cmd = rt.vel_cmd.copy()               # same command for both
        worst = 0.0
        for k in range(steps):
            env.t, env.last_action = rt.gait_t, rt.last_action.copy()
            a, b = rt.observe(), env._obs()
            self.assertEqual(a.shape, (60,))
            worst = max(worst, float(np.abs(a - b).max()))
            rt.tick()
        self.assertLess(worst, 1e-5, f"{task}: runtime obs differs from SkillEnv by {worst}")
        return worst

    def test_obs_equals_skillenv_stand(self):
        self.check_obs_match("stand", "stand")

    def test_obs_equals_skillenv_walk(self):           # velocity command and gait clock filled in
        self.check_obs_match("walk", "walk", vel=(0.03, 0.01, 0.3), skills=("stand", "walk"))

    def test_obs_equals_skillenv_sit_and_wake(self):
        self.check_obs_match("sit", "sit", skills=("stand", "sit"))
        self.check_obs_match("sit_to_stand", "sit+wake", skills=("stand", "sit", "sit_to_stand"))

    def test_onnx_matches_sb3(self):
        import package
        model, norm = package.load_brain(STAND)
        obs, _ = package.record_obs(model, norm, "stand", samples=300, seed=11)
        brain = rtm.OnnxBrain(STAND / "policy.onnx")
        got = np.array([brain(o) for o in obs])
        ref = package.sb3_act(model, norm, obs)
        self.assertLess(float(np.abs(got - ref).max()), 1e-5)
        self.assertLessEqual(float(np.abs(got).max()), 1.0)

    def test_command_switching(self):
        env = SkillEnv(task="stand", randomize=False, model_path=str(rtm.ROBOT_XML))
        env.reset(seed=0)
        brains, cfg = rtm.load_brains()
        stub = Stub()
        brains.update(sit=stub, sit_to_stand=stub, walk=stub)
        brains.pop("lie", None)
        rt = rtm.DuckRuntime(rtm.SimBackend(cfg["servo_names"], model=env.model, data=env.data),
                             brains=brains, config=cfg, log=None)
        self.assertEqual(rt.skill, "stand")
        rt.command("sleep")
        self.assertEqual((rt.skill, rt.obs_command), ("sit", "sit"))
        rt.tick()
        rt.command("wake")                                    # sit_to_stand, then stand after 3 s
        self.assertEqual((rt.skill, rt.obs_command), ("sit_to_stand", "stand"))
        rt.run(2.9)
        self.assertEqual(rt.skill, "sit_to_stand")
        rt.run(0.2)
        self.assertEqual(rt.skill, "stand")
        rt.command("walk", vx=0.02, yaw=0.3)
        self.assertEqual((rt.skill, rt.obs_command), ("walk", "walk"))
        np.testing.assert_allclose(rt.vel_cmd, [0.02, 0, 0.3])
        o = rt.observe()
        self.assertEqual(o[46 + 1], 1.0)                      # one-hot: walk
        np.testing.assert_allclose(o[53:56], [0.02, 0, 0.3], atol=1e-6)
        np.testing.assert_allclose(o[56:58], [0, 1], atol=1e-6)   # gait clock starts at 0
        rt.command("walk", vx=5.0)                            # clipped to the trained range
        self.assertLessEqual(rt.vel_cmd[0], 0.25 * rt.L + 1e-9)
        msg = rt.command("lie")                               # no brain: stand, and say so
        self.assertEqual(rt.skill, "stand")
        self.assertIn("no lie brain", msg)
        self.assertIn("lie", rt.fallbacks)
        np.testing.assert_allclose(rt.observe()[53:58], 0)    # standing: no velocity, no clock
        rt.command("idle")
        self.assertEqual(rt.skill, "stand")

    def test_text_commands(self):
        L = 0.7
        self.assertEqual(rtm.parse("go to sleep")[0], "sleep")
        self.assertEqual(rtm.parse("wake up")[0], "wake")
        kind, args = rtm.parse("walk forward", L)
        self.assertEqual(kind, "walk")
        self.assertGreater(args["vel"]["vx"], 0)
        kind, args = rtm.parse("turn left", L)
        self.assertEqual(kind, "walk")
        self.assertGreater(args["vel"]["yaw"], 0)
        self.assertLess(rtm.parse("turn right", L)[1]["vel"]["yaw"], 0)
        self.assertEqual(rtm.parse("say hello"), ("say", {"text": "hello"}))
        self.assertEqual(rtm.parse("lie down")[0], "lie")
        self.assertEqual(rtm.parse("stop")[0], "stand")
        self.assertIsNone(rtm.parse("blorp"))
        self.assertEqual(rtm.clauses("sit, then wake up and walk forward"), ["sit", "wake up", "walk forward"])

    def test_head_rate_limit(self):
        env = SkillEnv(task="stand", randomize=False, model_path=str(rtm.ROBOT_XML))
        env.reset(seed=0)
        rt = runtime_on(env)
        rt.look(1.0)
        rt.tick()
        self.assertAlmostEqual(rt.targets[rt.head], 2.0 / 50, places=6)
        rt.run(1.0)
        self.assertAlmostEqual(rt.targets[rt.head], 1.0, places=6)

    def test_jaw_follows_audio(self):
        env = SkillEnv(task="stand", randomize=False, model_path=str(rtm.ROBOT_XML))
        env.reset(seed=0)
        rt = runtime_on(env)
        sr = 22050
        audio, loud = demo_voice(seconds=2.0, sr=sr)
        rt.say(audio, sr)
        ref = MouthSync(sr)
        jaw, want, loudness = [], [], []
        for k in range(100):                                  # 2 s of speech, 20 ms per tick
            rt.tick()
            chunk = audio[k * ref.hop:(k + 1) * ref.hop]
            want.append(ref.feed(chunk))
            jaw.append(env.data.ctrl[rt.jaw])                 # what went to the jaw servo
            loudness.append(np.abs(loud[k * ref.hop:(k + 1) * ref.hop]).mean())
        jaw = np.array(jaw)
        np.testing.assert_allclose(jaw, want, atol=1e-9)      # exactly MouthSync on the same audio
        self.assertGreater(jaw.max(), 0.5 * MAX_OPEN)         # opens on syllables
        self.assertGreater(np.corrcoef(jaw, loudness)[0, 1], 0.5)
        rt.run(0.6)                                           # speech over: mouth closes
        self.assertFalse(rt.speaking)
        self.assertLess(env.data.ctrl[rt.jaw], 0.02)

    def test_stand_brain_keeps_duck_up(self):
        rt = rtm.DuckRuntime(log=None)
        rt.look(1.5)
        audio, _ = demo_voice(seconds=2.0)
        rt.say(audio, 22050)
        ups = []
        rt.run(4.0, every=lambda r: ups.append(r.backend.posture()["up"]))
        self.assertGreater(min(ups), 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
