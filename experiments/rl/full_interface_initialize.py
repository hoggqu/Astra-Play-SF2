"""Create a fresh, explicitly identified PPO; no emulator or training is run."""
import argparse
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
from .full_interface_identity import INTERFACE, OBSERVATION_INTERFACE, training_metadata


class ShapeEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, shape=(800,), dtype=np.float32)
    action_space = gym.spaces.Discrete(85)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(800, dtype=np.float32), {}

    def step(self, action):
        raise RuntimeError('ShapeEnv cannot generate training experience')


def create_model(seed=42):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    model = PPO('MlpPolicy', ShapeEnv(), seed=seed, device='cpu', n_steps=256,
                batch_size=64, n_epochs=4, learning_rate=3e-4, gamma=.99, ent_coef=.01,
                policy_kwargs={'net_arch': {'pi':[64,64], 'vf':[64,64]}})
    model.astra_action_interface = INTERFACE
    model.astra_observation_interface = OBSERVATION_INTERFACE
    model.astra_optimizer_origin = 'fresh_full85_feedback'
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    metadata = training_metadata(Path(__file__).resolve().parent)
    args.output.mkdir(parents=True, exist_ok=False)
    model = create_model(args.seed)
    model.save(args.output/'ppo-initial.zip')
    from astra_play_sf2.runner import sha256
    metadata.update(schema='astra.rl-full85-initialization.v1', status='complete', seed=args.seed,
                    training_steps=0, optimizer_state_entries=0,
                    model_sha256=sha256(args.output/'ppo-initial.zip'))
    (args.output/'result.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__': main()
