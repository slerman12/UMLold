# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time
from math import inf

from Datasets.Suites import DMC, Atari, Classify


class Environment:
    def __init__(self, task_name, frame_stack, action_repeat, max_episode_frames, truncate_episode_frames,
                 seed=0, train=True, suite="DMC", offline=False, generate=False, batch_size=1, num_workers=1):
        self.suite = suite
        self.offline = offline
        self.generate = generate

        self.env = self.raw_env.make(task_name, frame_stack, action_repeat, max_episode_frames,
                                     truncate_episode_frames, offline, generate, train, seed, batch_size, num_workers)

        self.env.reset()

        self.episode_step = self.last_episode_len = self.episode_reward = self.last_episode_reward = 0
        self.daybreak = None

    @property
    def raw_env(self):
        if self.suite.lower() == "dmc":
            return DMC
        elif self.suite.lower() == "atari":
            return Atari
        elif self.suite.lower() == 'classify':
            return Classify

    def __getattr__(self, item):
        return getattr(self.env, item)

    def rollout(self, agent, steps=inf, vlog=False):
        if self.daybreak is None:
            self.daybreak = time.time()  # "Daybreak" for whole episode

        experiences = []
        video_image = []

        exp = self.exp

        self.offline = self.offline or self.env.depleted or self.generate
        self.episode_done = False

        if self.offline and agent.training:
            agent.step += 1
            agent.episode += 1
            self.episode_done = True
            return None, None, None

        step = 0
        while not self.episode_done and step < steps:
            # Act
            action = agent.act(exp.observation)

            if not self.generate:
                exp = self.env.step(action.cpu().numpy())

            exp.step = agent.step

            experiences.append(exp)

            if vlog or self.generate:
                frame = action[:24].view(-1, *exp.observation.shape[1:]) if self.generate \
                    else self.env.physics.render(height=256, width=256, camera_id=0) \
                    if hasattr(self.env, 'physics') else self.env.render()
                video_image.append(frame)

                import torch
                video_image.append(torch.tensor(exp.observation[:24]).to(action.device) / 127.5 - 1)

            # Tally reward, done, step
            self.episode_reward += exp.reward.mean()
            self.episode_done = exp.last() or self.generate
            step += 1

        self.episode_step += step

        if self.episode_done:
            if agent.training:
                agent.episode += 1
            self.env.reset()

            self.last_episode_len = self.episode_step
            self.last_episode_reward = self.episode_reward

        # Log stats
        sundown = time.time()
        frames = self.episode_step * self.action_repeat

        logs = {'time': sundown - agent.birthday,
                'step': agent.step,
                'frame': agent.step * self.action_repeat,
                'episode': agent.episode,
                'accuracy' if self.suite.lower() == 'classify' else 'reward': self.episode_reward,
                'fps': frames / (sundown - self.daybreak)}

        if self.episode_done:
            self.episode_step = self.episode_reward = 0
            self.daybreak = sundown

        return experiences, logs, video_image
