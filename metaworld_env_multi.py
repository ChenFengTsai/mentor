import os
import gym
import numpy as np
from dm_env import StepType, specs
import dm_env
import numpy as np
from gym import spaces
from typing import Any, NamedTuple
from collections import deque, defaultdict
import random

class MetaWorld:
    def __init__(
        self,
        name,
        seed=None,
        action_repeat=1,
        size=(64, 64),
        camera=None,
    ):
        import metaworld
        from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
        )

        os.environ["MUJOCO_GL"] = "egl"

        task = f"{name}-v2-goal-observable"
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        self._env = env_cls(seed=seed)
        self._env._freeze_rand_vec = False
        self._size = size
        self._action_repeat = action_repeat
        self._camera = camera
        self.task_name = name

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "state": self._env.observation_space,
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
            "task_name": gym.spaces.Discrete(1),
        }
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        return {"action": action}

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        reward = 0.0
        success = 0.0
        for _ in range(self._action_repeat):
            state, rew, done, info = self._env.step(action["action"])
            success += float(info["success"])
            reward += rew or 0.0
        success = min(success, 1.0)
        assert success in [0.0, 1.0]
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": False,
            "is_terminal": False,
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ),
            "state": state,
            "success": success,
            "task_name": self.task_name,
        }
        return obs

    def reset(self):
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        state = self._env.reset()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ),
            "state": state,
            "success": False,
            "task_name": self.task_name,
        }
        return obs

class NormalizeAction:
    def __init__(self, env, key="action"):
        self._env = env
        self._key = key
        space = env.act_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self._env.step({**action, self._key: orig})

class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            obs["is_last"] = True
            self._step = None
        return obs

    def reset(self):
        self._step = 0
        return self._env.reset()

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    success: Any
    task_name: Any = None

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)

class metaworld_wrapper():
    def __init__(self, env, nstack=3):
        self._env = env
        self.nstack = nstack
        wos = env.obs_space['image']
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros(low.shape, low.dtype)

        self.observation_space = spaces.Box(low=np.transpose(low, (2, 0, 1)), 
                                          high=np.transpose(high, (2, 0, 1)), 
                                          dtype=np.uint8)

    def observation_spec(self):
        return specs.BoundedArray(self.observation_space.shape,
                                  np.uint8,
                                  0,
                                  255,
                                  name='observation')

    def action_spec(self):
        return specs.BoundedArray(self._env.act_space['action'].shape,
                                  np.float32,
                                  self._env.act_space['action'].low,
                                  self._env.act_space['action'].high,
                                  'action')

    def reset(self):
        time_step = self._env.reset()
        obs = time_step['image']
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return ExtendedTimeStep(observation=np.transpose(self.stackedobs, (2, 0, 1)),
                               step_type=StepType.FIRST,
                               action=np.zeros(self.action_spec().shape, dtype=self.action_spec().dtype),
                               reward=0.0,
                               discount=1.0,
                               success=time_step['success'],
                               task_name=time_step.get('task_name', 'unknown'))
    
    def step(self, action):
        action = {'action': action}
        time_step = self._env.step(action)
        obs = time_step['image']
        self.stackedobs = np.roll(self.stackedobs, shift=-obs.shape[-1], axis=-1)
        self.stackedobs[..., -obs.shape[-1]:] = obs

        if time_step['is_first']:
            step_type = StepType.FIRST
        elif time_step['is_last']:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
            
        return ExtendedTimeStep(observation=np.transpose(self.stackedobs, (2, 0, 1)),
                               step_type=step_type,
                               action=action['action'],
                               reward=time_step['reward'],
                               discount=1.0,
                               success=time_step['success'],
                               task_name=time_step.get('task_name', 'unknown'))

def make_single_env(name, frame_stack, action_repeat, seed, size=(84, 84)):
    """Create a single MetaWorld environment"""
    env = MetaWorld(name, seed, action_repeat, size, 'corner2')
    env = NormalizeAction(env)
    env = TimeLimit(env, 250)
    env = metaworld_wrapper(env, frame_stack)
    return env

class OpenOnlyMultiTaskEnv:
    """Multi-task environment for open tasks only: door-open, drawer-open, window-open"""
    
    def __init__(self, frame_stack=3, action_repeat=1, seed=None, size=(84, 84), episode_range=(5, 10)):
        # Only open tasks
        self.task_names = ['door-open', 'drawer-open', 'window-open']
        self.task_to_id = {name: idx for idx, name in enumerate(self.task_names)}
        self.episode_range = episode_range
        self.current_task_idx = 0
        self.current_episode_count = 0
        self.target_episodes = 0
        
        # Create environments for each task
        self.envs = {}
        for i, task_name in enumerate(self.task_names):
            task_seed = seed + i if seed is not None else None
            self.envs[task_name] = make_single_env(
                task_name, frame_stack, action_repeat, task_seed, size
            )
        
        # Use first environment's spaces as reference
        first_env = list(self.envs.values())[0]
        self.observation_space = first_env.observation_space
        
        # Training statistics
        self.task_stats = defaultdict(lambda: {'episodes': 0, 'success_rate': 0.0, 'rewards': []})
        
    def observation_spec(self):
        return list(self.envs.values())[0].observation_spec()
    
    def action_spec(self):
        return list(self.envs.values())[0].action_spec()
        
    def _select_next_task(self):
        # Create shuffled task list if needed
        if not hasattr(self, "_shuffled_tasks") or not self._shuffled_tasks:
            self._shuffled_tasks = list(range(len(self.task_names)))
            random.shuffle(self._shuffled_tasks)
            print(f"New open-task cycle: {[self.task_names[i] for i in self._shuffled_tasks]}")
        
        self.current_task_idx = self._shuffled_tasks.pop(0)
        self.current_episode_count = 0
        self.target_episodes = random.randint(*self.episode_range)
        
        print(f"Switching to: {self.current_task_name}, target episodes: {self.target_episodes}")

    @property
    def current_task_name(self):
        return self.task_names[self.current_task_idx]
    
    @property 
    def current_env(self):
        return self.envs[self.current_task_name]
    
    def reset(self):
        if self.target_episodes == 0 or self.current_episode_count >= self.target_episodes:
            self._select_next_task()
            
        time_step = self.current_env.reset()
        return time_step
    
    def step(self, action):
        time_step = self.current_env.step(action)
        
        if time_step.last():
            self.current_episode_count += 1
            # Update statistics
            self.task_stats[self.current_task_name]['episodes'] += 1
            self.task_stats[self.current_task_name]['rewards'].append(time_step.reward)
            
            # Update success rate
            if hasattr(time_step, 'success') and time_step.success:
                total_episodes = self.task_stats[self.current_task_name]['episodes']
                current_successes = self.task_stats[self.current_task_name]['success_rate'] * (total_episodes - 1)
                if time_step.success:
                    current_successes += 1
                self.task_stats[self.current_task_name]['success_rate'] = current_successes / total_episodes
        
        return time_step
    
    def print_statistics(self):
        print("\n" + "="*50)
        print("OPEN-TASK TRAINING STATISTICS")
        print("="*50)
        for task_name in self.task_names:
            if task_name in self.task_stats:
                stats = self.task_stats[task_name]
                rewards = stats['rewards']
                print(f"Task: {task_name}")
                print(f"  Episodes: {stats['episodes']}")
                print(f"  Success Rate: {stats['success_rate']:.3f}")
                print(f"  Avg Reward: {np.mean(rewards) if rewards else 0.0:.3f}")
                print("-" * 30)

def make(name, frame_stack, action_repeat, seed):
    """
    Create environment - supports 'open_only' for multi-task or individual task names
    """
    if name == 'open_only':
        return OpenOnlyMultiTaskEnv(
            frame_stack=frame_stack,
            action_repeat=action_repeat, 
            seed=seed,
            size=(84, 84),
            episode_range=(5, 10)
        )
    else:
        # Single task (maintains backward compatibility)
        return make_single_env(name, frame_stack, action_repeat, seed)