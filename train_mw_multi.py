import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import utils
import torch
from dm_env import specs

import metaworld_env_multi as mw  # Using the simplified open-only environment

from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import wandb
import math
import re

from utils import models_tuple
from copy import deepcopy

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.cfg = cfg
        
        # Add device setup here
        import setup_utils as setup_utils
        device_mapper, device_obj = setup_utils.setup_device(cfg)
        
        print("#"*20)
        print(f'\nworkspace: {self.work_dir}')
        print(self.cfg)
        self.last_save_step = -9999
        
        # Check if this is open-only multi-task
        task_name = getattr(cfg, 'task_name', getattr(cfg, 'task', 'open_only'))
        self.is_open_multitask = (task_name == 'open_only')
        
        if self.cfg.use_wandb:
            exp_name = '_'.join([task_name, str(cfg.seed)])
            group_name = re.search(r'\.(.+)\.', cfg.agent._target_).group(1)
            name_1 = task_name
            name_2 = group_name
            try:
                name_2 += '_' + cfg.title
            except:
                pass
            name_3 = exp_name
            wandb.init(project=name_1,
                       group=name_2,
                       name=name_3,
                       config=cfg)
                       
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self._discount = cfg.discount
        self._discount_alpha = getattr(cfg, 'discount_alpha', 0.0)
        self._discount_alpha_temp = getattr(cfg, 'discount_alpha_temp', 1.0)
        self._discount_beta = getattr(cfg, 'discount_beta', 0.0)
        self._discount_beta_temp = getattr(cfg, 'discount_beta_temp', 1.0)
        self._nstep = cfg.nstep
        self._nstep_alpha = getattr(cfg, 'nstep_alpha', 0.0)
        self._nstep_alpha_temp = getattr(cfg, 'nstep_alpha_temp', 1.0)
        self.setup()
        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(), self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=self.cfg.use_tb,
                             use_wandb=self.cfg.use_wandb)
        
        # create envs
        task_name = getattr(self.cfg, 'task_name', getattr(self.cfg, 'task', 'open_only'))
        
        self.train_env = mw.make(task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed)
        self.eval_env = mw.make(task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed)
        
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1, ), np.float32, 'reward'),
                      specs.Array((1, ), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')
        self.replay_loader, self.buffer = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers, self.cfg.save_snapshot,
            math.floor(self._nstep + self._nstep_alpha),
            self._discount - self._discount_alpha - self._discount_beta)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @property
    def discount(self):
        return self._discount - self._discount_alpha * math.exp(
            -self.global_step /
            self._discount_alpha_temp) - self._discount_beta * math.exp(
                -self.global_step / self._discount_beta_temp)

    @property
    def nstep(self):
        return math.floor(self._nstep + self._nstep_alpha *
                          math.exp(-self.global_step / self._nstep_alpha_temp))

    def update_buffer(self):
        #self.buffer.update_discount(self.discount)
        self.buffer.update_nstep(self.nstep)
        return
    
    def eval(self):
        step, episode, total_reward, total_sr = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        
        # Track open-task specific metrics
        if self.is_open_multitask:
            task_rewards = {}
            task_success_rates = {}

        while eval_until_episode(episode):
            episode_sr = False
            episode_reward = 0
            time_step = self.eval_env.reset()
            
            # Track current task for open-only multi-task
            current_task = getattr(time_step, 'task_name', 'unknown')
            if self.is_open_multitask:
                if current_task not in task_rewards:
                    task_rewards[current_task] = []
                    task_success_rates[current_task] = []
                    
            self.video_recorder.init(self.eval_env, enabled=False)
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                episode_sr = episode_sr or time_step.success
                self.video_recorder.record(self.eval_env)
                episode_reward += time_step.reward
                step += 1

            total_reward += episode_reward
            total_sr += episode_sr
            
            # Track task-specific metrics for open-only multi-task
            if self.is_open_multitask:
                task_rewards[current_task].append(episode_reward)
                task_success_rates[current_task].append(episode_sr)
                
            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')
            
        # Log overall metrics
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_success_rate', total_sr / episode)
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            
            # Log task-specific metrics for open-only multi-task
            if self.is_open_multitask:
                for task_name in ['door-open', 'drawer-open', 'window-open']:
                    if task_name in task_rewards and task_rewards[task_name]:
                        avg_reward = np.mean(task_rewards[task_name])
                        avg_sr = np.mean(task_success_rates[task_name])
                        
                        log(f'task_{task_name.replace("-", "_")}_reward', avg_reward)
                        log(f'task_{task_name.replace("-", "_")}_success_rate', avg_sr)
                        
                        print(f"Eval {task_name}: Reward {avg_reward:.3f}, Success {avg_sr:.3f}")

    def train(self):
        # predicates 
        train_until_step = utils.Until(getattr(self.cfg, 'num_train_frames', 1000000),
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        
        # Stats printing for open-only multi-task
        if self.is_open_multitask:
            stats_every_step = utils.Every(getattr(self.cfg, 'stats_every_frames', 100000),
                                         self.cfg.action_repeat)

        episode_step, episode_reward, episode_sr = 0, 0, False
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        
        current_task = getattr(time_step, 'task_name', 'unknown')
        
        metrics = None
        print("Starting open-task training...")
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_success_rate', episode_sr)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)
                        
                        # Log current task info for open-only multi-task
                        if self.is_open_multitask:
                            # log(f'current_task', current_task)
                            log(f'task_{current_task.replace("-", "_")}_episode_reward', episode_reward)
                            log(f'task_{current_task.replace("-", "_")}_episode_success_rate', episode_sr)
                            
                            # Print task info
                            if hasattr(self.train_env, 'current_episode_count') and hasattr(self.train_env, 'target_episodes'):
                                print(f"Episode {self.global_episode}: {current_task}, "
                                      f"Progress {self.train_env.current_episode_count}/{self.train_env.target_episodes}, "
                                      f"Reward: {episode_reward:.3f}, Success: {episode_sr}")
                
                # update priority queue (if using)
                if hasattr(self.agent, 'tp_set'):
                    self.agent.tp_set.add(episode_reward,\
                                            deepcopy(self.agent.actor),\
                                            deepcopy(self.agent.critic),\
                                            deepcopy(self.agent.critic_target),\
                                            deepcopy(self.agent.value_predictor),\
                                            moe=deepcopy(self.agent.actor.moe.experts),\
                                            gate=deepcopy(self.agent.actor.moe.gate))                    
                
                # reset env
                time_step = self.train_env.reset()
                current_task = getattr(time_step, 'task_name', 'unknown')
                
                self.replay_storage.add(time_step)
                if self.cfg.save_snapshot and self.global_step - self.last_save_step >= self.cfg.save_interval:
                    self.last_save_step = self.global_step
                    self.save_snapshot(self.global_step)
                episode_sr = False
                episode_step = 0
                episode_reward = 0

            # Print open-task statistics periodically
            if self.is_open_multitask and hasattr(self, 'stats_every_step') and stats_every_step(self.global_step):
                if hasattr(self.train_env, 'print_statistics'):
                    self.train_env.print_statistics()

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step) and self.global_step % self.cfg.update_every_steps == 0:   
                metrics = self.agent.update(
                    self.replay_iter, self.global_step
                ) if self.global_step % self.cfg.update_every_steps == 0 else dict()
                if hasattr(self.agent, 'tp_set'):
                    metrics = self.agent.tp_set.log(metrics)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            episode_sr = episode_sr or time_step.success
            
            self.replay_storage.add(time_step)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self, step_id=None):
        if step_id is None:
            snapshot = self.work_dir / 'snapshot.pt'
        else:
            if not os.path.exists(str(self.work_dir) + '/snapshots'):
                os.makedirs(str(self.work_dir) + '/snapshots')
            snapshot = self.work_dir / 'snapshots' / 'snapshot_{}.pt'.format(step_id)
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, step_id=None):
        if step_id is None:
            snapshot = self.work_dir / 'snapshot.pt'
        else:
            snapshot = self.work_dir / 'snapshots' / 'snapshot_{}.pt'.format(step_id)
        if not snapshot.exists():
            raise FileNotFoundError(f"Snapshot {snapshot} not found.")
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfgs):
    # from train_mw_open import Workspace as W
    root_dir = Path.cwd()
    workspace = Workspace(cfgs)
    if cfgs.load_from_id:
        snapshot = root_dir / 'snapshots' / f'snapshot_{cfgs.load_id}.pt'
    else:
        snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()        
    workspace.train()


if __name__ == '__main__':
    main()
    # python train_mw_multi.py task@_global_=open_only seed=42 agent@_global_=mentor_mw device=cuda:4