from arguments import get_args
from Dagger import DaggerAgent, ExampleAgent
import numpy as np
import time
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import configparser
import os
import cv2
import random
import shutil
from multiprocessing import Process


def plot(record):
    with open('performance.txt', 'w+') as file:
        file.write(str(record))
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(record['steps'], record['mean'],
            color='blue', label='reward')
    ax.fill_between(record['steps'], record['min'], record['max'],
                    color='blue', alpha=0.2)
    ax.set_xlabel('number of steps')
    ax.set_ylabel('Average score per episode')
    ax1 = ax.twinx()
    ax1.plot(record['steps'], record['query'],
             color='red', label='query')
    ax1.set_ylabel('queries')
    reward_patch = mpatches.Patch(lw=1, linestyle='-', color='blue', label='score')
    query_patch = mpatches.Patch(lw=1, linestyle='-', color='red', label='query')
    patch_set = [reward_patch, query_patch]
    ax.legend(handles=patch_set)
    fig.savefig('performance.png')


# the wrap is mainly for speed up the game
# the agent will act every num_stacks frames instead of one frame
class Env(object):
    def __init__(self, env_name, num_stacks):
        self.env = gym.make(env_name)
        # num_stacks: the agent acts every num_stacks frames
        # it could be any positive integer
        self.num_stacks = num_stacks
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        reward_sum = 0
        for stack in range(self.num_stacks):
            obs_next, reward, done, info = self.env.step(action)
            reward_sum += reward
            if done:
                self.env.reset()
                return obs_next, reward_sum, done, info
        return obs_next, reward_sum, done, info

    def reset(self):
        return self.env.reset()


class ImgShow(Process):
    def __init__(self, img):
        super(ImgShow, self).__init__()
        self.img = img

    def run(self) -> None:
        cv2.imshow('Current Frame', self.img)
        cv2.waitKey()


def is_int(x):
    try:
        int(x)
        return True
    except ValueError:
        return False


def main():
    # load hyper parameters
    args = get_args()
    num_updates = int(args.num_frames // args.num_steps)
    start = time.time()
    record = {'steps': [0],
              'max': [0],
              'mean': [0],
              'min': [0],
              'query': [0]}

    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    data_index = config.getint('data', 'data_index')

    actions = [0, 1, 2, 3, 4, 5, 11, 12]
    key_map = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 11,
        7: 12,
        8: -1
    }

    # query_cnt counts queries to the expert
    query_cnt = 0

    # environment initial
    envs = Env(args.env_name, args.num_stacks)
    # action_shape is the size of the discrete action set, here is 18
    # Most of the 18 actions are useless, find important actions
    # in the tips of the homework introduction document
    action_shape = envs.action_space.n
    # observation_shape is the shape of the observation
    # here is (210,160,3)=(height, weight, channels)
    observation_shape = envs.observation_space.shape
    print(action_shape, observation_shape)

    # agent initial
    # you should finish your agent with DaggerAgent
    # e.g. agent = MyDaggerAgent()
    agent = ExampleAgent()

    # You can play this game yourself for fun
    if args.play_game:
        obs = envs.reset()
        while True:
            im = Image.fromarray(obs)
            im.save('imgs/' + str('screen') + '.jpeg')
            action = int(input('input action'))
            while action < 0 or action >= action_shape:
                action = int(input('re-input action'))
            obs_next, reward, done, _ = envs.step(action)
            obs = obs_next
            if done:
                obs = envs.reset()

    data_set = {'data': [], 'label': []}
    # start train your agent
    for i in range(data_index):
        data_path = 'data/data_batch_' + str(i) + '/'
        for j in range(args.num_steps):
            pic_path = data_path + str(j) + '.jpeg'
            data_set['data'].append(cv2.imread(pic_path))
        with open(data_path + 'label.txt', 'r') as f:
            for label_tmp in f.readlines():
                data_set['label'].append(int(label_tmp))
    agent.update(data_set['data'], data_set['label'])
    with open('performance.txt') as f:
        record_temp = eval(f.readline())
        if record_temp is not None:
            record = record_temp

    for i in range(data_index, num_updates):
        # an example of interacting with the environment
        # we init the environment and receive the initial observation
        obs = envs.reset()
        # we get a trajectory with the length of args.num_steps
        for step in range(args.num_steps):
            # Sample actions
            epsilon = 0.05
            if np.random.rand() < epsilon:
                # we choose a random action
                action = envs.action_space.sample()
            else:
                # we choose a special action according to our model
                action = agent.select_action(obs)

            # interact with the environment
            # we input the action to the environments and it returns some information
            # obs_next: the next observation after we do the action
            # reward: (float) the reward achieved by the action
            # down: (boolean)  whether it’s time to reset the environment again.
            #           done being True indicates the episode has terminated.
            obs_next, reward, done, _ = envs.step(action)
            # we view the new observation as current observation
            obs = obs_next
            # if the episode has terminated, we need to reset the environment.
            if done:
                envs.reset()

            # an example of saving observations
            if args.save_img:
                im = Image.fromarray(obs)
                im.save('imgs/' + str(step) + '.jpeg')
            data_set['data'].append(obs)

        # You need to label the images in 'imgs/' by recording the right actions in label.txt
        with open('imgs/label.txt', 'w+') as f:
            img_set = data_set['data'][-args.num_steps:]
            for img in img_set:
                cv2.imshow('Current Frame', img)
                cmd_in = cv2.waitKey(0) - 48
                while cmd_in not in key_map.keys():
                    pass
                cmd_in = key_map.get(cmd_in)
                print(cmd_in)
                if cmd_in is -1:
                    f.write(str(actions[random.randint(0, 7)]) + '\n')
                else:
                    f.write(str(cmd_in) + '\n')

        if not os.path.exists('data/data_batch_' + str(data_index) + '/'):
            shutil.copytree('./imgs', 'data/data_batch_' + str(data_index))
        data_index += 1
        config.set('data', 'data_index', str(data_index))
        config.write(open('config.ini', 'w'))
        # After you have labeled all the images, you can load the labels
        # for training a model
        with open('imgs/label.txt', 'r') as f:
            for label_tmp in f.readlines():
                data_set['label'].append(int(label_tmp))

        # design how to train your model with labeled data
        agent.update(data_set['data'], data_set['label'])

        if (i + 1) % args.log_interval == 0:
            total_num_steps = (i + 1) * args.num_steps
            obs = envs.reset()
            reward_episode_set = []
            reward_episode = 0
            # evaluate your model by testing in the environment
            for step in range(args.test_steps):
                action = agent.select_action(obs)
                # you can render to get visual results
                # envs.render()
                obs_next, reward, done, _ = envs.step(action)
                reward_episode += reward
                obs = obs_next
                if done:
                    reward_episode_set.append(reward_episode)
                    reward_episode = 0
                    envs.reset()
            if len(reward_episode_set) == 0:
                reward_episode_set.append(0)
            end = time.time()
            print(
                "TIME {} Updates {}, num timesteps {}, FPS {} \n query {}, avrage/min/max reward {:.1f}/{:.1f}/{:.1f}"
                    .format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)),
                    i, total_num_steps,
                    int(total_num_steps / (end - start)),
                    query_cnt,
                    np.mean(reward_episode_set),
                    np.min(reward_episode_set),
                    np.max(reward_episode_set)
                ))
            record['steps'].append(total_num_steps)
            record['mean'].append(np.mean(reward_episode_set))
            record['max'].append(np.max(reward_episode_set))
            record['min'].append(np.min(reward_episode_set))
            record['query'].append(query_cnt)
            plot(record)


if __name__ == "__main__":
    main()
