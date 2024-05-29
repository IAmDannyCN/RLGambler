#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
from copy import deepcopy
from stable_baselines3 import PPO

class DiceGameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DiceGameEnv, self).__init__()
        self.action_space = gym.spaces.MultiBinary(5)
        self.observation_space = gym.spaces.MultiDiscrete([7]*11)
        self.verbose = False
        self.finetune = False
        self.role = ''
        self.oppo_model = None
        self.last_overhead = -1
        self.score_table = {66666 : 2250, 55555 : 2249, 44444 : 2248, 33333 : 2247, 22222 : 2246, 11111 : 2245, 56666 : 1444, 46666 : 1443, 36666 : 1442, 26666 : 1441, 16666 : 1440, 55556 : 1439, 45555 : 1438, 35555 : 1437, 25555 : 1436, 15555 : 1435, 44446 : 1434, 44445 : 1433, 34444 : 1432, 24444 : 1431, 14444 : 1430, 33336 : 1429, 33335 : 1428, 33334 : 1427, 23333 : 1426, 13333 : 1425, 22226 : 1424, 22225 : 1423, 22224 : 1422, 22223 : 1421, 12222 : 1420, 11116 : 1419, 11115 : 1418, 11114 : 1417, 11113 : 1416, 11112 : 1415, 23456 : 1214, 12345 : 1213, 55666 : 612, 44666 : 611, 33666 : 610, 22666 : 609, 11666 : 608, 55566 : 607, 44555 : 606, 33555 : 605, 22555 : 604, 11555 : 603, 44466 : 602, 44455 : 601, 33444 : 600, 22444 : 599, 11444 : 598, 33366 : 597, 33355 : 596, 33344 : 595, 22333 : 594, 11333 : 593, 22266 : 592, 22255 : 591, 22244 : 590, 22233 : 589, 11222 : 588, 11166 : 587, 11155 : 586, 11144 : 585, 11133 : 584, 11122 : 583, 45666 : 382, 35666 : 381, 25666 : 380, 15666 : 379, 34666 : 378, 24666 : 377, 14666 : 376, 23666 : 375, 13666 : 374, 12666 : 373, 45556 : 372, 35556 : 371, 25556 : 370, 15556 : 369, 34555 : 368, 24555 : 367, 14555 : 366, 23555 : 365, 13555 : 364, 12555 : 363, 44456 : 362, 34446 : 361, 24446 : 360, 14446 : 359, 34445 : 358, 24445 : 357, 14445 : 356, 23444 : 355, 13444 : 354, 12444 : 353, 33356 : 352, 33346 : 351, 23336 : 350, 13336 : 349, 33345 : 348, 23335 : 347, 13335 : 346, 23334 : 345, 13334 : 344, 12333 : 343, 22256 : 342, 22246 : 341, 22236 : 340, 12226 : 339, 22245 : 338, 22235 : 337, 12225 : 336, 22234 : 335, 12224 : 334, 12223 : 333, 11156 : 332, 11146 : 331, 11136 : 330, 11126 : 329, 11145 : 328, 11135 : 327, 11125 : 326, 11134 : 325, 11124 : 324, 11123 : 323, 45566 : 222, 35566 : 221, 25566 : 220, 15566 : 219, 44566 : 218, 34466 : 217, 24466 : 216, 14466 : 215, 33566 : 214, 33466 : 213, 23366 : 212, 13366 : 211, 22566 : 210, 22466 : 209, 22366 : 208, 12266 : 207, 11566 : 206, 11466 : 205, 11366 : 204, 11266 : 203, 44556 : 202, 34455 : 201, 24455 : 200, 14455 : 199, 33556 : 198, 33455 : 197, 23355 : 196, 13355 : 195, 22556 : 194, 22455 : 193, 22355 : 192, 12255 : 191, 11556 : 190, 11455 : 189, 11355 : 188, 11255 : 187, 33446 : 186, 33445 : 185, 23344 : 184, 13344 : 183, 22446 : 182, 22445 : 181, 22344 : 180, 12244 : 179, 11446 : 178, 11445 : 177, 11344 : 176, 11244 : 175, 22336 : 174, 22335 : 173, 22334 : 172, 12233 : 171, 11336 : 170, 11335 : 169, 11334 : 168, 11233 : 167, 11226 : 166, 11225 : 165, 11224 : 164, 11223 : 163, 34566 : 112, 24566 : 111, 14566 : 110, 23566 : 109, 13566 : 108, 12566 : 107, 23466 : 106, 13466 : 105, 12466 : 104, 12366 : 103, 34556 : 102, 24556 : 101, 14556 : 100, 23556 : 99, 13556 : 98, 12556 : 97, 23455 : 96, 13455 : 95, 12455 : 94, 12355 : 93, 34456 : 92, 24456 : 91, 14456 : 90, 23446 : 89, 13446 : 88, 12446 : 87, 23445 : 86, 13445 : 85, 12445 : 84, 12344 : 83, 33456 : 82, 23356 : 81, 13356 : 80, 23346 : 79, 13346 : 78, 12336 : 77, 23345 : 76, 13345 : 75, 12335 : 74, 12334 : 73, 22456 : 72, 22356 : 71, 12256 : 70, 22346 : 69, 12246 : 68, 12236 : 67, 22345 : 66, 12245 : 65, 12235 : 64, 12234 : 63, 11456 : 62, 11356 : 61, 11256 : 60, 11346 : 59, 11246 : 58, 11236 : 57, 11345 : 56, 11245 : 55, 11235 : 54, 11234 : 53, 13456 : 52, 12456 : 51, 12356 : 50, 12346 : 49, }



    def reset(self, seed=None, options=None):
        self.state = np.random.randint(1, 7, size=5+5)
        self.rolls_left = 2
        if self.finetune == True and self.role == 'B':
            oppo_obs = np.append(np.concatenate((self.state[5:10], self.state[0:5])), self.rolls_left)
            oppo_action, _states = self.oppo_model.predict(oppo_obs, deterministic=True)
            for i in range(5):
                if oppo_action[i] == 1:
                    self.state[i] = np.random.randint(1, 7)
        return np.append(self.state, self.rolls_left), {}

    def step(self, action):
        assert self.rolls_left > 0, "No rolls left"
        if self.verbose:
            print("Action:", action)
        
        if not self.finetune:
            # Pretrain, target is higher score
            for i in range(5):
                if action[i] == 1:
                    self.state[i] = np.random.randint(1, 7)
            self.rolls_left -= 1

            done = self.rolls_left == 0
            reward = self.calculate_score(self.state[0:5]) if done else 0

            return np.append(self.state, self.rolls_left), reward, done, False, {}
        else:
            # Finetune, target is win over the opposite
            if self.role == 'A':
                # I play first
                for i in range(5):
                    if action[i] == 1:
                        self.state[i] = np.random.randint(1, 7)
                
                oppo_obs = np.append(np.concatenate((self.state[5:10], self.state[0:5])), self.rolls_left)
                oppo_action, _states = self.oppo_model.predict(oppo_obs, deterministic=True)
                if self.verbose:
                    print(f"Advact: {oppo_action}")
                for i in range(5):
                    if oppo_action[i] == 1:
                        self.state[i+5] = np.random.randint(1, 7)
                
                self.rolls_left -= 1
                done = self.rolls_left == 0
                
                compete_reward = self.compete(self.state[0:5], self.state[5:10]) if done else 0
                overhead = self.calculate_score(self.state[0:5]) >= self.calculate_score(self.state[5:10])
                if not done:
                    reward = 0
                    self.last_overhead = overhead
                else:
                    reward = overhead - self.last_overhead + compete_reward
                
                return np.append(self.state, self.rolls_left), reward, done, False, {}
            
            elif self.role == 'B':
                # The opposite play first
                for i in range(5):
                    if action[i] == 1:
                        self.state[i] = np.random.randint(1, 7)
                
                self.rolls_left -= 1
                done = self.rolls_left == 0
                
                compete_reward = self.compete(self.state[0:5], self.state[5:10]) if done else 0
                overhead = self.calculate_score(self.state[0:5]) >= self.calculate_score(self.state[5:10])
                if not done:
                    reward = 0
                    self.last_overhead = overhead
                else:
                    reward = overhead - self.last_overhead + compete_reward
                
                if not done:
                    oppo_obs = np.append(np.concatenate((self.state[5:10], self.state[0:5])), self.rolls_left)
                    oppo_action, _states = self.oppo_model.predict(oppo_obs, deterministic=True)
                    for i in range(5):
                        if oppo_action[i] == 1:
                            self.state[i+5] = np.random.randint(1, 7)

                return np.append(self.state, self.rolls_left), reward, done, False, {}
            else:
                assert self.role == 'A' or self.role == 'B'

    def calculate_score(self, state):
        return self.score_table[int(''.join(map(str, sorted(state))))]
    
    def compete(self, state1, state2):
         return 100 if self.calculate_score(state1) >= self.calculate_score(state2) else -200

    def render(self, mode='human'):
        print(f"Dice states: {self.state}, score: {self.calculate_score(self.state[0:5])}")
        
    def judge(self):
        # print(f"Winning State: {self.calculate_score(self.state[0:5]) >= self.calculate_score(self.state[5:10])}")
        return self.calculate_score(self.state[0:5]) >= self.calculate_score(self.state[5:10])

    def close(self):
        pass
    
    def set_state(self, state):
        self.state = state
    
    def set_rolls_left(self, rolls_left):
        self.rolls_left = rolls_left


def pre_train():
    env = DiceGameEnv()
    
    model = PPO("MlpPolicy", env, verbose=1, device='cuda')
    model.learn(total_timesteps=262144)
    model.save("pretrain_model")
   
def fine_tune():
    A_env = DiceGameEnv()
    B_env = DiceGameEnv()
    A_env.role = 'A'
    B_env.role = 'B'
    
    A_model = PPO.load("pretrain_model", env=A_env, device='cuda')
    B_model = PPO.load("pretrain_model", env=B_env, device='cuda')
    for T in range(10):
        print(f"Training Epoch: {T}")
        print(f"Finetune ON")
        A_env.finetune = True
        B_env.finetune = True
        A_env.oppo_model = B_model
        A_model.learn(total_timesteps=16384)
        B_env.oppo_model = A_model
        B_model.learn(total_timesteps=16384)
       
        print(f"Finetune OFF")
        A_env.finetune = False
        B_env.finetune = False
        A_model.learn(total_timesteps=8192)
        B_model.learn(total_timesteps=8192)
    
    A_model.save("A_model")
    B_model.save("B_model")

def evaluate(T=10000):
    A_model = PPO.load("A_model")
    B_model = PPO.load("B_model")
    
    win_cnt = 0
    for t in range(T):
        print(f"Round {t}")
        test_env = DiceGameEnv()
        test_env.verbose = False
        test_env.finetune = True
        test_env.role = 'A'
        test_env.oppo_model = B_model
        obs, info = test_env.reset()

        for _ in range(2):
            action, _states = A_model.predict(obs, deterministic=True)
            obs, rewards, dones, truncated, info = test_env.step(action)
            if dones:
                break
        win_cnt += test_env.judge()
    print(f"Winning Possibility: {100 * win_cnt / T}%")


def run_demo(T=20):
    A_model = PPO.load("A_model")
    B_model = PPO.load("B_model")
    
    for t in range(T):
        print(f"Round {t}")
        test_env = DiceGameEnv()
        test_env.verbose = True
        test_env.finetune = True
        test_env.role = 'A'
        test_env.oppo_model = B_model
        obs, info = test_env.reset()

        test_env.render()
        for _ in range(2):
            action, _states = A_model.predict(obs, deterministic=True)
            obs, rewards, dones, truncated, info = test_env.step(action)
            test_env.render()
            if dones:
                break
        print("[WIN]" if test_env.judge() else "[LOSE]")
        print()


if __name__ == '__main__':
    # pre_train()
    # fine_tune()
    # evaluate(1000)
    run_demo()