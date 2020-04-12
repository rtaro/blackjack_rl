import random
import time
import math
import copy
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import dfgui
import csv

import myenv
import gym

import json

from bokeh.events import ButtonClick
from bokeh.plotting import ColumnDataSource, figure, output_file, show, save, reset_output
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.models.widgets import Slider, Button, Select, RadioGroup, TextInput, TableColumn, DataTable, NumberFormatter
from bokeh.layouts import Column, Row, widgetbox, gridplot
from bokeh.io import curdoc

import seaborn as sns; sns.set()



class Agent():
    # EL\q_learning.pyを参考に書いた
    def __init__(self, epsilon):
        self.Q = {}
        self.epsilon = epsilon
        self.reward_log = []
        self.chip_log = []

    def policy(self, state, actions):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(actions))
        else:
            if state in self.Q and sum(self.Q[state]) != 0:
                return np.argmax(self.Q[state])
            else:
                return np.random.randint(len(actions))

    def init_log(self):
        self.reward_log = []
        self.chip_log = []

    def log(self, reward):
        self.reward_log.append(reward)

    def log_chip(self, chip):
        self.chip_log.append(chip)

    def show_reward_log(self, interval=100, episode=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print("At Episode {} average reward is {} (+/-{}).".format(episode, mean, std))
            with open('reward.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(rewards)
        else:
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:(i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            plt.figure()
            plt.title("Reward History")
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds, alpha=0.4, color="g")
            plt.plot(indices, means, "o-", color="g", label="Rewards for each {} episode".format(interval))
            plt.legend(loc="best")
            plt.savefig("Reward_History.png")
            plt.show()


            self.plot_chip_history(interval=interval)

            # reset_output()
            # output_file("reward_history.html")
            # tools = "pan, box_zoom, lasso_select, box_select, poly_select, tap, wheel_zoom, save, reset"
            # p1 = figure(title="Reward History", x_axis_label='Episodes', y_axis_label='Rewards',
            #             plot_width=800, plot_height=500, tools=tools)
            # p1.quad(bottom=means - stds, top=means + stds, left=min(indices), right=max(indices),
            #         color="green", alpha=0.2)
            # p1.line(x=indices, y=means, legend_label='Rewards for each {} episode'.format(interval),
            #         line_color="green", line_alpha=1.0)
            # p1.circle(x=indices, y=means, legend_label='Rewards for each {} episode'.format(interval),
            #           color="green", fill_alpha=0.5, size=6)
            # # 凡例をクリックしたときにプロットを非表示にする
            # p1.legend.click_policy = "hide"
            # layout = Row(p1)
            # curdoc().add_root(layout)
            # show(layout)

    def draw_reward_history(self, episode, history):
        fig = plt.figure()
        plt.title("Reward History at episode {}".format(episode))
        plt.grid()
        indices = list(range(0, len(history)))
        plt.plot(indices, history, "o-", color="g", label="Chip for each game")
        plt.legend(loc="best")
        plt.savefig("fig/reward_history_{}".format(episode))
        plt.close(fig)
        with open('reward_history.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(history)

    def plot_chip_history(self, interval):
        indices = list(range(0, len(self.chip_log), interval))
        means = []
        stds = []
        for i in indices:
            rewards = self.chip_log[i:(i + interval)]
            means.append(np.mean(rewards))
            stds.append(np.std(rewards))
        means = np.array(means)
        stds = np.array(stds)
        plt.figure()
        plt.title("Chip History")
        plt.grid()
        plt.fill_between(indices, means - stds, means + stds, alpha=0.1, color="g")
        plt.plot(indices, means, "o-", color="g", label="Chips for each {} episode".format(interval))
        plt.legend(loc="best")
        plt.savefig("Chip_History.png")
        plt.show()

    def save_Q(self):
        # f = open('Q_table.json', 'w')
        # # json.dump関数でファイルに書き込む
        # json.dump(self.Q, f, indent=4)

        # print(self.Q)
        # print(type(self.Q))


        new_Q = sorted(list(self.Q.items()))
        # print(new_Q)

        Q_df, Q_hard_df, Q_soft_df, Q_sa_df = self.make_Q_table()
        # print(Q_df)
        Q_df.to_csv('./Q_table.csv')
        Q_hard_df.to_csv('./Q_hard_table.csv')
        Q_soft_df.to_csv('./Q_soft_table.csv')
        Q_sa_df.to_csv(('./Q_sa_table.csv'))

        # DataFrameを可視化
        # dfgui.show(Q_df)
        # dfgui.show(Q_hard_df)
        # dfgui.show(Q_soft_df)


    def make_Q_table(self):
        # 2次元配列データに加工
        Q_df = pd.DataFrame(columns=['Player', 'Dealer', 'Ace', 'Hit Flag', 'Stand', 'Hit', 'Double Down', 'Surrender',
                                     'Best Action Index', 'Best Action'])
        action_name = ('S', 'H', 'D', 'Su')  # ('Stand', 'Hit', 'Double Down', 'Surrender')
        Q_hard_df = pd.DataFrame(columns=['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A'],
                                         index=['4', '5', '6', '7', '8', '9', '10', '11', '12',
                                                '13', '14', '15', '16', '17', '18', '19', '20', '21'])
        Q_soft_df = pd.DataFrame(columns=['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A'],
                                         index=['4', '5', '6', '7', '8', '9', '10', '11', '12',
                                                '13', '14', '15', '16', '17', '18', '19', '20', '21'])
        Q_sa_df = pd.DataFrame(columns=['State', 'Action'])

        # p : Playerのポイント
        # d : Dealerのポイント
        # ace : Aceを含む→1，Aceを含まない→0
        for p in range(4,22):
            for d in range(1,11):
                for ace in range(0,2):
                    for hit_flag in range(0, 2):

                        key = tuple(np.array([p, d, ace, hit_flag])) # '(' + p + ',' + d + ',' + ace + ')'
                        if key in self.Q:
                            best_action_index = np.argmax([self.Q[key][0], self.Q[key][1], self.Q[key][2], self.Q[key][3]])
                            best_action_name = action_name[best_action_index]
                            tmp_se = pd.Series([p, d, ace, hit_flag,
                                                self.Q[key][0], self.Q[key][1], self.Q[key][2], self.Q[key][3],
                                                best_action_index, best_action_name], index=Q_df.columns)

                            Q_df = Q_df.append(tmp_se, ignore_index=True)

                            tmp_se_sa = pd.Series([(p, d, ace, hit_flag), best_action_index], index=Q_sa_df.columns)
                            Q_sa_df = Q_sa_df.append(tmp_se_sa, ignore_index=True)
                            # if ace == 0:
                            #     Q_hard_hand_table[int(p), int(d)] = best_action_name
                            # else:
                            #     Q_soft_hand_table[int(p), int(d)] = best_action_name
                        else:
                            tmp_se = pd.Series([p, d, ace, hit_flag, None, None, None, None, None, None],
                                               index=Q_df.columns)
                            Q_df = Q_df.append(tmp_se, ignore_index=True)

                            tmp_se_sa = pd.Series([(p, d, ace, hit_flag), None], index=Q_sa_df.columns)
                            Q_sa_df = Q_sa_df.append(tmp_se_sa, ignore_index=True)

        for p in range(4, 21):
            # Pandasでは&, |, ~を使う．and, or, notは使えない
            bool_list = (Q_df['Player'] == p) & (Q_df['Ace'] == 0) & (Q_df['Hit Flag'] == 0)
            temp1 = Q_df[bool_list]['Best Action']
            temp2 = np.append(temp1.values[1:10], temp1.values[0])
            Q_hard_df.loc[str(p)] = pd.Series(temp2).values #.values #  Q_df.values[bool_list]['Best Action']

        for p in range(12, 22):
            # Pandasでは&, |, ~を使う．and, or, notは使えない
            bool_list = (Q_df['Player'] == p) & (Q_df['Ace'] == 1) & (Q_df['Hit Flag'] == 0)
            temp1 = Q_df[bool_list]['Best Action']
            temp2 = np.append(temp1.values[1:10], temp1.values[0])
            Q_soft_df.loc[str(p)] = pd.Series(temp2).values  # Q_df.values[bool_list]['Best Action']

        return Q_df, Q_hard_df, Q_soft_df, Q_sa_df

    def make_strategy_table(self, Q_df):
        Q_hard_df = pd.DataFrame({'2' : Q_df['2']})

    def plot_Q(self, hard, soft):
        # Hard hand のBest Action
        hard_hand_map = sns.heatmap(hard.iloc[4:, :], annot=True, fmt=".0f", linewidths=.5, linecolor="Black",
                                    cbar=False, cmap="Paired", )
        plt.figure()
        # Soft hand のBest Action
        soft_hand_map = sns.heatmap(soft.iloc[12:, :], annot=True, fmt=".0f", linewidths=.5, linecolor="Black",
                                    cbar=False, cmap="Paired", )

    def load_Q_table(self):
        Q_sa = pd.read_csv('./Q_sa_table.csv', index_col=1)  # State をindexにする
        return Q_sa

    def load_basic_strategy(self):
        basic_strategy = pd.read_csv('./basic_strategy_table.csv', index_col=1)  # State をindexにする
        return basic_strategy









class QLearningAgent(Agent):
    # EL\q_learning.pyを参考に書いた
    def __init__(self, epsilon=0.1):  # もともとepsilon=0.1
        super().__init__(epsilon)

    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=5000):
        self.init_log()
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda: [0] * len(actions))
        for e in range(episode_count):
            s = env.reset()
            # print(s)
            done = False
            reward_history = []
            while not done:
                if render:
                    env.render()
                print(s)
                a = self.policy(s, actions)
                # print("action is : " + str(a))
                n_state, reward, done, info = env.step(a)
                # print("episode : " + str(e) +
                #       ", n_state : " +  str(n_state) +
                #       ", reward : " + str(reward) +
                #       ", done : " + str(done) +
                #       ", game count : " + str(info))

                reward_history.append(reward)
                gain = reward + gamma * max(self.Q[n_state])
                estimated = self.Q[s][a]
                self.Q[s][a] += learning_rate * (gain - estimated)
                s = n_state
            else:
                self.log(np.round(np.sum(reward_history)))
                self.log_chip(chip=info)
                # print(info)

            if e != 0 and e % report_interval == 0:
                # print("e= " + str(e))
                self.show_reward_log(episode=e, interval=50)
        self.draw_reward_history(episode=e, history=reward_history)

        env.close()

    def play(self, env, episode_count=10, render=False, report_interval=1):
        self.init_log()

        # load basic strategy
        strategy = self.load_basic_strategy()
        # load Q-learning result
        # strategy = self.load_Q_table()
        for e in range(episode_count):
            print("episode:" + str(e))
            s = env.reset()
            done = False
            reward_history = []
            while not done:
                if render:
                    env.render()
                print(s)
                a = strategy.at[str(s), 'Action']
                if a == np.nan:
                    print("action is None" + " state:" + str(s))
                if a >= 2 and s[3] == 1:
                    a = 1
                    print("--------------Double down/Surrender できません．Hitします-------------------")
                n_state, reward, done, info = env.step(a)
                reward_history.append(reward)
                s = n_state
            else:
                self.log(np.round(np.sum(reward_history)))
                self.log_chip(chip=info)

            # self.draw_reward_history(episode=e, history=reward_history)
        print(str(episode_count) + "エピソードの平均獲得チップは" + str(np.mean(self.reward_log)) + "です")










def train():
    agent = QLearningAgent()
    env = gym.make('BlackJack-v0')
    agent.learn(env, episode_count=20000, report_interval=200)
    agent.save_Q()
    agent.show_reward_log(interval=200)

def play():
    agent = QLearningAgent()
    env = gym.make('BlackJack-v0')
    agent.play(env, episode_count=1000, report_interval=1)
    agent.show_reward_log(interval=1)

def test():
    agent = QLearningAgent()
    t = agent.load_Q_table()
    print(t)


if __name__ == "__main__":
    train()
    # play()
