import random
import copy

import gym
import gym.spaces
import numpy as np


import myenv.blackjack
from myenv.blackjack import Game


class BlackJackEnv(gym.Env):
    metadata = {'render.mode': ['human', 'ansi']}

    MAX_GAME_COUNT = 10  # 1エピソード内で行う最大のゲーム数

    def __init__(self):
        super().__init__()

        self.game = Game()
        self.game.start()

        # action_space, observation_space, reward_range を設定する
        self.action_space = gym.spaces.Discrete(4)  # hit, stand, double down, surrender

        high = np.array([
            30,  # player max
            11,  # dealer max
            1,   # is_soft_hand
            1,   # after hit
        ])
        low = np.array([
            2,  # player min
            1,  # dealer min
            0,  # is_soft_hand
            0,  # before hit
        ])
        self.observation_space = gym.spaces.Box(low=low, high=high)
        self.reward_range = [-10000, 10000]  # 報酬の最小値と最大値のリスト
        self._reset()

    def _reset(self):
        # 状態を初期化し，初期の観測値を返す
        # 諸々の変数を初期化する
        self.done = False
        self.bet_done = False
        # self.steps = 0

        self.chip = 0  # self.game.player.chip.balance

        self.game.start()  # game_mode=1にする

        if self.bet_done == False:
            self.game.reset_game()
            self.game.bet(bet=100)
            self.game.deal()
            self.bet_done = True

        return self._observe()

    def _step(self, action):
        # action を実行し，結果を返す
        # 1ステップ進める処理を記述．戻り値はobservation, reward, done（ゲーム終了したか）, info(追加の情報の辞書
        # )

        if action == 0:
            action_str = 's'  # Stand
        elif action == 1:
            action_str = 'h'  # Hit
        elif action == 2:
            action_str = 'd'  # Double down
        elif action == 3:
            action_str = 'r'  # Surrender
        else:
            print(action)
            print("未定義のActionです")
            print(self._observe())

        hit_flag_before_step = self.game.player.hit_flag
        self.game.player_step(action=action_str)

        if self.game.player.done:
            # プレーヤーのターンが終了したとき
            self.game.dealer_turn()
            self.game.judge()
            acquired_chip = self.game.pay_chip()
            self.game.check_chip()
            reward = self._get_reward(acquired_chip)
            # self.game.ask_next_game()
            self.game.check_deck()
            self.bet_done = False
            if self.bet_done == False:
                self.game.reset_game()
                self.game.bet(bet=100)
                self.game.deal()
                self.bet_done = True

        elif action >= 2 and hit_flag_before_step is True:
            reward = -1e5  # ルールに反する場合はペナルティを与える

        else:
            # プレーヤーのターンを継続するとき
            reward = 0  #self._get_reward()

        observation = self._observe()

        # if action == 2 or action == 3:
        #     print(action, reward)

        self.done = self._is_done()
        return observation, reward, self.done, self.game.player.chip.balance

    def _render(self, mode='human', close=False):
        # 環境を可視化する
        # human の場合はコンソールに出力．ansi の場合は StringIO を返す
        pass
        # outfile = StringIO() if mode == 'ansi' else sys.stdout
        # outfile.write('\n'.join( for row in self._observe) + '\n')

        # return outfile

    def _close(self):
        # 環境を閉じて，後処理をする
        pass

    def _seed(self, seed=None):
        # ランダムシードを固定する
        pass

    def _get_reward(self, acquired_chip):
        # 報酬を返す
        reward = acquired_chip - self.game.player.chip.bet #.game.player.chip.balance
        if reward < 0:
            reward = reward * 1
        return reward

    def _is_done(self):
        if self.game.game_mode == 2:  # チップがMINIMUM_CHIPを下回ったらgame_mode=2にする
            return True
        elif self.game.game_count > self.MAX_GAME_COUNT:
            return True
        else:
            return False

    def _observe(self):
        # observation は object であること
        # if self.bet_done == True:
        #     # self.player_point = self.game.player.hand.sum_point()
        #     # self.dealer_up_card = self.game.dealer.hand.hand[0].point
        #     self.observation = np.array([
        #         self.game.player.hand.calc_final_point(),
        #         self.game.dealer.hand.hand[0].point,
        #         self.game.player.hand.is_soft_hand])
        # else:
        #     # self.player_point = 0
        #     # self.dealer_up_card = 0
        #     self.observation = np.array([0,
        #                                  0,
        #                                  self.game.player.hand.is_soft_hand])
        # self.observation = [self.player_point, self.dealer_up_card]

        if self.game.player.done:
            self.observation = np.array([
                self.game.player.hand.calc_final_point(),
                self.game.dealer.hand.calc_final_point(),
                self.game.player.hand.is_soft_hand,
                self.game.player.hit_flag])
        else:
            self.observation = np.array([
                self.game.player.hand.calc_final_point(),
                self.game.dealer.hand.hand[0].point,
                self.game.player.hand.is_soft_hand,
                self.game.player.hit_flag])

        observation = tuple(self.observation)

        return observation

