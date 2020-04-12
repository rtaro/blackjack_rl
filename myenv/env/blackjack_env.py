import gym
import gym.spaces
import numpy as np

from myenv.env.blackjack import Game


class BlackJackEnv(gym.Env):
    metadata = {'render.mode': ['human', 'ansi']}

    def __init__(self):
        super().__init__()

        self.game = Game()
        self.game.start()

        # action_space, observation_space, reward_range を設定する
        self.action_space = gym.spaces.Discrete(4)  # hit, stand, double down, surrender

        high = np.array([
            30,  # player max
            30,  # dealer max
            1,   # is_soft_hand
            1,   # hit flag true
        ])
        low = np.array([
            2,  # player min
            1,  # dealer min
            0,  # is_soft_hand false
            0,  # hit flag false
        ])
        self.observation_space = gym.spaces.Box(low=low, high=high)
        self.reward_range = [-10000, 10000]  # 報酬の最小値と最大値のリスト

        self.done = False
        self.reset()

    def reset(self):
        # 状態を初期化し，初期の観測値を返す
        # 諸々の変数を初期化する
        self.done = False

        self.game.reset_game()
        self.game.bet(bet=100)
        self.game.player.chip.balance = 1000  # 学習中は所持金がゼロになることはないとする
        self.game.deal()
        # self.bet_done = True

        return self.observe()

    def step(self, action):
        # action を実行し，結果を返す
        # 1ステップ進める処理を記述．戻り値はobservation, reward, done（ゲーム終了したか）, info(追加の情報の辞書)

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
            print(self.observe())

        hit_flag_before_step = self.game.player.hit_flag
        self.game.player_step(action=action_str)

        if self.game.player.done:
            # プレーヤーのターンが終了したとき
            self.game.dealer_turn()
            self.game.judge()
            reward = self.get_reward()
            self.game.check_deck()
            print(str(self.game.judgment) + " : " + str(reward))


        elif action >= 2 and hit_flag_before_step is True:
            reward = -1e3  # ルールに反する場合はペナルティを与える

        else:
            # プレーヤーのターンを継続するとき
            reward = 0

        observation = self.observe()
        self.done = self.is_done()
        return observation, reward, self.done, self.game.player.chip.balance

    def render(self, mode='human', close=False):
        # 環境を可視化する
        # human の場合はコンソールに出力．ansi の場合は StringIO を返す
        pass

    def close(self):
        # 環境を閉じて，後処理をする
        pass

    def seed(self, seed=None):
        # ランダムシードを固定する
        pass

    def get_reward(self):
        # 報酬を返す
        reward = self.game.pay_chip() - self.game.player.chip.bet
        return reward

    def is_done(self):
        if self.game.player.done:
            return True
        else:
            return False

    def observe(self):
        if self.game.player.done:
            observation = tuple([
                self.game.player.hand.calc_final_point(),
                self.game.dealer.hand.calc_final_point(),  # Dealerのカードの合計点
                int(self.game.player.hand.is_soft_hand),
                int(self.game.player.hit_flag)])
        else:
            observation = tuple([
                self.game.player.hand.calc_final_point(),
                self.game.dealer.hand.hand[0].point,  # Dealerのアップカードのみ
                int(self.game.player.hand.is_soft_hand),
                int(self.game.player.hit_flag)])

        return observation

