import random
import copy

import gym
import numpy as np
import gym.spaces

# 定数
NUM_DECK = 6  # デッキ数
NUM_PLAYER = 1  # プレイヤー数

INITIAL_CHIP = 1000  # 初期チップ
MINIMUM_BET = 100


class Card:
    '''
    カードを生成
    数字：A，２～１０，J，Q，K
    スート：スペード，ハート，ダイヤ，クラブ
    '''
    SUITS = '♠♥♦♣'
    RANKS = range(1, 14)  # 通常のRank
    SYMBOLS = "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"
    POINTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]  # BlackJack用のポイント

    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
        self.index = suit + self.SYMBOLS[rank - self.RANKS[0]]
        self.point = self.POINTS[rank - self.RANKS[0]]

    def __repr__(self):
        return self.index


class Deck:
    '''
    カードがシャッフルされたデッキ（山札を生成）
    '''
    def __init__(self):
        self.cards = [Card(suit, rank) \
            for suit in Card.SUITS \
            for rank in Card.RANKS]

        if NUM_DECK > 1:
            temp_cards = copy.deepcopy(self.cards)
            for i in range(NUM_DECK - 1):
                self.cards.extend(temp_cards)
        random.shuffle(self.cards)

    def draw(self, n=1):
        '''
        デッキから指定した枚数分だけ引く関数
        '''
        cards = self.cards[:n]
        del self.cards[:n]  # 引いたカードを山札から削除
        return cards

    def shuffle(self):
        '''
        デッキをシャッフルする
        '''
        random.shuffle(self.cards)
        return

    def count_cards(self):
        """
        デッキの残り枚数を返す
        """
        count = len(self.cards)
        return count
    '''
    def sum_rank(self):
        s = 0
        for i in range(len(self.cards)):
            s += self.cards[i].rank
        return s

    def calc_rank_expected_value(self):
        s = 0
        for i in range(len(self.cards)):
            s += self.cards[i].rank
        expected = s/self.count_cards()
        return expected
    '''


class Hand:
    """
    手札クラス
    """
    def __init__(self):
        self.hand = []
        self.is_soft_hand = False

    def add_card(self, card):
        self.hand.append(card)
    '''
    def sum_rank(self):
        s = 0
        for i in range(len(self.hand)):
            s += self.hand[i].rank
        return s
    '''
    def check_soft_hand(self):
        """
        ソフトハンド（Aを含むハンド）かチェックする
        """
        hand_list = []
        for i in range(len(self.hand)):
            hand_list.append(self.hand[i].point)
        hand_list.sort()  # 手札を昇順にソート

        if hand_list[0] == 1 and sum(hand_list[1:]) < 11:  # ソフトハンドなら
            self.is_soft_hand = True
        else:
            self.is_soft_hand = False

    def sum_point(self):
        """
        ポイントの合計値を返す
        """
        self.check_soft_hand()
        hand_list = []

        for i in range(len(self.hand)):
            hand_list.append(self.hand[i].point)
        hand_list.sort()  # 手札を昇順にソート
        s1 = 0  # Aを1とカウントする場合の初期値
        for i in range(len(self.hand)):
            s1 += self.hand[i].point

        if self.is_soft_hand == True:  # ソフトハンドなら
            s2 = 11  # 1枚目のAを11としてカウント
            for i in range(len(hand_list)-1):
                s2 += hand_list[i+1]
            s = [s1, s2]
        else:
            s = [s1]

        return s

    def calc_final_point(self):
        """
        BUSTしていない最終的なポイントを返す
        """
        temp_point = self.sum_point()
        if max(temp_point) > 22:
            p = temp_point[0]  # ポイントの大きい方がBUSTなら小さい方
        else:
            p = max(temp_point)
        return p

    def is_bust(self):
        """
        BUSTかどうか判定する
        """
        if min(self.sum_point()) > 21:  # ポイントの小さい方が21を超えていたら
            return True
        else:
            return False

    def deal(self, card):
        """
        Dealされたカードを手札に加える
        """
        for i in range(len(card)):
            self.add_card(card[i])

    def hit(self, card):
        # 1枚ずつHitする
        if len(card) == 1:
            self.add_card(card[0])
        else:
            print("カードの枚数が正しくありません")


class Player:
    def __init__(self):
        self.hand = Hand()
        self.chip = Chip()
        self.done = False
        self.hit_flag = False  # Player が Hit を選択済みかどうか示すフラグ
        self.is_human = False  # True:人がプレイ，False:自動プレイ

    def init_player(self):
        self.hand = Hand()
        self.done = False
        self.hit_flag = False

    def deal(self, card):
        self.hand.deal(card)

    def hit(self, card):
        # カードを1枚手札に加える
        self.hand.hit(card)
        self.hit_flag = True

    def stand(self):
        # ターン終了
        self.done = True

    def double_down(self, card):
        # betを2倍にして一度だけHitしてターン終了
        self.chip.balance -= self.chip.bet
        self.chip.bet = self.chip.bet * 2
        self.hand.hit(card)
        self.done = True  # double down後は一度しかhitできないルールとする

    def surrender(self):
        # Betを半分にして（betの半分を手持ちに戻して）ターン終了
        # self.chip.balance += int(self.chip.bet / 2)
        self.chip.bet = int(self.chip.bet / 2)
        self.chip.balance += self.chip.bet
        self.done = True

    def insurance(self):
        # 未実装
        pass

    def split(self):
        # 未実装
        pass


class Dealer:
    def __init__(self):
        self.hand = Hand()

    def init_dealer(self):
        self.hand = Hand()

    def deal(self, card):
        self.hand.deal(card)

    def hit(self, card):
        self.hand.hit(card)


class Chip:
    def __init__(self):
        self.balance = INITIAL_CHIP
        self.bet = 0

    def bet_chip(self, bet):
        self.balance -= bet
        self.bet = bet

    def pay_chip_win(self):
        self.balance += self.bet * 2

    def pay_chip_lose(self):
        self.balance = self.balance

    def pay_chip_push(self):
        self.balance += self.bet


class AI:
    def select_random1(self):
        r = random.randint(0, 1)
        if r > 0.5:
            selection = 'h'  # hit
        else:
            selection = 's'  # stand

        return selection

    def select_random2(self, hand):
        if hand <= 11:
            selection = 'h'
        else:
            r = random.randint(0, 1)
            if r > 0.5:
                selection = 'h'  # hit
            else:
                selection = 's'  # stand

        return selection

    def select_random3(self, hand, n):
        if hand < 11:
            selection = 'h'  # hit
        elif hand == 11 and n == 2:
            selection = 'd'  # double down
        elif hand == 16 and n == 2:
            selection = 'r'  # surrender
        elif hand > 17:
            selection = 's'  # stand
        else:
            r = random.randint(0, 1)
            if r > 0.5:
                selection = 'h'  # hit
            else:
                selection = 's'  # stand

        return selection


class Game:
    def __init__(self):
        self.game_mode = 0  # 0:開始待ち，1:ゲーム中, 2:ゲーム終了
        self.deck = Deck()
        self.player = Player()
        self.dealer = Dealer()
        self.judgment = 0
        self.game_count = 0
        self.start()

        self.message_on = False #self.player.is_human  # True:コンソールにメッセージ表示する，False:コンソールにメッセージ表示しない

    def start(self):
        self.deck.shuffle()
        self.game_mode = 1
        self.player = Player()
        self.dealer = Dealer()
        self.game_count = 0

    def reset_game(self):
        self.player.init_player()
        self.dealer.init_dealer()
        self.game_count += 1

    def bet(self, bet):
        self.player.chip.bet_chip(bet=bet)
        if self.message_on:
            print("$" + str(self.player.chip.bet) + " 賭けました")
            print("残りは $" + str(self.player.chip.balance))

    # カードを配る
    def deal(self, n=2):
        '''
        カードを配る
        '''
        card = self.deck.draw(n)
        self.player.deal(card)
        # print(self.player.hand.hand)

        card = self.deck.draw(n)
        self.dealer.deal(card)
        # print(self.dealer.hand.hand)

        self.judgment = 0   # 勝敗の判定
        self.player.done = False
        self.show_card()

    # Playerのターン
    def player_turn(self):
        '''
        プレーヤーのターン
        '''
        if self.player.hand.calc_final_point() == 21:  # 合計が21だったらすぐにDealerのターンへ
            self.player.done = True

        while not self.player.done and not self.player.hand.is_bust():
            if self.player.is_human is True:
                action = input("Hit(h) or Stand(s) or Double down(d) or Surrender(r): ")
            elif self.player.is_human is True and self.player.hit_flag:
                action = input("Hit(h) or Stand(s): ")  # Hitの後はhit/standのみ
            else:
                action = AI().select_random3(hand=self.player.hand.calc_final_point(), n=len(self.player.hand.hand))

            self.player_step(action=action)

    def player_step(self, action):
        if action == 'h':  # Hit
            card = self.deck.draw(1)
            self.player.hit(card)
            self.show_card()
            if self.player.hand.calc_final_point() == 21:  # 合計点が21になったらこれ以上Hitはできない
                self.player.done = True
            if self.player.hand.is_bust():
                self.player.done = True
                self.judgment = -1  # PlayerがBUSTしたら即負け
                if self.message_on:
                    print("Player BUST")

        elif action == 's':  # Stand
            self.player.stand()

        elif action == 'd' and self.player.hit_flag is False:  # Double down. Hit選択していない場合に可
            card = self.deck.draw(1)
            if self.player.chip.balance >= self.player.chip.bet:  # 残額が賭けた額以上にあればDouble Down可
                self.player.double_down(card)
                self.show_card()
                if self.message_on:
                    print("Double down が選択されました．掛け金を倍にしました")
                    print("残りは $" + str(self.player.chip.balance))
                if self.player.hand.is_bust():
                    self.player.done = True
                    self.judgment = -1  # PlayerがBUSTしたら即負け
                    if self.message_on:
                        print("Player BUST")
            else:  # 残額が賭けた額未満ならばHitとする
                print("チップが足りないためHitします")
                self.player.hit(card)
                self.show_card()
                if self.player.hand.calc_final_point() == 21:  # 合計点が21になったらこれ以上Hitはできない
                    self.player.done = True
                if self.player.hand.is_bust():
                    self.player.done = True
                    self.judgment = -1  # PlayerがBUSTしたら即負け
                    if self.message_on:
                        print("Player BUST")

        elif action == 'r' and self.player.hit_flag is False:  # Surrender. Hit選択していない場合に可
            self.player.surrender()
            self.judgment = -1  # Surrenderを選択したので負け
            if self.message_on:
                print("Surrender が選択されました")

        else:
            if self.message_on:
                print("正しい選択肢を選んでください")

    def show_card(self):
        '''
        プレーヤーのカードを表示
        '''
        if self.message_on:
            print("Playerのターン")
            print("Player : " + str(self.player.hand.hand) + " = " +
                  str(self.player.hand.sum_point()) + ", soft card : " + str(self.player.hand.is_soft_hand))
            print("Dealer : " + str(self.dealer.hand.hand[0].index) +
                  ", ? = " + str(self.dealer.hand.hand[0].point))
        else:
            pass

    def dealer_turn(self):
        '''
        ディーラーのターン
        '''
        if self.judgment == -1:
            return
        self.open_dealer()
        while self.dealer.hand.calc_final_point() < 17 and self.judgment == 0:
            card = self.deck.draw(1)
            self.dealer.hit(card)
            self.open_dealer()
        if self.dealer.hand.calc_final_point() > 21:
            self.judgment = 1
            if self.message_on:
                print("Dealer BUST")

    def open_dealer(self):
        '''
        hole cardをオープンする
        '''
        if self.message_on:
            print("Dealerのターン")
            print("Player : " + str(self.player.hand.hand) + " = " +
                  str(self.player.hand.calc_final_point()))
            print("Dealer : " + str(self.dealer.hand.hand) + " = " +
                  str(self.dealer.hand.calc_final_point()))
        else:
            pass

    def judge(self):
        '''
        勝敗の判定
        '''
        if self.judgment == 0 and self.player.hand.calc_final_point() > \
                self.dealer.hand.calc_final_point():
            self.judgment = 1
        elif self.judgment == 0 and self.player.hand.calc_final_point() < \
                self.dealer.hand.calc_final_point():
            self.judgment = -1
        elif self.judgment == 0 and self.player.hand.calc_final_point() == \
                self.dealer.hand.calc_final_point():
            self.judgment = 0

        if self.message_on:
            self.show_judgement()

    def pay_chip(self):
        previous_chip = self.player.chip.balance
        if self.judgment == 1:  # Player win
            self.player.chip.pay_chip_win()
        elif self.judgment == -1:  # Player lose
            self.player.chip.pay_chip_lose()
        elif self.judgment == 0:  # Push
            self.player.chip.pay_chip_push()
        if self.message_on:
            print("Playerの所持チップは $" + str(self.player.chip.balance))

        reward = self.player.chip.balance - previous_chip  # このゲームで得た報酬
        return reward

    def check_chip(self):
        if self.player.chip.balance < MINIMUM_BET:
            self.game_mode = 2
            if self.message_on:
                print("チップがMinimum Betを下回ったのでゲームを終了します")

    def show_judgement(self):
        '''
        勝敗の表示
        '''
        if self.message_on:
            print("")
            if self.judgment == 1:
                print("Playerの勝ち")
            elif self.judgment == -1:
                print("Playerの負け")
            elif self.judgment == 0:
                print("引き分け")
            print("")
        else:
            pass

    def ask_next_game(self):
        '''
        ゲームを続けるか尋ねる
        '''
        if self.player.is_human == True:
            while self.game_mode == 1:
                player_input = input("続けますか？ y/n: ")
                if player_input == 'y':
                    break
                elif player_input == 'n':
                    self.game_mode = 2
                    break
                else:
                    print('y/nを入力してください')
        else:
            pass  # 自動プレイなら継続する
        print('残りカード枚数は ' + str(self.deck.count_cards()))
        print("")

    def check_deck(self):
        '''
        カードの残り枚数をチェックし，少なければシャッフルする
        '''
        if self.deck.count_cards() < NUM_PLAYER * 10 + 5:
            self.deck = Deck()
            if self.message_on:
                print("デッキを初期化しました")
                print('残りカード枚数は ' + str(self.deck.count_cards()))
                print("")


def main():
    game = Game()
    game.start()
    while game.game_mode == 1:
        game.reset_game()       # いろいろをリセットする
        game.bet(bet=100)       # 賭ける
        game.deal()             # カードを配る
        game.player_turn()      # プレイヤーのターン
        game.dealer_turn()      # ディーラーのターン
        game.judge()            # 勝敗の判定
        game.pay_chip()         # チップの精算
        game.check_chip()       # プレイヤーの残額を確認
        game.ask_next_game()    # ゲームを続けるか尋ねる
        game.check_deck()       # 残りカード枚数の確認

    print("BlackJackを終了します")
    print(str(game.game_count) + "回ゲームをしました")

    return game.player.chip, game.game_count

'''
class BlackJackEnv(gym.Env):
    metadata = {'render.mode': ['human', 'ansi']}

    MAX_GAME_COUNT = 50  # 1エピソード内で行う最大のゲーム数

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
        ])
        low = np.array([
            2,  # player min
            1,  # dealer min
            0,  # is_soft_hand
        ])
        self.observation_space = gym.spaces.Box(low=low, high=high)  # プレイヤーのポイント，ディーラーのポイント
        self.reward_range = [-10000, 10000]  # 報酬の最小値と最大値のリスト
        self._reset()

    def _reset(self):
        # 状態を初期化し，初期の観測値を返す
        # 諸々の変数を初期化する
        self.done = False
        self.bet_done = False
        self.steps = 0

        self.chip = 0  # self.game.player.chip.balance

        self.game.start()  # game_mode=1にする

        # self.player_point = 0
        # self.dealer_up_card = 0
        # self.observation = [self.player_point, self.dealer_up_card]

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

        else:
            # プレーヤーのターンを継続するとき
            reward = 0  #self._get_reward()

        observation = self._observe()

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
        reward = acquired_chip - self.game.player.chip.bet  #.game.player.chip.balance
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

        self.observation = np.array([
            self.game.player.hand.calc_final_point(),
            self.game.dealer.hand.hand[0].point,
            self.game.player.hand.is_soft_hand])

        observation = tuple(self.observation)

        return observation
'''

if __name__ == '__main__':
    main()

