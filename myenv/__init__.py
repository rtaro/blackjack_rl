from gym.envs.registration import register

register(
    id='BlackJack-v0',
    entry_point='myenv.blackjack_env:BlackJackEnv',
)
