from pypokerengine.players import BasePokerPlayer
import sys
sys.path.insert(0, './cache/')
sys.path.insert(1, './EvgenyScripts/')

import bots.DQNPlayer as dqn

class FishPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
        return action, amount   # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


from pypokerengine.api.game import setup_config, start_poker

config = setup_config(max_round=10, initial_stack=100, small_blind_amount=5)
config.register_player(name="p1", algorithm=dqn.DQNPlayer())
config.register_player(name="p2", algorithm=FishPlayer())
config.register_player(name="p3", algorithm=FishPlayer())
config.register_player(name="p4", algorithm=FishPlayer())
config.register_player(name="p5", algorithm=FishPlayer())
config.register_player(name="p6", algorithm=FishPlayer())
config.register_player(name="p7", algorithm=FishPlayer())
config.register_player(name="p8", algorithm=FishPlayer())
config.register_player(name="p9", algorithm=FishPlayer())
game_result = start_poker(config, verbose=1)