from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate, _montecarlo_simulation
import random as rand
from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card, attach_hole_card_from_deck
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.utils.card_utils import _pick_unused_card, _fill_community_card


class CallPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
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


class FoldPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        call_action_info = valid_actions[0]
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

    
class HeuristicPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        self.nb_player = 9
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        call_action_info = valid_actions[1]
        fold_action_info = valid_actions[0]

        community_card = round_state['community_card']
        win_rate = estimate_hole_card_win_rate(nb_simulation=100, nb_player=self.nb_player,
                                               hole_card=gen_cards(hole_card),
                                               community_card=gen_cards(community_card))
        if win_rate > 1 / float(self.nb_player) + 0.1:
            action, amount = call_action_info["action"], call_action_info["amount"]
        else:
            action, amount = fold_action_info["action"], fold_action_info["amount"]
        return action, amount

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    
class RandomPlayer(BasePokerPlayer):
    def __init__(self):
        self.fold_ratio, self.call_ratio, self.raise_ratio = 1.0/5, 3.0/5, 1.0/5

    def set_action_ratio(self, fold_ratio, call_ratio, raise_ratio):
        ratio = [fold_ratio, call_ratio, raise_ratio]
        scaled_ratio = [ 1.0 * num / sum(ratio) for num in ratio]
        self.fold_ratio, self.call_ratio, self.raise_ratio = scaled_ratio

    def declare_action(self, valid_actions, hole_card, round_state):
        choice = self.__choice_action(valid_actions)
        action = choice["action"]
        amount = choice["amount"]
        if action == "raise":
            amount = rand.randrange(amount["min"], max(amount["min"], amount["max"]) + 1)
        return action, amount

    def __choice_action(self, valid_actions):
        r = rand.random()
        if r <= self.fold_ratio:
            return valid_actions[0]
        elif r <= self.call_ratio:
            return valid_actions[1]
        else:
            return valid_actions[2]


    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


'''
  MonteCarloBot courtesy of Victor Velev on Github.
'''

# Estimate the ratio of winning games given the current state of the game
def estimate_win_rate(n_simulations, n_players, hole_card, community_card=None):
    if not community_card: community_card = []

    # Make lists of Card objects out of the list of cards
    community_card = gen_cards(community_card)
    hole_card = gen_cards(hole_card)

    # Estimate the win count by doing a Monte Carlo simulation
    win_count = sum([montecarlo_simulation(n_players, hole_card, community_card) for _ in range(n_simulations)])
    return 1.0 * win_count / n_simulations


def montecarlo_simulation(n_players, hole_card, community_card):
    # Do a Monte Carlo simulation given the current state of the game by evaluating the hands
    community_card = _fill_community_card(community_card, used_card=hole_card + community_card)
    unused_cards = _pick_unused_card((n_players - 1) * 2, hole_card + community_card)
    opponents_hole = [unused_cards[2*i:2*i+2] for i in range(n_players - 1)]
    opponents_score = [HandEvaluator.eval_hand(hole, community_card) for hole in opponents_hole]
    my_score = HandEvaluator.eval_hand(hole_card, community_card)
    return 1 if my_score >= max(opponents_score) else 0


class MonteCarloBot(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.wins = 0
        self.losses = 0
        self.n_simulations = 1000

    def declare_action(self, valid_actions, hole_card, round_state):
        # Estimate the win rate
        win_rate = estimate_win_rate(self.n_simulations, self.n_players, hole_card, round_state['community_card'])

        # Check whether it is possible to call
        can_call = len([item for item in valid_actions if item['action'] == 'call']) > 0
        if can_call:
            # If so, compute the amount that needs to be called
            call_amount = [item for item in valid_actions if item['action'] == 'call'][0]['amount']
        else:
            call_amount = 0

        amount = None

        # If the win rate is large enough, then raise
        if win_rate > 0.5:
            raise_amount_options = [item for item in valid_actions if item['action'] == 'raise'][0]['amount']
            if win_rate > 0.85:
                # If it is extremely likely to win, then raise as much as possible
                action = 'raise'
                amount = raise_amount_options['max']
            elif win_rate > 0.75:
                # If it is likely to win, then raise by the minimum amount possible
                action = 'raise'
                amount = raise_amount_options['min']
            else:
                # If there is a chance to win, then call
                action = 'call'
        else:
            action = 'call' if can_call and call_amount == 0 else 'fold'

        # Set the amount
        if amount is None:
            items = [item for item in valid_actions if item['action'] == action]
            amount = items[0]['amount']

        return action, amount

    def receive_game_start_message(self, game_info):
        self.n_players = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        is_winner = self.uuid in [item['uuid'] for item in winners]
        self.wins += int(is_winner)
        self.losses += int(not is_winner)

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

NB_SIMULATION = 200
DEBUG_MODE = True
def log(msg):
    if DEBUG_MODE: print("[debug_info] --> %s" % msg)

class EmulatorPlayer(BasePokerPlayer):

    def set_opponents_model(self, model_player):
        self.opponents_model = model_player

    # setup Emulator with passed game information
    def receive_game_start_message(self, game_info):
        self.my_model = MyModel()
        nb_player = game_info['player_num']
        max_round = game_info['rule']['max_round']
        sb_amount = game_info['rule']['small_blind_amount']
        ante_amount = game_info['rule']['ante']

        self.emulator = Emulator()
        self.emulator.set_game_rule(nb_player, max_round, sb_amount, ante_amount)
        for player_info in game_info['seats']:
            uuid = player_info['uuid']
            player_model = self.my_model if uuid == self.uuid else self.opponents_model
            self.emulator.register_player(uuid, player_model)

    def declare_action(self, valid_actions, hole_card, round_state):
        try_actions = [MyModel.FOLD, MyModel.CALL, MyModel.MIN_RAISE, MyModel.MAX_RAISE]
        action_results = [0 for i in range(len(try_actions))]

        log("hole_card of emulator player is %s" % hole_card)
        for action in try_actions:
            self.my_model.set_action(action)
            simulation_results = []
            for i in range(NB_SIMULATION):
                game_state = self._setup_game_state(round_state, hole_card)
                round_finished_state, _events = self.emulator.run_until_round_finish(game_state)
                my_stack = [player for player in round_finished_state['table'].seats.players if player.uuid == self.uuid][0].stack
                simulation_results.append(my_stack)
            action_results[action] = 1.0 * sum(simulation_results) / NB_SIMULATION
            log("average stack after simulation when declares %s : %s" % (
                {0:'FOLD', 1:'CALL', 2:'MIN_RAISE', 3:'MAX_RAISE'}[action], action_results[action])
                )

        best_action = max(zip(action_results, try_actions))[1]
        self.my_model.set_action(best_action)
        return self.my_model.declare_action(valid_actions, hole_card, round_state)

    def _setup_game_state(self, round_state, my_hole_card):
        game_state = restore_game_state(round_state)
        game_state['table'].deck.shuffle()
        player_uuids = [player_info['uuid'] for player_info in round_state['seats']]
        for uuid in player_uuids:
            if uuid == self.uuid:
                game_state = attach_hole_card(game_state, uuid, gen_cards(my_hole_card))  # attach my holecard
            else:
                game_state = attach_hole_card_from_deck(game_state, uuid)  # attach opponents holecard at random
        return game_state

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

class MyModel(BasePokerPlayer):

    FOLD = 0
    CALL = 1
    MIN_RAISE = 2
    MAX_RAISE = 3

    def set_action(self, action):
        self.action = action

    def declare_action(self, valid_actions, hole_card, round_state):
        if self.FOLD == self.action:
            return valid_actions[0]['action'], valid_actions[0]['amount']
        elif self.CALL == self.action:
            return valid_actions[1]['action'], valid_actions[1]['amount']
        elif self.MIN_RAISE == self.action:
            return valid_actions[2]['action'], valid_actions[2]['amount']['min']
        elif self.MAX_RAISE == self.action:
            return valid_actions[2]['action'], valid_actions[2]['amount']['max']
        else:
            raise Exception("Invalid action [ %s ] is set" % self.action)

NB_SIMULATION = 200

class HonestPlayer(BasePokerPlayer):

    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state['community_card']
        win_rate = estimate_hole_card_win_rate(
                nb_simulation=NB_SIMULATION,
                nb_player=self.nb_player,
                hole_card=gen_cards(hole_card),
                community_card=gen_cards(community_card)
                )
        if win_rate >= 1.0 / self.nb_player:
            action = valid_actions[1]  # fetch CALL action info
        else:
            action = valid_actions[0]  # fetch FOLD action info
        return action['action'], action['amount']

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass