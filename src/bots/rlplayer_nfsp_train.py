''' Training of an NFSP agent in PyPokerEngine's No Limit Texas Holdem
    using an RL agent from RLCard (kudos to them).
'''
from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.game_state_utils import restore_game_state
from bots.ai_bot import MonteCarloBot
from mymodule.poker_ai.player_model import SomePlayerModel

import os
import torch

import rlcard
from rlcard.agents import NFSPAgentPytorch as NFSPAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

'''
    Setup RLPlayer Class
'''
class RLPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.wins = 0
        self.losses = 0
        '''
        Instantiate agent.
        '''
        # Setup RL NFSP agent
        # Set the iterations numbers and how frequently we evaluate/save plot
        evaluate_every = 10000
        evaluate_num = 10000
        episode_num = 100000
        # The intial memory size
        memory_init_size = 1000
        # Train the agent every X steps
        train_every = 64
        # The paths for saving the logs and learning curves
        log_dir = './training/nfsp/'
        # Set a global seed
        set_global_seed(0)
        # Set agent - TODO - determine PPE parameters
        self.agent = NFSPAgent(scope='nfsp',
                            action_num=3,
                            state_shape=54,
                            hidden_layers_sizes=[512,512],
                            min_buffer_size_to_learn=memory_init_size,
                            q_replay_memory_init_size=memory_init_size,
                            train_every=train_every,
                            q_train_every = train_every,
                            q_mlp_layers=[512,512],
                            device=torch.device('cpu'))
        # Init a Logger to plot the learning curve
        self.logger = Logger(log_dir)
        
    # example from MC player - TODO
    def declare_action(self, valid_actions, hole_card, round_state):
        # updates to round states
        self.state['legal_actions'] = valid_actions
        self.state['hole_card'] = hole_card
        self.state['round_state'] = round_state

        # Check whether it is possible to call
        can_call = len([item for item in valid_actions if item['action'] == 'call']) > 0
        if can_call:
            # If so, compute the amount that needs to be called
            call_amount = [item for item in valid_actions if item['action'] == 'call'][0]['amount']
        else:
            call_amount = 0

        # Set the amount
        amount = None
        if amount is None:
            items = [item for item in valid_actions if item['action'] == action]
            amount = items[0]['amount']
        
        # feed forward pass through agent NN
        action = self.agent.step()
        return action, amount

        # From MonteCarlo for ref
        # # Estimate the win rate
        # win_rate = estimate_win_rate(self.n_simulations, self.n_players, hole_card, round_state['community_card'])

        # # Check whether it is possible to call
        # can_call = len([item for item in valid_actions if item['action'] == 'call']) > 0
        # if can_call:
        #     # If so, compute the amount that needs to be called
        #     call_amount = [item for item in valid_actions if item['action'] == 'call'][0]['amount']
        # else:
        #     call_amount = 0

        # amount = None

        # # If the win rate is large enough, then raise
        # if win_rate > 0.5:
        #     raise_amount_options = [item for item in valid_actions if item['action'] == 'raise'][0]['amount']
        #     if win_rate > 0.85:
        #         # If it is extremely likely to win, then raise as much as possible
        #         action = 'raise'
        #         amount = raise_amount_options['max']
        #     elif win_rate > 0.75:
        #         # If it is likely to win, then raise by the minimum amount possible
        #         action = 'raise'
        #         amount = raise_amount_options['min']
        #     else:
        #         # If there is a chance to win, then call
        #         action = 'call'
        # else:
        #     action = 'call' if can_call and call_amount == 0 else 'fold'

        # # Set the amount
        # if amount is None:
        #     items = [item for item in valid_actions if item['action'] == action]
        #     amount = items[0]['amount']

        # return action, amount
        
    # Encode state data here upon receiving game/round updates and put in trajectory
    # Need to set agent.feed(ts) somewhere
    def receive_game_start_message(self, game_info):
        self.n_players = game_info['player_num']
        self.trajectory = 0 # placeholder
        self.agent.feed(trajectory)
        

    def receive_round_start_message(self, round_count, hole_card, seats):
        # update hole card in state vector
        self.trajectory = 1 # placeholder
        self.agent.feed(trajectory)
        pass

    def receive_street_start_message(self, street, round_state):
        # update commnunity cards in state vector
        self.trajectory = 1 # placeholder
        self.agent.feed(trajectory)
        pass

    def receive_game_update_message(self, action, round_state):
        # this is called upon every player action
        self.trajectory = 1 # placeholder
        self.agent.feed(trajectory)
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        is_winner = self.uuid in [item['uuid'] for item in winners]
        self.wins += int(is_winner)
        self.losses += int(not is_winner)
        # pass this data
        self.agent.feed(trajectory)
        # learn based of round result
        self.agent.train_sl()

    
def setup_ai():
    return RLPlayer()

# Setup environment for training
from pypokerengine.api.game import setup_config, start_poker
env = setup_config(max_round=10, initial_stack=100, small_blind_amount=5)

# Train
# Add players and start the game
env.register_player(name="MC", algorithm=MonteCarloBot())
env.register_player(name="RL", algorithm=RLPlayer())
game_result = start_poker(env, verbose=1)

# # Train
# for episode in range(episode_num):

#     # First sample a policy for the episode
#     for agent in agents:
#         agent.sample_episode_policy()

#     # Generate data from the environment
#     trajectories, _ = env.run(is_training=True)

#     # Feed transitions into agent memory, and train the agent
#     for i in range(env.player_num):
#         for ts in trajectories[i]:
#             agents[i].feed(ts)

#     # Evaluate the performance. Play with random agents.
#     if episode % evaluate_every == 0:
#         logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

# Close files in the logger
logger.close_files()

# Plot the learning curve
logger.plot('NFSP')

# Save model for playing against RLPlayer
save_dir = 'models/nfsp'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
state_dict = {}
for agent in agents:
    state_dict.update(agent.get_state_dict())
torch.save(state_dict, os.path.join(save_dir, 'nfspmodel.pth'))