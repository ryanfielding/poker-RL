from pypokerengine.players import BasePokerPlayer
import EvgenyScripts.PlayerModels as pm

import sys
sys.path.insert(0, './cache/')
sys.path.insert(1, './EvgenyScripts/')

import bots.DQNPlayer1v1 as dqn

from pypokerengine.api.game import setup_config, start_poker
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
import numpy as np

p1 = dqn.DQNPlayer()

config = setup_config(max_round=100, initial_stack=1000, small_blind_amount=15)
config.register_player(name="p1", algorithm=p1)
numGames = 50
opponents = [pm.HonestPlayer(), pm.MonteCarloBot(), pm.RandomPlayer(), pm.CallPlayer(), pm.HeuristicPlayer()]
xlabels =('vs HonestPlayer','vs MonteCarlo','vs RandomPlayer','vs CallPlayer','vs HeuristicPlayer')
plotdir = './plots/'
p1Wins_list = []
p2Wins_list = []

for opponent in range(len(opponents)):
    p1Wins = 0
    p2Wins = 0
    config = setup_config(max_round=100, initial_stack=1000, small_blind_amount=15)
    config.register_player(name="p1", algorithm=p1)
    config.register_player(name="p2", algorithm=opponents[opponent])
    for game in range(numGames):
        print('game: ', game)
        game_result = start_poker(config, verbose=0)
        print(game_result)

        x = ['RLPlayer', 'CPU']
        if game_result['players'][0]['stack'] > game_result['players'][1]['stack'] :
            p1Wins += 1
        else:
            p2Wins += 1
    p1Wins_list.append(p1Wins)
    p2Wins_list.append(p2Wins)

ind = np.arange(len(opponents)) 
width = 0.35       
plt.bar(ind, p1Wins_list, width, label='RLPlayer')
plt.bar(ind + width, p2Wins_list, width, label='Bot')
plt.ylabel('Num. of Wins')
plt.title('50 Games - RLPlayer vs Bots')

plt.xticks(ind + width / 2, xlabels, rotation=12)
plt.legend(loc='best')
plt.savefig(plotdir+'Simulate.png')
plt.show()
