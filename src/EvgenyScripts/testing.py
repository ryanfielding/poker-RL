'''
    Full Credit: EvgenyKashin @ Github
'''

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import Card, Deck
from pypokerengine.api.game import setup_config, start_poker

import pickle
import tensorflow as tf
import random

import sys
sys.path.insert(0, '../scripts/')

import PlayerModels as pm
from MyEmulator import MyEmulator
from DQNPlayer import DQNPlayer
from util import *

# %%
"""
## Initialization
"""

# %%
h_size = 128

# %%
%time main_wp = DQNPlayer(h_size=h_size, is_restore=True, is_train=False, debug=True, is_double=True)

# %%
"""
## Testing
"""

# %%
config = setup_config(max_round=2, initial_stack=1500, small_blind_amount=15, summary_file='/dev/null')

config.register_player(name="wp", algorithm=main_wp)
# config.register_player(name="r2", algorithm=RandomPlayer())
config.register_player(name="f2", algorithm=pm.CallPlayer())
config.register_player(name="f3", algorithm=pm.CallPlayer())
config.register_player(name="f4", algorithm=pm.CallPlayer())
config.register_player(name="f5", algorithm=pm.CallPlayer())
config.register_player(name="f6", algorithm=pm.CallPlayer())
config.register_player(name="f7", algorithm=pm.CallPlayer())
config.register_player(name="f8", algorithm=pm.CallPlayer())
config.register_player(name="f9", algorithm=pm.CallPlayer())

game_result = start_poker(config, verbose=1)

# %%
"""
## Metric
"""

# %%
%time main_wp = DQNPlayer(h_size=h_size, is_restore=True, is_train=False, debug=False, is_double=True)

# %%
config = setup_config(max_round=50, initial_stack=1500, small_blind_amount=15, summary_file='/dev/null')

config.register_player(name="wp", algorithm=main_wp)
# config.register_player(name="r2", algorithm=RandomPlayer())
config.register_player(name="CallPlayer1", algorithm=pm.CallPlayer())
config.register_player(name="CallPlayer2", algorithm=pm.CallPlayer())
config.register_player(name="FoldPlayer1", algorithm=pm.FoldPlayer())
config.register_player(name="FoldPlayer2", algorithm=pm.FoldPlayer())
config.register_player(name="HeuristicPlayer1", algorithm=pm.HeuristicPlayer())
config.register_player(name="HeuristicPlayer2", algorithm=pm.HeuristicPlayer())
config.register_player(name="RandomPlayer1", algorithm=pm.RandomPlayer())
config.register_player(name="RandomPlayer2", algorithm=pm.RandomPlayer())

# %%
%%time
d = None
for i in range(100):
    game_result = start_poker(config, verbose=0)
    t = pd.DataFrame(game_result['players'])
    t['round'] = i
    if d is None:
        d = t
    else:
        d = pd.concat((d, t))

# %%
"""
### With training only with CallPlayer for 3 hours
"""

# %%
d.groupby('name').mean()['stack'].sort_values()

# %%
"""
### With training  with different players for 4 hours
"""

# %%
d.groupby('name').mean()['stack'].sort_values()

# %%
"""
### With training  with different players for 8 hours
"""

# %%
d.groupby('name').mean()['stack'].sort_values()

# %%
"""
### With training  with different players for 15 hours
"""

# %%
d.groupby('name').mean()['stack'].sort_values()

# %%
"""
### With training  with different players for 26 hours
"""

# %%
d.groupby('name').mean()['stack'].sort_values()

# %%
