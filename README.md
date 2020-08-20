# poker-RL
Python poker Texas Hold'em bot through PyTorch/TensorFlow reinforcement learning and PyPokerEngine.

# Play vs the AI in GUI
#run this command in terminal
*   `python3 -m pypokergui serve poker_conf.yaml --port 8000 --speed moderate`

# Training
To visualize tf with tensorboard
*   `tensorboard --logdir=log/DQN2/`

# Versions (running on Mac OS X 10.15.3)
*   brew install pyenv
*   pyenv install 3.6.0 # link pip3 and python3 aliases
*   #add these to .bash_profile
*     alias python3=/Users/Ryan/.pyenv/shims/python3
*     alias pip3=/Users/Ryan/.pyenv/shims/pip3
*   python3 --version #ensure 3.6.0
*   pip3 install -r requirements.txt #installs fine with python 3.6.0
*     #make bug fix for pypokergui shown below

# Contributions
*   Initial setup thanks to https://github.com/VIVelev/PokerBot-AI
*   Bug fix in python module pypokergui 'site-packages\pypokergui\server\templates'. See here for fix: https://github.com/chrisking/PyPokerGUI/commit/00d63bec4b2a3ee195851521362f812321fa8e4d
*   RLCard Contributors for references: https://github.com/datamllab/rlcard
*   EvgenyKashin @ Github for Double-Dueling-DQN architecture.
