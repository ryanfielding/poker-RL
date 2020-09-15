# poker-RL
Python poker Texas Hold'em AI through TensorFlow reinforcement learning and PyPokerEngine.

# Install
Clone this repo, then run the command:
*	  pip3 install -r requirements.txt

# Play vs the AI in GUI
Run this command in terminal.
*     python3 -m pypokergui serve poker_conf.yaml --port 8000 --speed moderate

# Training
To train the AI at 1v1 poker (be sure to set which player you'd like to train against in line 104 of the script).
*	  cd src/
*	  python3 training1v1.py
Pre-trained models can be found in src/cache/models/DQN2 for 1v1 or DQN9 for 1v8 poker.
To visualize tf with tensorboard run this command in terminal:
*     tensorboard --logdir=log/DQN2/

# Simulating / Testing
Play a game against chosen player:
*	  cd src/
*	  python3 play2.py
Or simulate 50 games against various players and save plots:
*	  python3 simulate.py

# Reports
The project was completed for a directed study in school. Feel free to check them out in reports/ (written in Latex).

# Versions
*	running on Mac OS X 10.15.3
*   brew install pyenv
*   pyenv install 3.6.0 # link pip3 and python3 aliases
*   add these to .bash_profile
*     alias python3=/Users/Ryan/.pyenv/shims/python3
*     alias pip3=/Users/Ryan/.pyenv/shims/pip3
*   python3 --version (ensure 3.6.0)
*   make bug fix for pypokergui shown below

# Props
*   Initial setup thanks to https://github.com/VIVelev/PokerBot-AI
*   Bug fix in python module pypokergui 'site-packages\pypokergui\server\templates'. See here for fix: https://github.com/chrisking/PyPokerGUI/commit/00d63bec4b2a3ee195851521362f812321fa8e4d
*   RLCard Contributors for references: https://github.com/datamllab/rlcard
*   EvgenyKashin @ Github for Double-Dueling-DQN architecture: https://github.com/EvgenyKashin/TensorPoker
