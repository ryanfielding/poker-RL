'''
    Full Credit: EvgenyKashin @ Github
'''

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
#%matplotlib inline

from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import Card, Deck
from pypokerengine.api.game import setup_config, start_poker

import pickle
import tensorflow as tf
import random
from tensorflow import keras

import sys
sys.path.insert(0, '../cache/')

import PlayerModels as pm
from MyEmulator import MyEmulator
from DQNPlayer1v1 import DQNPlayer
from util import *

# %%
"""
### Graph
"""

# %%
def update_target_graph(tf_vars, tau):
    total_vars = len(tf_vars)
    ops = []
    for i, var in enumerate(tf_vars[0:total_vars // 2]):
        ops.append(tf_vars[i + total_vars // 2].assign((var.value() * tau) +
                                                       (tf_vars[i + total_vars // 2].value() * (1 - tau))))
    return ops

def update_target(ops, sess):
    for op in ops:
        sess.run(op)

# %%
"""
## Training
"""

# %%
batch_size = 128
update_freq = 50 # how often to update model
y = 0.99 # discount
start_E = 1 # starting chance of random action
end_E = 0.2 # final chance of random action
num_episodes = 500000 # total games of poker - was 500k
annealings_steps = num_episodes/5 # how many steps to reduce start_E to end_E
pre_train_steps = 300 # how many steps of random action before training begin - was 5000 (1000 for DQN2)
load_model = False
path = '/Users/Ryan/Repos/poker-RL/src/cache/models/DQN2LogTests/DQN'
h_size = 128 # the size of final conv layer before spliting it into advantage and value streams
tau = 0.01 # rate to update target network toward primary network
is_dueling = True # whether or not to use dueling architecture

# %%
emul = MyEmulator()
# players_info = {
#     "1": { "name": "f1", "stack": 1500 },
#     "2": { "name": "f2", "stack": 1500 },
#     "3": { "name": "f3", "stack": 1500 },
#     "4": { "name": "f4", "stack": 1500 },
#     "5": { "name": "f5", "stack": 1500 },
#     "6": { "name": "f6", "stack": 1500 },
#     "7": { "name": "f7", "stack": 1500 },
#     "8": { "name": "f8", "stack": 1500 },
#     "9": { "name": "f9", "stack": 1500 }
# }
players_info = {
    "1": { "name": "f1", "stack": 1500 },
    "2": { "name": "f2", "stack": 1500 }
}
emul.set_game_rule(len(players_info), 50, 15, 0)
my_uuid = '2' # RL player number

def init_emul(my_uuid_):
    global my_uuid
    my_uuid = my_uuid_
    
    # emul.register_player("1", pm.CallPlayer())
    # emul.register_player("2", pm.CallPlayer())
    # emul.register_player("3", pm.FoldPlayer())
    # emul.register_player("4", pm.FoldPlayer())
    # emul.register_player("5", pm.HeuristicPlayer())
    # emul.register_player("6", pm.HeuristicPlayer())
    # emul.register_player("7", pm.RandomPlayer())
    # emul.register_player("8", pm.RandomPlayer())
    # emul.register_player("9", pm.CallPlayer())

    emul.register_player("1", pm.HeuristicPlayer())
    emul.register_player("2", pm.CallPlayer())
    # emul.register_player("3", pm.FoldPlayer())
    # emul.register_player("4", pm.FoldPlayer())
    # emul.register_player("5", pm.HeuristicPlayer())
    # emul.register_player("6", pm.HeuristicPlayer())
    # emul.register_player("7", pm.RandomPlayer())
    # emul.register_player("8", pm.RandomPlayer())
    # emul.register_player("9", pm.CallPlayer())

    # players_info = {
    #     "1": { "name": "MCBot", "stack": 1500 },
    #     "2": { "name": "CallPlayer2", "stack": 1500 },
    #     "3": { "name": "FoldPlayer1", "stack": 1500 },
    #     "4": { "name": "FoldPlayer2", "stack": 1500 },
    #     "5": { "name": "HeuristicPlayer1", "stack": 1500 },
    #     "6": { "name": "HeuristicPlayer2", "stack": 1500 },
    #     "7": { "name": "RandomPlayer1", "stack": 1500 },
    #     "8": { "name": "RandomPlayer2", "stack": 1500 },
    #     "9": { "name": "DQN", "stack": 1500 }
    # }
    players_info = {
        "1": { "name": "Bot", "stack": 1500 },
        "2": { "name": "DQN", "stack": 1500 }
    }
    # print (my_uuid)
    # print ('Num players:')
    # print (len(players_info))

# %%
tf.compat.v1.reset_default_graph()
main_wp = DQNPlayer(h_size, is_double=True)
target_wp = DQNPlayer(h_size, is_main=False, is_double=True)

# %%%%time
init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver(max_to_keep=3)
trainables = tf.compat.v1.trainable_variables()
target_ops = update_target_graph(trainables, tau)
my_buffer = ExperienceBuffer()
# disable eager
tf.compat.v1.disable_eager_execution()

e = start_E
step_drop = (start_E - end_E) / annealings_steps

j_list = []
r_list = []
action_list = []
total_steps = 0
q1_list = []
q2_list = []
e_list = []
q_tar_list = []
q_act_list = []
err_list = []
g_list = []
v_list = []

# setup scalar logging
logdir = '../cache/logs/DQN2Logs'
plotdir = '../cache/plots/'
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
r=tf.Variable(0.0)
tf.compat.v1.summary.scalar('Perf/Reward', tensor=tf.convert_to_tensor(r))
perf_summaries = tf.compat.v1.summary.merge_all()

config = tf.compat.v1.ConfigProto()
sess = tf.compat.v1.Session(config=config)

with tf.compat.v1.Session() as sess:
    writer = tf.compat.v1.summary.FileWriter(logdir, sess.graph)
    sess.run(init)
    if load_model:
        print('Loading model')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    for i in range(num_episodes):
        episode_buffer = ExperienceBuffer()
        init_emul(str(np.random.randint(1, len(players_info) + 1)))
        
        initial_state = emul.generate_initial_game_state(players_info)
        msgs = []
        game_state, events = emul.start_new_round(initial_state)
        is_last_round = False
        r_all = 0
        j = 0
        
        last_img_state = None
        last_features = None
        last_action_num = None
        
        while not is_last_round:
            j += 1
            a = emul.run_until_my_next_action(game_state, my_uuid, msgs)
            
            # need to make move
            if len(a) == 4:
                game_state, valid_actions, hole_card, round_state = a
                img_state = img_from_state(hole_card, round_state)
                img_state = process_img(img_state)
                
                street = round_state['street']
                bank = round_state['pot']['main']['amount']
                stack = [s['stack'] for s in round_state['seats'] if s['uuid'] == my_uuid][0]
                other_stacks = [s['stack'] for s in round_state['seats'] if s['uuid'] != my_uuid]
                dealer_btn = round_state['dealer_btn']
                small_blind_pos = round_state['small_blind_pos']
                big_blind_pos = round_state['big_blind_pos']
                next_player = round_state['next_player']
                round_count = round_state['round_count']
                estimation = main_wp.hole_card_est[(hole_card[0], hole_card[1])]

                features = get_street(street)
                features.extend([bank, stack, dealer_btn, small_blind_pos, big_blind_pos, next_player, round_count])
                features.extend(other_stacks)
                features.append(estimation)
                        
                # add to buffer last hand 
                if last_img_state is not None:
                    episode_buffer.add(np.reshape(np.array([last_img_state, last_features, last_action_num,
                                                            0, img_state, features, 0]), [1, 7]))
                
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    action_num = np.random.randint(0, main_wp.total_num_actions)
                else:
                    action_num = sess.run(main_wp.predict, feed_dict={main_wp.scalar_input: [img_state],
                                                                      main_wp.features_input: [features]})[0]
                    
                action_list.append(action_num)
                action, amount = get_action_by_num(action_num, valid_actions)                    

                game_state, msgs = emul.apply_my_action(game_state, action, amount)
                total_steps += 1
        
                last_img_state = img_state.copy()
                last_features = features.copy()
                last_action_num = action_num
                
                if total_steps > pre_train_steps:
                    if e > end_E:
                        e -= step_drop
                    
                    if total_steps % (update_freq) == 0:
                        train_batch = my_buffer.sample(batch_size)

                        Q1 = sess.run(main_wp.predict,
                                      feed_dict={main_wp.scalar_input: np.vstack(train_batch[:, 4]),
                                                 main_wp.features_input: np.vstack(train_batch[:, 5])})
                        Q1_ = sess.run(main_wp.Q_out,
                                      feed_dict={main_wp.scalar_input: np.vstack(train_batch[:, 4]),
                                                 main_wp.features_input: np.vstack(train_batch[:, 5])})
        
                        Q2 = sess.run(target_wp.Q_out,
                                      feed_dict={target_wp.scalar_input: np.vstack(train_batch[:, 4]),
                                                 target_wp.features_input: np.vstack(train_batch[:, 5])})
                        end_multiplier = -(train_batch[:, 6] - 1)
                        double_Q = Q2[range(batch_size), Q1]
                        double_Q_ = Q1_[range(batch_size), Q1]
                        
                        if is_dueling:
                            target_Q = train_batch[:, 3] + (y * double_Q * end_multiplier)
                        else:
                            target_Q = train_batch[:, 3] + (y * double_Q_ * end_multiplier)

                        _, er, g, v = sess.run([main_wp.update_model,
                                          main_wp.loss, main_wp.grad_norms, main_wp.var_norms],
                                     feed_dict={
                                         main_wp.scalar_input: np.vstack(train_batch[:, 0]),
                                         main_wp.features_input: np.vstack(train_batch[:, 1]),
                                         main_wp.target_Q: target_Q,
                                         main_wp.actions: train_batch[:, 2]
                                     })
                        update_target(target_ops, sess)
                        
                        r = np.mean(r_list[-2:])
                        j = np.mean(j_list[-2:])
                        q1 = double_Q_[0]
                        q2 = double_Q[0]
                        al = np.mean(action_list[-10:])

                        #build lists
                        q1_list.append(q1)
                        q2_list.append(q2)
                        e_list.append(e)
                        q_tar_list.append(target_Q[0])
                        q_act_list.append(Q1[0])
                        err_list.append(er)
                        g_list.append(g)
                        v_list.append(v)

                        #summary = tf.summary()
                        #r_summary = tf.summary.scalar('Perf/Reward', data=float(r), step=total_steps)
                        #tf.compat.v1.summary.scalar('Perf/Reward', tensor=tf.convert_to_tensor(r))

                        # summary.value.add(tag='Perf/Reward', simple_value=float(r))
                        # summary.value.add(tag='Perf/Lenght', simple_value=float(j))
                        # summary.value.add(tag='Perf/Action_list', simple_value=al)
                        # summary.value.add(tag='Perf/E', simple_value=e)                        
                        # summary.value.add(tag='Q/Q1', simple_value=float(q1))
                        # summary.value.add(tag='Q/Q2', simple_value=float(q2))
                        # summary.value.add(tag='Q/Target', simple_value=target_Q[0])
                        # summary.value.add(tag='Q/Action', simple_value=Q1[0])
                        # summary.value.add(tag='Loss/Error', simple_value=er)
                        # summary.value.add(tag='Loss/Grad_norm', simple_value=g)
                        # summary.value.add(tag='Loss/Var_norm', simple_value=v)
                        print('reward: ', r)
                        
                        #summ = sess.run(perf_summaries)
                        #writer.add_summary(summ, total_steps)
                        #writer.flush()
                        #tf.summary.write(summ_writer, tensor=r, step=total_steps)
                        
                        #main_wp.summary_writer.add_summary(summary, total_steps)
                        if total_steps % (update_freq * 2) == 0:
                            main_wp.summary_writer.flush()     
                        print ('Trained model at', total_steps)
            else:
                game_state, reward = a
                if reward >= 0:
                    reward = np.log(1 + reward)
                else:
                    reward = -np.log(1 - reward)
                r_all += reward
                # add to buffer last hand 
                if last_img_state is not None:
                    episode_buffer.add(np.reshape(np.array([last_img_state, last_features, last_action_num,
                                                            reward, last_img_state, last_features, 1]), [1, 7]))
                
                is_last_round = emul._is_last_round(game_state, emul.game_rule)
                game_state, events = emul.start_new_round(game_state)

                last_img_state = None
                last_action_num = None   

        my_buffer.add(episode_buffer.buffer)
        r_list.append(r_all)
        j_list.append(j)
        
        print(i)
        if i % 100 == 0:
            # saver.save(sess, path + '/model_' + str(i) + '.ckpt')
            #saver.save(sess, path, i)
            saver.save(sess, path, global_step = i)
            print('Saved model.')
            print(i, total_steps, np.mean(r_list[-10:]), e, np.median(action_list[-200:]))
        if i%20 == 0 and i > 0:
            #save plots
            plt.plot(r_list)
            plt.ylabel('reward')
            plt.savefig(plotdir+'rewards'+str(i)+'.png')
            plt.close()

            plt.plot(action_list)
            plt.ylabel('Actions')
            plt.savefig(plotdir+'Acts'+str(i)+'.png')
            plt.close()

            plt.plot(j_list)
            plt.ylabel('length')
            plt.savefig(plotdir+'length'+str(i)+'.png')
            plt.close()

            plt.plot(e_list)
            plt.ylabel('error')
            plt.savefig(plotdir+'error'+str(i)+'.png')
            plt.close()
            
            plt.plot(q1_list)
            plt.ylabel('Q1')
            plt.savefig(plotdir+'Q1_'+str(i)+'.png')
            plt.close()

            plt.plot(q2_list)
            plt.ylabel('Q2')
            plt.savefig(plotdir+'Q2_'+str(i)+'.png')
            plt.close()

            plt.plot(q_act_list)
            plt.ylabel('Q Action')
            plt.savefig(plotdir+'QAction'+str(i)+'.png')
            plt.close()

            plt.plot(q_tar_list)
            plt.ylabel('Q Target')
            plt.savefig(plotdir+'QTarget'+str(i)+'.png')
            plt.close()

            plt.plot(err_list)
            plt.ylabel('Loss')
            plt.savefig(plotdir+'Loss'+str(i)+'.png')
            plt.close()

            plt.plot(g_list)
            plt.ylabel('Grad Norm')
            plt.savefig(plotdir+'GNorm'+str(i)+'.png')
            plt.close()

            plt.plot(v_list)
            plt.ylabel('Var Norm')
            plt.savefig(plotdir+'VNorm'+str(i)+'.png')
            plt.close()

            #print('r: ', r_list)
        #write tensor board graph
        #writer = tf.compat.v1.summary.FileWriter(logdir, sess.graph)
    # save after episodes complete
    
    saver.save(sess, path, global_step = i)
print('Mean reward: {}'.format(sum(r_list) / num_episodes))

# %%
"""
To see the training progress type in console:
tensorboard --logdir=log/DQN/
"""
