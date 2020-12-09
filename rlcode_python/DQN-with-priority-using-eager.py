import numpy as np
import gym
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD 
from tqdm import tqdm
from keras import backend as K
import tensorflow as tf
tf.executing_eagerly()

weight = 1

class sumtree:
    def __init__(self,capacity):
        self.write = 0
        self.addnum = 0
        self.capacity = capacity
        self.tree = np.zeros(2*capacity -1)
        self.data = np.zeros(capacity, dtype=object)
    def propagate(self, change, idx):
        parent = (idx-1)//2
        self.tree[parent]=self.tree[parent]+change
        #print('propagate',idx,change,self.tree)
        if parent !=0:
            self.propagate(change,parent)
    def update(self,p, idx):
        #print('update',idx)
        change = p - self.tree[idx]
        #print(change)
        self.tree[idx] = p
        #print('update',idx,self.tree)
        self.propagate(change,idx)
    def add(self, p, transition):
        self.data[self.write]=transition
        idx = self.capacity-1+self.write
        self.write =self.write + 1
        self.update(p,idx)
        self.addnum =self.addnum + 1
        if self.write >= self.capacity:
            self.write = 0 
        #print(self.tree)
    def retrive(self,idx,s):
        
        #print(idx,'idx')
        left_leave = 2 * idx +1
        right_leave = 2*idx +2
        if left_leave >= len(self.tree):
            #print('return',idx)
            return idx
        if self.tree[left_leave]>=s:
            #print('left')
            return self.retrive(left_leave,s)
        else:
            
            #print('right')
            return self.retrive(right_leave, s)
    def get(self, s):
        get_idx = self.retrive(0, s)
        #print(get_idx,self.data,self.tree)
        return get_idx, self.tree[get_idx], self.data[get_idx - self.capacity+1]

    def total(self):
        return self.tree[0]
    def sample(self,n,beta,beta_improve_perstep):
        choose_batch = np.zeros(n, dtype=object)
        beta=np.min([1,beta+beta_improve_perstep]) 
        ISweight = np.zeros(n)
        choose_idx = np.zeros(n)
        pri_seg = self.total()/n
        min_priority = np.min(self.tree[-self.capacity:])
        for choose_n in range(n):
            random_left = pri_seg*choose_n
            random_right = pri_seg*(choose_n+1)

            random_s = np.random.uniform(random_left, random_right)
            print(random_right,random_s,self.total())
            #print(self.total(),choose_n, random_s,sumtree.tree,'random')
            choose_idx[choose_n], priority_choose, choose_batch[choose_n]=self.get(random_s)
            priority_nominize = priority / self.total()
            ISweight[choose_n] = np.power(priority_nominize/min_priority, -beta)
            #print('sample',choose_batch)
        return choose_batch, choose_idx, ISweight, beta
def actionchoose(action_size,state):
    if np.random.binomial(1,0.1) == 1:
        return random.randrange(action_size)
    action_value = model.predict(state)
    return np.argmax(action_value)
def mse_with_IS(y_true,y_pred):
    print(y_true,'y_true')
    weight_array=y_true.numpy()
    weight_use= weight_array[0][2]
    y_true_without_weight=tf.convert_to_tensor(np.reshape(weight_array[0][0:2],[1,2]))
    return K.mean(K.square(y_true_without_weight-y_pred), axis=-1)


def change(weightwww):
    global weight
    weight = weightwww


if __name__ == "__main__":
    minibatch_size = 30
    gamma = 0.95
    beta = 0.4
    beta_improve_perstep = 0.001
    replay_inter=10
    sgd =   SGD(lr=0.01, decay=1e-6,momentum=0.9, nesterov=True)
    env = gym.make('CartPole-v1')
    action_size = env.action_space.n
    space_size = env.observation_space.shape[0]
    model = Sequential([
        Dense(24,input_dim=space_size),
        Activation('relu'),
        Dense(36),
        Activation('softmax'),
        Dense(action_size),
        Activation('linear')
    ])

    
    sumtree = sumtree(10)
    model_=model
    model.compile(loss=mse_with_IS,
                    optimizer=sgd,
                    metrics = ['accuracy'],
                    run_eagerly=True)
    for i in tqdm(range(1000)):
        state = env.reset()
        state = np.reshape(state, [1, space_size])
        for j in range(500):
            env.render()
            action=actionchoose(action_size, state)
            #print(action)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state,[1,space_size])
            action_target = np.argmax(model.predict(next_state))
            priority = np.abs(reward + gamma * model_.predict(next_state)[0][action_target] - model.predict(state)[0][action])
            sumtree.add(priority, (state, action, reward, next_state, done))             
            state = next_state
            #print(sumtree.addnum,'addnum',sumtree.write)
            if done:
                #print('end')
                break
            if sumtree.addnum%replay_inter ==0 and sumtree.addnum/replay_inter>1:
                #print('do')

                minibatch,minibatch_idx,ISweight_use,beta =sumtree.sample(minibatch_size, beta, beta_improve_perstep)
                model_=model
                idx_num = 0
                #print(sumtree.tree)
                #print(minibatch_idx)
                #print(minibatch)
                #print(sumtree.addnum,'addnum')
                for state, action, reward, next_state, done in minibatch:
                    tree_idx = minibatch_idx[idx_num]
                    #print(sumtree.addnum,minibatch_idx,'tree')
                    weight_tree = ISweight_use[idx_num]

                    if done:
                        traget_value_action = reward
                    else:
                        action_target = np.argmax(model.predict(next_state))
                        traget_value_action = reward + gamma * model_.predict(next_state)[0][action_target]
                    
                    target = model.predict(state)
                    priority_change = abs(traget_value_action - target[0][action])
                    sumtree.update(priority_change,np.int(tree_idx))
                    target[0][action]=weight*(traget_value_action-target[0][action])
                    target[0][1-action]=0
                    target_with_weight=np.append(target,weight_tree)
                    target_with_weight=np.reshape(target_with_weight,[1,3])
                    model.fit(state,target_with_weight,epochs=2, verbose=0)
                    idx_num = idx_num + 1
                    #print(idx_num,'idx_num',minibatch)
