
import numpy as np
import gym
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD 
from tqdm import tqdm
from keras import backend as K



class sumtree:
    def __init__(self,capacity):
        self.write = 0
        self.addnum = 0
        self.capacity = capacity
        self.tree = np.zeros(2*capacity -1)
        self.data = np.zeros(capacity)
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
            return self.retrive(right_leave, s-self.tree[left_leave])
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
        #min_priority = np.min(self.tree[-self.capacity:])
        min_priority =1
        for choose_n in range(n):
            random_left = pri_seg*choose_n
            random_right = pri_seg*(choose_n+1)

            random_s = np.random.uniform(random_left, random_right)
            #print(random_right,random_s,self.total())
            #print(self.total(),choose_n, random_s,sumtree.tree,'random')
            choose_idx[choose_n], priority_choose, choose_batch[choose_n]=self.get(random_s)
            priority_nominize = priority_choose / self.total()
            ISweight[choose_n] = np.power(priority_nominize/min_priority, -beta)
            #print('sample',choose_batch)
        return choose_batch, choose_idx, ISweight, beta

if __name__ == "__main__":
    sumtree=sumtree(32)
    sumtree.add(1,1)
    sumtree.add(2,3)   
    sumtree.add(3,4)
    for i in range(19):
        sumtree.add(np.random.randint(1,4),5)
       #print(sumtree.write)
    q,k,j=sumtree.get(3.5)
    minibatch,minibatch_idx,ISweight_use,beta =sumtree.sample(10, 1, 0)
    print(minibatch_idx)