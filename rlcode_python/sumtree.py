
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
        if self.write > self.capacity:
            self.write = 0 
        print(self.tree)
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

if __name__ == "__main__":
	sumtree=sumtree(5)
	sumtree.add(1,1)
	sumtree.add(2,3)
	sumtree.add(3,4)
	q,k,j=sumtree.get(5.9)
	print(q,k,j, sumtree.tree, sumtree.data)