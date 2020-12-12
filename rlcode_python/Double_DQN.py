import numpy as np
import gym
import random
import matplotlib
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD 
from tqdm import tqdm
import matplotlib.pyplot as plt 



def actionchoose(action_size,state):
    if np.random.binomial(1,0.1) == 1:
        return random.randrange(action_size)
    action_value = model.predict(state)
    return np.argmax(action_value)



if __name__ == "__main__":
    tragectory = deque(maxlen=128)
    minibatch_size = 30
    gamma = 0.95
    sgd =   SGD(lr=0.01, decay=1e-6,momentum=0.9, nesterov=True)
    env = gym.make('CartPole-v1')
    action_size = env.action_space.n
    space_size = env.observation_space.shape[0]
    model = Sequential([
        Dense(24,input_dim=space_size),
        Activation('relu'),
        Dense(20),
        Activation('sigmoid'),
        Dense(action_size),
        Activation('linear')
    ])
    model.compile(loss='mse',
                    optimizer=sgd,
                    metrics = ['accuracy'])
    reward_plot=np.zeros(11)
    reward_sum=0
    for i in tqdm(range(10)):
        reward_plot[i]= reward_sum
        reward_sum=0
        state = env.reset()
        state = np.reshape(state, [1, space_size])
        for j in range(500):
            env.render()
            action=actionchoose(action_size, state)
            print(action)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else 0
            next_state = np.reshape(next_state,[1,space_size])
            tragectory.append((state, action, reward, next_state, done))
             
            state = next_state
            reward_sum = reward_sum + reward
            if done:
                print('end')
                break
            if len(tragectory) > minibatch_size:
                minibatch=random.sample(tragectory, minibatch_size)
                model_=model
                for state, action, reward, next_state, done in minibatch:
                    if done:
                        traget_value_action = reward
                    else:
                        action_target = np.argmax(model.predict(next_state))

                        traget_value_action = reward + gamma * model_.predict(next_state)[0][action_target]
                        print('122344567',action_target,model_.predict(next_state),traget_value_action)
                    target = model.predict(state)
                    target[0][action]=traget_value_action
                    model.fit(state,target,epochs=2, verbose=0)
    plt.plot(reward_plot)
    plt.show()
