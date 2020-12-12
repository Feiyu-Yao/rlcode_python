import numpy as np
import gym
import random
from collections import deque
from keras.models import Model
from keras.layers import Dense, Activation, Add, Subtract, Lambda, Input
from keras.optimizers import SGD 
from tqdm import tqdm
import keras.backend as K



def actionchoose(action_size,state):
	if np.random.binomial(1,0.1) == 1:
		return random.randrange(action_size)
	action_value = model.predict(state)
	return np.argmax(action_value)



if __name__ == "__main__":
	tragectory = deque(maxlen=1000)
	minibatch_size = 30
	gamma = 0.95
	sgd = 	SGD(lr=0.01, decay=1e-6,momentum=0.9, nesterov=True)
	env = gym.make('CartPole-v1')
	action_size = env.action_space.n
	space_size = env.observation_space.shape[0]
	inputs = Input(shape=(4,))
	x =Dense(16, activation='relu')(inputs)
	x =Dense(16,activation='linear')(x)
	value = Dense(2, activation='linear')(x)
	a = Dense(2,activation='linear')(x)
	mean_value = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
	advantage = Subtract()([a, mean_value])
	q = Add()([value, advantage])
	model = Model(inputs=inputs, outputs=q)




	model.compile(loss='mse',
					optimizer=sgd,
					metrics = ['accuracy'])
	for i in tqdm(range(1000)):
		state = env.reset()
		state = np.reshape(state, [1, space_size])
		for j in range(500):
			env.render()
			action=actionchoose(action_size, state)
			print(action)
			next_state, reward, done, _ = env.step(action)
			reward = reward if not done else -10
			next_state = np.reshape(next_state,[1,space_size])
			tragectory.append((state, action, reward, next_state, done))
			 
			state = next_state
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
						traget_value_action = reward + gamma * np.max(model_.predict(next_state))
					target = model.predict(state)
					target[0][action]=traget_value_action
					model.fit(state,target,epochs=2, verbose=0)

