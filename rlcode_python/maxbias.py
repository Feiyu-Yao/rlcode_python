import numpy as np

#action0=left
#action1=right

def rewardfun(action):
	if action == 0:
		reward1 = np.random.normal(-0.1, 1)
		reward2 = np.random.normal(-0.1, 1)
		reward3 = np.random.normal(-0.1, 1)
		reward4 = np.random.normal(-0.1, 1)
		reward5 = np.random.normal(-0.1, 1)
		reward = np.array([reward1,reward2,reward3,reward4,reward5]).max()
	if action == 1:
		reward = 0
	return reward

def run(value):
	alpha = 0.1
	reward = 0
	if np.random.binomial(1,0.1) == 1:
		action = np.random.choice([0, 1])
	else:
		action = value.argmax()
	reward = rewardfun(action)
	value[action] = value[action] + alpha * (reward - value[action])
	return value





def experiment():
	value = np.zeros(2)
	for i in range(50000):
		value = run(value)
	return value

if __name__=='__main__':
	value = experiment()
	print(value)