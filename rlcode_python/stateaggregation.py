

import numpy as np
N_state = 1000
END_STATE = [0,N_state+1]
STEP_SIZE = 100



def get_action():
	if np.random.binomial(1,0.5) == 1:
		action = 1
	else:
		action = -1

	#print(action)
	return action

def step(state, action):
	gap = np.random.randint(1, STEP_SIZE+1)
	gap = gap * action
	state_after_gap = state + gap
	reward = 0
	if state_after_gap < 1:
		state_after_gap = 0
		reward = -1
	if state_after_gap > N_state:
		state_after_gap = N_state+1 
		reward = 1
	return state_after_gap, reward



def gradient_MC(value_function):
	alpha = 0.00002
	state = 500
	tragectory = [state]
	while state not in END_STATE:
		action = get_action()
		next_state, reward = step(state, action)
		tragectory.append(next_state)
		state = next_state
	#print(tragectory)

	for state in tragectory[:-1]:
		delta = alpha * (reward - value_function.value(state))
		#print(delta, state)
		value_function.update(delta, state)
	return value_function









class value_function:
	def __init__(self, num_of_groups):
		self.num_of_groups = num_of_groups
		self.group_size = N_state // num_of_groups
		self.params = np.zeros(num_of_groups)

	def value(self, state):
		if state in END_STATE:
			return 0
		group_index = (state - 1) // self.group_size
		return self.params[group_index]

	def update(self, delta, state):
		group_index = (state-1) // self.group_size
		self.params[group_index] += delta
		#print(delta, state, self.params)

	

if __name__ == '__main__':
	episode = 100000
	value_function = value_function(10)
	for i in range(episode):
		value_function = gradient_MC(value_function) 
	print(value_function.params)


