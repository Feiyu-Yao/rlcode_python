import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 



# goal
GOAL = 100

# all states, including state 0 and state 100
STATES = np.arange(GOAL + 1)


ph=0.55


def figure4_3():
	state_value=np.zeros(101)
	state_value[100]=1


	while True:
		old_value=state_value.copy()
		for stat in np.arange(100):
			state=stat+1
			action_returns =[]
			for action in np.arange(min(state,100-state)+1):
				action_returns.append(ph*state_value[state+action]+(1-ph)*state_value[state-action])
			#print(state)
			state_value[state]=np.max(action_returns)
		max_value_change = abs(state_value-old_value).max()
		print(state_value)
		if max_value_change < 1e-9:
			break
	policy=np.zeros(101)
	for stat in np.arange(100):
		state=stat+1
		action__return=[]
		for action in np.arange(min(state,100-state)+1):
			action__return.append(ph*state_value[state+action]+(1-ph)*state_value[state-action])
		policy[state]=np.argmax(action__return)

	plt.plot(state_value)
	plt.show()

if __name__ == '__main__':
	figure4_3()





