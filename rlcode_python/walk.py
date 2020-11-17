import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 

#A=1,B=2,C=3,D=4,E=5
#
#

#state = [1,2,3,4,5]
#action = [-1,1]
def actionrand():
	actionrandn=np.random.randint(0,2)
	if actionrandn == 0:
		actionrandn = -1
	return actionrandn

def play(value, Reward, value_track, alpha):

	state = 2
	while True:
		action = actionrand()

		old_state = state
		state += action

		reward = 0
		if state == 6:
			reward = 1
		Reward.append(reward)
		value[old_state] = value[old_state] + alpha * ( reward + value[state] - value[old_state])
		value_track[old_state].append(value[old_state])
		if state == 0 or state == 6:
			return value, Reward, value_track






def begin():
	alpha = 0.1
	value_track=[[]for i in range(7)]
	value=0.5*np.ones(7)
	value[0] = 0
	value[6] = 0
	Reward = []
	for i in range(500000):
		value, Reward, value_track = play(value, Reward, value_track, alpha)
	return value

if __name__=='__main__':
	value=begin()
	print(value)
	plt.plot(value)
	plt.show()



