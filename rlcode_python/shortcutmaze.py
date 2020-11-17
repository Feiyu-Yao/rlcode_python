import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def environment(state1, state2, action, Time):
	#action0 == up
	#action1 == down
	#action2 == left
	#action3 == right


	if Time < 3000:
		Obstruct = [1,2,3,4,5,6,7,8]
	else:
		Obstruct = [1,2,3,4,5,6,7]
	if action == 0:
		newstate1 = state1 - 1
		newstate2 = state2
	if action == 1:
		newstate1 = state1 + 1
		newstate2 = state2
	if action == 2:
		newstate1 = state1
		newstate2 = state2 - 1
	if action == 3:
		newstate1 = state1
		newstate2 = state2 + 1
	if newstate1 < 0 or newstate1 > 5:
		newstate1 = state1
	if newstate2 < 0 or newstate2 > 8:
		newstate2 = state2  
	if newstate1 == 3 and (newstate2 in Obstruct):
		newstate1 = state1
		newstate2 = state2
	return newstate1, newstate2
def run(Time, valuespace, observation, n, observationlist, plotreward, cumulativereward):
	alpha = 1
	gamma = 0.95
	num = 0
	state1 = 5
	state2 = 3
	while True:
		Time += 1
		num += 1
		#print(Time)
		#print(state1, state2)
		if np.random.binomial(1,0.1) == 1:
			action = np.random.choice([0,1,2,3])
		else:
			action = np.random.choice([action_ for action_, value_ in enumerate(valuespace[state1, state2, :]) if value_ == valuespace[state1, state2, :].max()])
		next_state1, next_state2 = environment(state1, state2, action, Time)
		#print(next_state1, next_state2)
		if (state1, state2, action) not in observationlist:
			observationlist.append((state1, state2, action))
		if next_state1 == 0 and next_state2 == 8:
			reward = 1
		else:
			reward = 0
		observation[state1, state2, action, 0] = reward
		observation[state1, state2, action, 1] = next_state1
		observation[state1, state2, action, 2] = next_state2
		valuespace[state1, state2, action] = valuespace[state1, state2, action] + alpha * ( reward + gamma * valuespace[next_state1, next_state2, :].max() - valuespace[state1, state2, action])
		for j in range(n):
			np.random.shuffle(observationlist)
			state1_in_planning, state2_in_planning, action_in_planning = observationlist[0]
			reward_in_planning = observation[state1_in_planning, state2_in_planning, action_in_planning, 0]
			new_state1_in_planning = observation[state1_in_planning, state2_in_planning, action_in_planning, 1]
			new_state2_in_planning = observation[state1_in_planning, state2_in_planning, action_in_planning, 2]
			valuespace[state1_in_planning, state2_in_planning, action_in_planning] = valuespace[state1_in_planning, state2_in_planning, action_in_planning] + alpha * ( reward_in_planning + gamma * valuespace[int(new_state1_in_planning), int(new_state2_in_planning), :].max() - valuespace[state1_in_planning, state2_in_planning, action_in_planning])
		plotreward.append(cumulativereward)
		state1 = next_state1
		state2 = next_state2
		if next_state1 == 0 and next_state2 == 8:
			cumulativereward += 1
			return valuespace, observation, Time, observationlist, plotreward, cumulativereward, num

def begin():
	n = 50
	NUM = []
	Time = 0
	plotreward = []
	cumulativereward = 0
	observationlist = []
	valuespace = np.zeros([6,9,4])
	observation = np.zeros([6,9,4,3])
	while Time < 6000:
		valuespace, observation, Time, observationlist, plotreward, cumulativereward, num = run(Time, valuespace, observation, n, observationlist, plotreward, cumulativereward)
		NUM.append(num)
		#print(NUM)
	return plotreward[0:6000]

	#plt.plot(plotreward)
	#plt.xlabel('-')
	#plt.show()

if __name__ == '__main__':
	PLOTR = np.zeros([6000,3])
	for i in range(3):
		PLOTR[:,i]=begin()
		#print(i)
		print(PLOTR[:,i])
	plt.plot(np.sum(PLOTR,axis=1)/3)
	plt.show()