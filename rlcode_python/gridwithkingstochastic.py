import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#action0=left
#action1=up
#action2=right
#action3=down
#action4=left_up
#action5=left_down
#action6=right_up
#action7=right_down
def environment(stateline, statecolumn, wind, action):
	kkk = np.random.choice([1,2,3])
	if kkk == 1:
		stateline -= 1
	if kkk == 3:
		stateline += 1
	if action == 0:
		statecolumn -= 1
	if action == 1:
		stateline -= 1
	if action == 2:
		statecolumn += 1
	if action == 3:
		stateline += 1
	if action == 4:
		stateline -= 1
		statecolumn -=1
	if action == 5:
		stateline +=1
		statecolumn -=1
	if action == 6:
		stateline -= 1
		statecolumn += 1
	if action == 7:
		stateline += 1
		statecolumn += 1
	if statecolumn < 0:
		statecolumn = 0
	if statecolumn > 9:
		statecolumn = 9
	if stateline > 6:
		stateline = 6
	if stateline < 0:
		stateline = 0
	return stateline, statecolumn
	



def action_take_random():
	actionrand = np.random.randint(0,4)
	return actionrand



























def play(wind,valuestate,Reward):
	alpha = 0.2
	gamma = 0.5
	stateline = 3
	statecolumn = 0
	time=0
	if np.random.binomial(1,0.1) == 1:
		action = np.random.choice([0,1,2,3])
	else:
		action = np.random.choice([action_ for action_, value_ in enumerate(valuestate[stateline, statecolumn, :]) if value_ == valuestate[stateline, statecolumn, :].max()])
	print('begin')
	while True:
		old_action = action
		#state_action_pair = [stateline, statecolumn, action]
		old_stateline = stateline
		old_statecolumn = statecolumn
		stateline, statecolumn = environment(stateline, statecolumn, wind, action)
		#if np.random.binomial(1,0.1) == 1:
		#	action = np.random.choice([0,1,2,3])
		#else:
		action = np.random.choice([action_ for action_, value_ in enumerate(valuestate[stateline, statecolumn, :]) if value_ == valuestate[stateline, statecolumn, :].max()])
		print(stateline,statecolumn,action)
		reward = Reward[stateline, statecolumn]
		time += 1
		valuestate[old_stateline, old_statecolumn, old_action] = valuestate[old_stateline, old_statecolumn, old_action] + alpha * \
			(reward + gamma * valuestate[stateline, statecolumn, action] - valuestate[old_stateline, old_statecolumn, old_action])
		if stateline == 3 and statecolumn == 7:
			return valuestate, time
		


def begin():
	wind = [0,0,0,1,1,1,2,2,1,0]
	valuestate = np.zeros([7,10,8])
	Reward = -1 * np.ones([7,10])
	Reward[3,7] = 0
	lll=[]
	Time=[]
	oldmax=valuestate.max(axis=2)
	tragety=[]
	for i in range(20000):
		valuestate, time = play(wind, valuestate, Reward)
		valueaction = valuestate.max(axis=2)

		Time.append(time)
		lll.append((sum(sum(abs(np.array(oldmax) - np.array(valueaction))))))
		if sum(sum(abs(np.array(oldmax) - np.array(valueaction)))) < 1e-2:
			return valuestate, lll, Time


	return valuestate, lll, Time

if __name__=='__main__':
	valuestate,lll, Time = begin()
	bestaction = valuestate.argmax(axis=2)
	valueaction = valuestate.max(axis=2)
	print(bestaction)
	print(valuestate)
	print(Time)
	plt.plot(Time)
	plt.show()