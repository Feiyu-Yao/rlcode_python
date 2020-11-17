import numpy as np

#action0 = up
#action1 = down
#action2 = left 
#action3 = right 


def nextstate(state1, state2, action):
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
	if newstate1 > 5 or newstate1 < 0:
		newstate1 = state1
	if newstate2 > 8 or newstate2 < 0:
		newstate2 = state2
	if (newstate1 == 1 or newstate1 == 2 or newstate1 == 3) and newstate2 == 2:
		newstate1 = state1
		newstate2 = state2
	if (newstate1 == 0 or newstate1 == 1 or newstate1 == 2) and newstate2 == 7:
		newstate1 = state1
		newstate2 = state2
	if newstate1 == 4 and newstate2 == 5:
		newstate1 = state1
		newstate2 = state2

	return newstate1, newstate2








def beginstate():
	state1 = 2
	state2 = 0
	return state1, state2


def run(valuespace, observation, n, observelist, checklist, Time, statenum):
	alpha = 0.1
	gamma = 0.95
	ka = 0.0000001
	tor = 0
	state1, state2 = beginstate()
	choosemap = np.zeros([6,9])
	num = 0
	while True:
		Time += 1
		if np.random.binomial(1,0) == 1:
			action = np.random.choice([0,1,2,3])
		else:
			valuechoose = valuespace[state1, state2, :] + ka * np.sqrt(Time*np.ones(4) - checklist[state1, state2, :]) 
			action = np.random.choice([action_ for action_, value_ in enumerate(valuechoose) if value_ == valuechoose.max()])
			#####action = np.random.choice([action_ for action_, value_ in enumerate(valuespace[state1, state2, :]) if value_ == valuespace[state1, state2, :].max()])
		next_state1, next_state2 = nextstate(state1, state2, action)
		if (state1, state2, action) not in observelist:
			observelist.append((state1, state2, action))
		if (state1, state2) not in statenum:
			statenum.append((state1, state2))
		#print(observelist)
		if next_state1 == 0 and next_state2 == 8:
			reward = 1
		else:
			reward = 0
		tor = Time - checklist[state1, state2, action]
		checklist[state1, state2, action] = Time
		reward = reward + ka * np.sqrt(tor)
		observation[state1, state2, action, 0] = reward
		observation[state1, state2, action, 1] = next_state1
		observation[state1, state2, action, 2] = next_state2
		valuespace[state1, state2, action] = valuespace[state1, state2, action] + alpha * ( reward + gamma * valuespace[next_state1, next_state2,:].max() - valuespace[state1, state2, action])
		for j in range(n):
			np.random.shuffle(observelist)
			np.random.shuffle(statenum)
			state1_take, state2_take = statenum[0]
			actionrand = np.random.choice([0,1,2,3])
			#print('n',state1_take, state2_take)
			if (state1_take, state2_take, actionrand) in observelist:
				#print('IN')
				action_take = actionrand
				reward = observation[state1_take, state2_take, action_take, 0]
				state1_takeafter = observation[state1_take, state2_take, action_take, 1]
				state2_takeafter = observation[state1_take, state2_take, action_take, 2]
				#print(state1_takeafter,state2_takeafter)
				valuespace[int(state1_take), int(state2_take), int(action_take)] = valuespace[int(state1_take), int(state2_take), int(action_take)] + alpha * ( reward + gamma * valuespace[int(state1_takeafter), int(state2_takeafter), :].max() - valuespace[int(state1_take), int(state2_take), int(action_take)])
			else:
				#print('OUT')
				#kkk = alpha * ( + gamma * valuespace[int(state1_take), int(state2_take), :].max() - valuespace[int(state1_take), int(state2_take), actionrand])
				#print(kkk)
				valuespace[int(state1_take), int(state2_take), actionrand] = valuespace[int(state1_take), int(state2_take), actionrand] + alpha * ( + gamma * valuespace[int(state1_take), int(state2_take), :].max() - valuespace[int(state1_take), int(state2_take), actionrand])
		if next_state1 == 0 and next_state2 == 8:
			print('choosemap')
			return valuespace, observation, observelist, num, choosemap, checklist, Time, statenum
		state1 = next_state1
		state2 = next_state2
		num += 1
		#Time += 1
		#print(num)
		#print(next_state1,next_state2)
		#print(state1, state2)
		#print(Time)
	
		#print(observelist)
		choosemap[state1][state2] = choosemap[state1][state2]+1
	#print(choosemap)
	#print(valuespace[2,0,:])
	#print(checklist)







def begin(episode, n):
	Time = 0
	valuespace = np.zeros([6,9,4])
	observation = np.zeros([6,9,4,3])
	observelist = []
	checklist = np.zeros([6,9,4])
	choosemap = np.zeros([6,9])
	statenum = []
	NUM = []
	for i in range(episode):
		valuespace, observation, observelist, num, choosemap, checklist, Time, statenum = run(valuespace, observation, n, observelist, checklist, Time, statenum)
		NUM.append(num)
	#print(valuespace)
	print(NUM)
	print(choosemap)
	print(Time)
	#print(valuespace.argmax(axis=2))
	#print(checklist)

if __name__ == '__main__':
	begin(20, 10)
