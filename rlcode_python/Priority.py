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


def run(valuespace, observation, n, observelist, Time, PQueue):
	alpha = 0.5
	gamma = 0.95
	theta = 0.0001
	state1, state2 = beginstate()
	choosemap = np.zeros([6,9])
	num = 0
	while True:
		Time += 1
		num += 1
		if np.random.binomial(1,0.1) == 1:
			action = np.random.choice([0,1,2,3])
		else:
			action = np.random.choice([action_ for action_, value_ in enumerate(valuespace[state1, state2, :]) if value_ == valuespace[state1, state2, :].max()])
		next_state1, next_state2 = nextstate(state1, state2, action)
		if (next_state1, next_state2, action) not in observelist:
			observelist.append((next_state1, next_state2, action))
		#print(observelist)
		if next_state1 == 0 and next_state2 == 8:
			reward = 1 
		else:
			reward = 0
		observation[state1, state2, action, 0] = reward
		observation[state1, state2, action, 1] = next_state1
		observation[state1, state2, action, 2] = next_state2
		#valuespace[state1, state2, action] = valuespace[state1, state2, action] + alpha * ( reward + gamma * valuespace[next_state1, next_state2,:].max() - valuespace[state1, state2, action])
		if np.abs(reward + gamma * valuespace[next_state1, next_state2,:].max() - valuespace[state1, state2, action]) > theta:
			PQueue[state1, state2, action] = np.abs(reward + gamma * valuespace[next_state1, next_state2,:].max() - valuespace[state1, state2, action])

		if PQueue.any() != 0:
			#print(Time)
			#print('0',np.round(PQueue[:,:,0],5))
			#print('1',np.round(PQueue[:,:,1],5))
			#print('2',np.round(PQueue[:,:,2],5))
			#print('3',np.round(PQueue[:,:,3],5))
			
			for j in range(n):
				np.random.shuffle(observelist)
				(state1_take_set, state2_take_set, action_take_set) = np.where(PQueue == np.max(PQueue))

				setsize = np.size(state2_take_set)
				setchoose = np.random.randint(setsize)
				state1_take = state1_take_set[setchoose]
				state2_take = state2_take_set[setchoose]
				action_take = action_take_set[setchoose]
				print(state1_take,state2_take)
				reward = observation[state1_take, state2_take, action_take, 0]
				state1_takeafter = observation[state1_take, state2_take, action_take, 1]
				state2_takeafter = observation[state1_take, state2_take, action_take, 2]
				PQueue[state1_take, state2_take, action_take] = 0
				#print('after',state1_takeafter,state2_takeafter)
				valuespace[int(state1_take), int(state2_take), int(action_take)] = valuespace[int(state1_take), int(state2_take), int(action_take)] + alpha * ( reward + gamma * valuespace[int(state1_takeafter), int(state2_takeafter), :].max() - valuespace[int(state1_take), int(state2_take), int(action_take)])
				(state1_change_set, state2_change_set, action_change_set) = np.where((observation[:,:,:,1] == state1_take) & (observation[:,:,:,2] == state2_take))
				backnum = np.size(state1_change_set)
				#print(state1_change_set, state2_change_set, action_change_set)
				
				#print(backnum)
				for predict in range(backnum):
					#print(predict)
					#print(state1_change_set[predict], state2_change_set[predict], action_change_set[predict])
					#print(observation[state1_change_set[predict], state2_change_set[predict], action_change_set[predict], 0])
					if np.abs(observation[state1_change_set[predict], state2_change_set[predict], action_change_set[predict], 0] + gamma * valuespace[state1_take, state2_take, :].max() - valuespace[state1_change_set[predict], state2_change_set[predict], action_change_set[predict]]) > theta:
						PQueue[state1_change_set[predict], state2_change_set[predict], action_change_set[predict]] = np.abs(observation[state1_change_set[predict], state2_change_set[predict], action_change_set[predict], 0] + gamma * valuespace[state1_take, state2_take, :].max() - valuespace[state1_change_set[predict], state2_change_set[predict], action_change_set[predict]])
						print(state1_change_set[predict], state2_change_set[predict], action_change_set[predict])
				#print('0',np.round(PQueue[:,:,0],5))
				#print('1',np.round(PQueue[:,:,1],5))
				#print('2',np.round(PQueue[:,:,2],5))
				#print('3',np.round(PQueue[:,:,3],5))
			

			if next_state1 == 0 and next_state2 == 8:
				#print(choosemap)
				#break
				return valuespace, observation, observelist, num, choosemap, Time, PQueue
		state1 = next_state1
		state2 = next_state2
		
		#print(num)
		#print(next_state1,next_state2)
		#print(state1, state2)
		#print(num)
		#print(observelist)
		choosemap[state1][state2] = choosemap[state1][state2]+1
	#print(choosemap)







def begin(episode, n):
	valuespace = np.zeros([6,9,4])
	observation = np.zeros([6,9,4,3])
	observelist = []
	PQueue = np.zeros([6,9,4])
	choosemap = np.zeros([6,9])
	NUM = []
	Time = 0
	for i in range(episode):
		valuespace, observation, observelist, num, choosemap, Time, PQueue = run(valuespace, observation, n, observelist, Time, PQueue)
		NUM.append(num)
	#print(valuespace)
	print(NUM)
	#print('0',np.round(PQueue[:,:,0],5))
	#print('1',np.round(PQueue[:,:,1],5))
	#print('2',np.round(PQueue[:,:,2],5))
	#print('3',np.round(PQueue[:,:,3],5))
	#print(choosemap)
	#print(valuespace.argmax(axis=2))
	#print(Time)

if __name__ == '__main__':
	begin(30, 3)




