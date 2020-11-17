import numpy as np

def judge(state):
	if state == 0:
		reward = -1
		return reward 
	if state == 20:
		reward = 1
	else:
		reward = 0
	return reward


def chooseaction():
	if np.random.binomial(1,0.5) == 1:
		action = 1
	else:
		action = -1
	return action


def run(value):
	n = 4
	alpha = 0.4
	GAMMA = 1
	state = 10
	actionnum = 0
	State = []
	IFEND = 0
	State.append(state)
	Reward = []

	while True:
		if IFEND == 0:
			action = chooseaction()
			actionnum += 1
			next_state = state + action
			state = next_state
			reward = judge(next_state)
			Reward.append(reward)
			State.append(next_state)
		if actionnum >= n:
			if IFEND == 0:
				rewardmcsum = 0
				for i in range(actionnum - n, actionnum):
					rewardmcsum += pow(GAMMA, i - actionnum + n  ) * Reward[i] 
					#print(i - actionnum + n)
					#print('what',i)
				value[State[actionnum-n]] = value[State[actionnum-n]] + alpha * ( rewardmcsum + pow(GAMMA, n) * value[State[actionnum]] - value[State[actionnum-n]])
				#N_STATESprint(actionnum - n)
			if IFEND == 1:
				print('stop')
				for j in range(T - n + 1, T):
					rewardmcsum = 0
					for i in range(j , T):
						rewardmcsum += pow(GAMMA, i - T + n - 1 ) * Reward[i] 
					
						#print('i',i)
						#print(i - T + n)
					#print('T',T)
					#print('j',j)
					value[State[j]] = value[State[j]] + alpha * ( rewardmcsum + pow(GAMMA, T - j) * value[State[T]] - value[State[j]] )
					#print(j)
				return value

		if next_state == 0 or next_state == 20:
			IFEND = 1
			T = actionnum
			#print(Reward)
			#print(State)
	


def begin():
	episode = 3000
	value = np.zeros(21)
	value[0] = 0
	TRUE_VALUE = np.arange(-20, 22, 2) / 20.0
	TRUE_VALUE[0] = 0
	TRUE_VALUE[20] = 0	
	value[20] = 0
	for i in range(episode):
		value = run(value)
		print(np.sqrt(sum(np.power(value - TRUE_VALUE,2)/19)))
	#print(type(TRUE_VALUE))
	#print(type(sum(sum(abs(value - TRUE_VALUE)))))
	#print(value)
	



if __name__ == '__main__':
	begin()
