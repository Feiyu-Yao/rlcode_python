import numpy as np
import matplotlib
import matplotlib.pyplot as plt

value = []
value.append(0)
value_count=0
REWARDD=[]
def play():
	play_tragetray=[]
	while True:
		policy_behavior=np.random.binomial(1,0.5)
		#0-left 1-right
		if policy_behavior == 0:
			environment_left=np.random.binomial(1,0.9)
			#0.9-return 0.1-terminate
			if environment_left == 1:
				play_tragetray.append(policy_behavior)
			if environment_left == 0:
				reward = 1
				return reward, play_tragetray
		if policy_behavior == 1:
			reward = 0
			return reward, play_tragetray

def experiment(episode):
	value_count=0
	for i in range(episode):
		reward, play_tragetray = play()
		first_happen=set()
		value_countepi=0

		valuepi = 0
		for action in play_tragetray:
			if action in first_happen:
				continue 
			#first_happen.add(action)
			value_countepi+=1
			rio=1.0/pow(0.5, value_countepi)
			valuepi=rio*reward
			value_count+=1

			REWARDD.append(valuepi)
			lll=np.sum(REWARDD)
			kkk=lll/value_count
			value.append(kkk)

			
	plt.plot(value)
	plt.xlabel('Episodes (log scale)')
	plt.ylabel('Ordinary Importance Sampling')
	plt.xscale('log')
	plt.show()
	print(REWARDD)

if __name__ == "__main__":
	experiment(5000)









