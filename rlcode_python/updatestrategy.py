import numpy as np
from tqdm import tqdm 
import matplotlib
import matplotlib.pyplot as plt

MAX_STEPS = 20000
checkpoint_num = 100
ACTION = [0, 1]
terminate_prob = 0.1


class Task():
	def __init__(self, n_states, b):
		self.n_states = n_states
		self.b = b
		self.transition = np.random.randint(n_states, size=(n_states, len(ACTION), b))
		self.reward = np.random.randn(n_states, len(ACTION), b)
	def step(self, state, action):
		if np.random.binomial(1, 0.1) == 1:
			return self.n_states, 0
		next = np.random.randint(self.b)
		aaa = self.transition
		bbb = self.reward
		#print(next)
		#print(aaa,bbb)
		return aaa[state, action, next], bbb[state, action, next]


def evaluate(q, tasks):
	#print(q)
	run = 100
	rewardcolumn = []
	for i in range(run):
		state = 0
		reward_every_run = 0
		while state != tasks.n_states:
			action = np.random.choice([action_ for action_, value in enumerate(q[state]) if value == np.max(q[state]) ])
			state, r = tasks.step(state, action)
			reward_every_run += r
			#print(r)
		rewardcolumn.append(reward_every_run)
	#print(rewardcolumn)
	return np.mean(rewardcolumn)





def uniform(tasks, EVAL):
	performance = []
	q = np.zeros((tasks.n_states,2))
	for steps in tqdm(range(MAX_STEPS)):
		state = steps // len(ACTION) % tasks.n_states
		action = steps % len(ACTION)
		next_state = tasks.transition[state, action]
		q[state, action] = (1 - terminate_prob) * np.mean( tasks.reward[state, action] + np.max(q[next_state],axis = 1))
		if steps%EVAL == 0:
			vi = evaluate(q, tasks)
			performance.append([steps, vi])
	return zip(*performance)


def on_policy(tasks, EVAL):
	epson = 0.1
	performance = []
	run = 100
	state = 0
	#print(tasks)
	q = np.zeros((tasks.n_states,2))
	for steps in tqdm(range(MAX_STEPS)):
		#print(state)
		if np.random.rand() < epson:
			action = np.random.choice(ACTION)
		else:
			#print(state,np.size(q))
			action = np.random.choice([action_ for action_, value_ in enumerate(q[state]) if value_ == np.max(q[state])])
		next_state = tasks.transition[state, action]
		#print(next_state)
		#print(tasks.reward())
		#kkk=0
		#print(np.max(q[next_state],axis = 1))
		q[state, action] = (1 - terminate_prob ) * np.mean(tasks.reward[state, action] + np.max(q[next_state], axis = 1))
		state, _ = tasks.step(state, action)
		if state == tasks.n_states:
			state = 0
		if steps%EVAL == 0:
			vi = evaluate(q, tasks)
			#print(vi)
			performance.append([steps, vi])
	return zip(*performance)

















def setting():
	state_num = [1000, 10000]
	branch = [1, 3, 10]

	average_time = 30
	methods = [on_policy, uniform]
	for i, n in enumerate(state_num):
		plt.subplot(2, 1, i+1)
		for b in branch:
			tasks = [Task(n, b) for _ in range(average_time)]
			for method in methods:
				value = []
				for task in tasks:


					steps, v = method(task, MAX_STEPS/ checkpoint_num)
					#print(v)
					value.append(v)
				value = np.mean(np.asarray(value), axis = 0)
				plt.plot(steps, value, label='b = %d, %s' % (b, method.__name__))
		plt.title('%d states' %(n))

		plt.ylabel('value of start state')
		plt.legend()

	plt.subplot(2, 1, 2)
	plt.xlabel('computation time, in expected updates')
	plt.savefig('../figure_8_8.png')
	plt.show()


if __name__ == '__main__':
	setting()




