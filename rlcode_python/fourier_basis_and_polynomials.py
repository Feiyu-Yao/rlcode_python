import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
ACTIONS = [-1, 1]
def compute_true_value():
    # true state value, just a promising guess
    true_value = np.arange(-1001, 1003, 2) / 1001.0

    # Dynamic programming to find the true state values, based on the promising guess above
    # Assume all rewards are 0, given that we have already given value -1 and 1 to terminal states
    while True:
        old_value = np.copy(true_value)
        for state in STATES:
            true_value[state] = 0
            for action in ACTIONS:
                for step in range(1, 100 + 1):
                    step *= action
                    next_state = state + step
                    next_state = max(min(next_state, N_STATES + 1), 0)
                    # asynchronous update for faster convergence
                    true_value[state] += 1.0 / (2 * 100) * true_value[next_state]
        error = np.sum(np.abs(old_value - true_value))
        if error < 1e-2:
            break
    # correct the state value for terminal states to 0
    true_value[0] = true_value[-1] = 0

    return true_value


N_STATES = 1000
END_STATES = [0,N_STATES+1]
runs = 1
episodes = 5000
STATES = np.arange(1, N_STATES + 1)
orders = [5, 10,20]

alphas = [1e-3, 5e-3]

errors = np.zeros((len(alphas), len(orders), episodes))


def action_get():
	if np.random.binomial(1,0.5) == 1:
		return 1
	else:
		return -1
def step(state, action):
	gap = np.random.randint(0, N_STATES+1)
	gap = action * gap
	next_state = state + gap
	if next_state < 1:
		next_state = 0
		rewardget = -1
	elif next_state > N_STATES:
		next_state = N_STATES+1
		rewardget = 1
	else:
		rewardget = 0
	return next_state, rewardget





def gradient_MC(valuefunction, alpha):
	state = 500
	tragetory = [state]
	while state not in END_STATES:
		action = action_get()
		next_state, rewards = step(state, action)
		tragetory.append(next_state)
		state = next_state
	for state in tragetory[:-1]:
		delta = alpha*(rewards - valuefunction.value(state))
		print('delta',delta)
		valuefunction.update(delta, state)







class BaseValueFunction():
	def __init__(self, order, type):
		self.order = order
		self.weights = np.zeros(order + 1)
		self.bases = []
		if type == 'Poly':
			for i in range(0, order + 1):
				self.bases.append(lambda s, i = 1: pow(s,i))
		elif type == 'Fourier':
			for i in range(0, order + 1):
				self.bases.append(lambda s, i = i: np.cos(i*np.pi*s))

	def value(self, state):
		state /= float(N_STATES)
		feature = np.asarray([func(state) for func in self.bases])
		return np.dot(self.weights, feature)

	def update(self, delta,state):
		state /= float(N_STATES)
		deviative_value = np.asarray([func(state) for func in self.bases])
		self.weights += delta * deviative_value

if __name__ == '__main__':
	true_value = compute_true_value()
	labels = [['polynomial basis'] * 3, ['fourier basis'] * 3]
	for run in range(runs):
		for i in range(1,2):
			value_functions = [BaseValueFunction(orders[i], 'Poly'), BaseValueFunction(orders[i], 'Fourier')]
			for j in range(1,2):
				for episode in tqdm(range(episodes)):
					gradient_MC(value_functions[j], alphas[j])
					state_values = [value_functions[j].value(state) for state in STATES]
					errors[j, i, episode] += np.sqrt(np.mean(np.power(true_value[1: -1] - state_values, 2)))
                    # get the root-mean-squared error
				aaaaa = state_values
				print(true_value[1:-1])
				
	print(aaaaa)
	print(errors)
	errors /= runs
	for i in range(1,2):
		for j in range(1,2):
			plt.plot(errors[i, j, :], label='%s order = %d' % (labels[i][j], orders[j]))
	plt.xlabel('Episodes')
	plt.show()

				#print([value_functions[j].value(state) for state in STATES])


