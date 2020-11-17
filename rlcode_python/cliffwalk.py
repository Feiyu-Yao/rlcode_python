import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#action0=up
#action1=down
#action2=left
#action3=right


def environment(line,column,action,reward_map):
	if action == 0:
		line += -1
	if action == 1:
		line += 1
	if action == 2:
		column += -1
	if action ==3:
		column += 1
	if line < 0:
		line = 0
	if column < 0:
		column = 0
	if column > 11:
		column = 11
	if line >= 3:
		if column == 0 or column == 11:
			line = 3
			reward = reward_map[line,column]
			print(line,column,reward)
			return line, column, reward
		else:
			line = 3
			column = 0
			reward = -100
			print(line,column,reward)
			return line, column, reward
	else:
		reward = reward_map[line,column]
		print(line,column,reward)
	return line, column, reward


def experiment(value_grid, reward_map):
	alpha = 0.5
	gamma = 1
	line = 3
	column = 0
	if np.random.binomial(1,0.1) == 1:
		action = np.random.choice([0,1,2,3])
	else:
		action = np.random.choice([action_ for action_, value_ in enumerate(value_grid[line,column,:]) if value_ == value_grid[line,column,:].max()])
	reward_sum = 0
	reward = 0
	print('begin')
	while True:
		old_line = line
		old_column = column
		old_action = action
		
		line, column, reward = environment(line,column, action, reward_map)
		reward_sum += reward
		
		if np.random.binomial(1,0.1) == 1:
			action = np.random.choice([0,1,2,3])
		else:
			action = np.random.choice([action_ for action_, value_ in enumerate(value_grid[line,column,:]) if value_ == value_grid[line,column,:].max()])
		value_grid[old_line, old_column, old_action] = value_grid[old_line, old_column, old_action] + alpha * (reward + gamma * value_grid[line, column, :].max() - value_grid[old_line, old_column, old_action])
		if line == 3 and column == 11:
			return reward_sum, value_grid


def run():
	episode = 500
	value_grid = np.zeros([4,12,4])
	reward_map = -1*np.ones([4,12])
	reward_map[3,1:11] = -100
	reward_map[3,11] = 0
	rewardprint=[]
	for i in range(episode):
		reward_sum,value_grid = experiment(value_grid, reward_map) 
		rewardprint.append(reward_sum)
	
	bestaction = value_grid.max(axis=2)
	print(bestaction)
	print(rewardprint)
	plt.plot(rewardprint)
	plt.show()


if __name__=='__main__':
	run()