import numpy as np
import matplotlib.pyplot as plt

stategy_hit=0
stategy_stand=1
ACTION=[stategy_hit,stategy_stand]
policy_player=np.zeros(22)
for i in range(12,20):
	policy_player[i] = stategy_hit
policy_player[20]=stategy_stand
policy_player[21]=stategy_stand

policy_dealer=np.zeros(22)
for i in range(12,17):
	policy_dealer[i]=stategy_hit
for i in range(17,22):
	policy_dealer[i]=stategy_stand


def hit():
	hitnum=np.random.randint(1,14)
	if hitnum > 10:
		hitnum = 10
	return hitnum

def usefulace(ace):
	if ace ==1:
		return 11
	else:
		return ace

def play(actions):
	#initiation player
	play_tragetory=[]
	playersum=0
	playeruseace=False
	playeracenumber=0
	playeruseacenumber=0
	while playersum < 12:
		card = hit()
		if card == 1:
			playeracenumber+=1
		card = usefulace(card)
		playersum+=card
		if playersum > 21:
			print(playersum)
			playeruseace = True
			playeruseacenumber+=1
			playersum-=10

	#initiation dealer
	dealersum=0
	dealeruseace=False
	dealeracenumber=0
	dealeruseacenumber=0
	dealercard1=hit()
	dealercard2=hit()
	dealersum=usefulace(dealercard1)+usefulace(dealercard2)
	if dealercard1 == 1:
		dealeracenumber+=1
	if dealercard2 == 1:
		dealeracenumber+=1
		dealeruseace=True
		dealersum-=10


	#start the game
	nnn=0
	while True:
		nnn+=1
		n=0
		
		action=actions[playersum-12,dealercard1-1]
		play_tragetory.append([playersum,dealercard1,action])
		if action == 0:
			card=hit()
			if card ==1:
				playeracenumber+=1
			card = usefulace(card)
			playersum+=card
			n=playeracenumber-playeruseacenumber
			#print(n)
			
			if playersum > 21:
				if n != 0:
					playersum-=10
					playeruseacenumber+=1
					if playersum>21:

						return -1, play_tragetory
					print(playersum,n)
				else:
					return -1, play_tragetory
					break
		else:
			break



	while True:
		if dealersum < 17:
			card = hit()
			if card ==1:
				dealeracenumber+=1
			card = usefulace(card)
			dealersum+=card
			m = dealeracenumber-dealeruseacenumber
			if dealersum > 21:
				if m != 0:
					dealersum-=10
					m-=1
				else:
					return 1, play_tragetory
		else:
			break

	if dealersum > playersum:
		return -1, play_tragetory
	if dealersum == playersum:
		return 0, play_tragetory
	if dealersum < playersum:
		return 1, play_tragetory















def montecarlo(episode):
	value=np.zeros((10,10,2))
	valuecount=np.ones((10,10,2))
	rewardone=0
	rewardzero=0
	rewardminus=0 

	def greedy(value):
		actions=value.argmax(axis=2)
		return actions

	for i in range(episode):
		actions=greedy(value)
		reward, play_tragetory = play(actions)
		first_visit=set()
		for playersum, dealercard1, action in play_tragetory:
			state_action=(playersum, dealercard1, action)
			if state_action in first_visit:
				continue
			first_visit.add(state_action)
			value[playersum-12, dealercard1-1, action] += reward
			valuecount[playersum-12, dealercard1-1, action] += 1
			print(reward)
			if reward == 1:
				rewardone+=1
			if reward == 0:
				rewardzero+=1
			if reward == -1:
				rewardminus+=1


	return value/valuecount, rewardone, rewardzero, rewardminus




if __name__ == '__main__':
	qqq, ppp, jjj, kkk=montecarlo(500000)
	print(np.max(qqq,axis=2),ppp, jjj, kkk)









