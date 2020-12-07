class sumtree:
	def __init__(self.capacity):
		self.write = 0
		self.capacity = capacity
		self.tree = np.zeros(2*capacity -1)
		self.data = np.zeros(capacity, dtype=object)
	def propagate(self, change, idx):
		parent = (idx-1)//2
		self.tree[parent]=+change
		if parent !=0:
			propagate(change,parent)
	def update(self,p, idx):
		change = p - self.tree[idx]
		self.tree[idx] = p
		propagate(change,idx)
	def add(self, p, transition):
		self.data[write]=transition
		idx = capacity-1+write 
		self.tree[idx]=p
		self.write =+1
		if self.write > capacity:
			write = 0 
	def retrive(self,idx,s)
		left_leave = 2 * idx +1
		right_leave = 2*idx +2
		if left_leave >= len(self.tree)
			return idx
		if self.tree[left_leave]>=s:
			retrive(left_leave,s)
		else:
			retrive(right_leave, s)
	def get(self, s)
		idx = retrive(0, s)
		return self.tree[idx], self.data[idx - capacity+1]

	def total(self):
		return self.tree[0]