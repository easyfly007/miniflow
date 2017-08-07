
import numpy as np

class Node(object):
	def __init__(self, inbound_nodes = []):
		self.inbound_nodes = inbound_nodes
		self.outbound_nodes = []
		for n in self.inbound_nodes:
			n.outbound_nodes.append(self)
		self.value = None
		self.gradients = {}
	
	def forward(self):
		raise NotImplementedError
	def backward(self):
		raise NotImplementedError


class Input(Node):
	def __init__(self):
		Node.__init__(self)

	def forward(self, value = None):
		if value is not None:
			self.value = value
	def backward(self):
		self.gradients = {self: 0}
		for n in self.outbound_nodes:
			grad_cost = n.gradients[self]
			self.gradients[self] += grad_cost *1


class Add(Node):
	def __init__(self, x, y):
		Node.__init__(self, [x, y])

	def forward(self):
		self.value = 0.0
		for n in self.inbound_nodes:
			self.value += n.value

class Linear(Node):
	def __init__(self, [inputs, weights, bias]):
		Node.__init__(self, [inputs, weights, bias])
	def forward(self):
		x = np.array(self.inbound_nodes[0].value)
		w = np.array(self.inbound_nodes[1].value)
		b = np.array(self.inbound_nodes[2].value)
		self.value = np.dot(w, x) + b
	def backward(self):
		self.gradients = {n:np.zeros_like(n.value) for n in self.inbound_nodes}
		for n in self.outbound_nodes:
			grad_cost = n.gradients[self]
			self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
			self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
			self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis = 0, keepdims = False)


class Sigmoid(Node):
	def __init__(self, x):
		Node.__init__(self, [x])
	def _sigmoid(self, x):
		return 1.0/(1.0+ np.exp(-x))
	def forward(self):
		x = self.inbound_nodes[0].value
		self.value = self._sigmoid(x)
	def backward(self):
		self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
		for n in self.outbound_nodes:
			grad_cost = n.gradients[self]
			self.gradients[self.inbound_nodes[0]] = self.value * (1. - self.value) * grad_cost

class MSE(Node):
	def __init__(self, y, a):
		Node.__init__(self, [y, a])
	def forward(self):
		y = self.inbound_nodes[0].value.reshape(-1, 1)
		a = self.inbound_nodes[1].value.reshape(-1, 1)
		self.m = self.inbound_nodes[0].value.shape[0]
		self.diff = y - a
		self.value = np.mean(self.diff ** 2)
	def backward(self):
		self.gradients[self.inbound_nodes[0]] = (2. / self.m) * self.diff
		self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff


def forward_pass(output_node, sorted_nodes):
	'''
	perform a forward pass through a list of sorted nodes
	arguments:
		output_node: the output node of the graph (no outgoing edges)
		sorted_nodes: a topologically sorted list of nodes
	returns the output node's value
	'''
	for n in sorted_nodes:
		n.forward()

	return output_node.value

def forward_and_backward(graph):
	for n in graph:
		n.forward()
	for n in graph[::-1]:
		n.backward()

def topological_sort(feed_dict):
	'''
	sort generic nodes in topological order using Kahn's algorithm
	feed_dict: a dictionary where the key is a Input node and the value is the respective value
	returns a list of sorted nodes
	'''
	input_nodes = [n for n in feed_dict.keys()]
	G = {}
	nodes = [n for n in input_nodes]
	while len(nodes) >0:
		n = nodes.pop(0)
		if n not in G:
			G[n] = {'in': set(), 'out': set()}
			for m in n.outbound_nodes:
				if m not in G:
					G[m] = {'in':set(), 'out': set()}
				G[n]['out'].add(m)
				G[m]['in'].add(n)
				nodes.append(m)
	L = []
	S = set(input_nodes)
	while len(S) > 0:
		n = S.pop()
		if isinstance(n, Input):
			n.value = feed_dict[n]
		L.append(n)
		for m in n.outbound_nodes:
			G[n]['out'].remove(m)
			G[m]['input'].remove(n)
			if len(G[m]['in']) == 0:
				S.add(m)
	return L
