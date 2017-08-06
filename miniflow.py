
import numpy as np

class Node(object):
	def __init__(self, inbound_nodes = []):
		self.inbound_nodes = inbound_nodes
		self.outbound_nodes = []
		for n in self.inbound_nodes:
			n.outbound_nodes.append(self)
		self.value = None


class Input(Node):
	def __init__(self):
		Node.__init__(self)

	def forward(self, value = None):
		if value is not None:
			self.value = value

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
