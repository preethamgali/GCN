import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.linalg import fractional_matrix_power
import os
import tensorflow as tf

if 'cora' not in os.listdir():
  !wget https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz
  !tar -xvzf /content/cora.tgz

edges_cites = {
    "source" : [],
    "dest" : []
}

node_content = {
    "node" : [],
    "word_vector" : [],
    "label" : []
}

with open("/content/cora/cora.cites") as f:
  for edge in f.read().split('\n'):
    if len(edge) != 0:
      source, dest = edge.split()
      edges_cites["source"].append(int(source))
      edges_cites["dest"].append(int(dest))

with open("/content/cora/cora.content") as f:
  for node_info in f.read().split('\n'):
    if len(node_info) != 0:
      n_i = node_info.split()
      node, word_vector, label = n_i[0], n_i[1:-1], n_i[-1]
      node_content["node"].append(int(node))
      node_content['word_vector'].append([int(i) for i in word_vector])
      node_content["label"].append(label)

node_dict = {}
for i,n in enumerate(node_content["node"]):
  node_dict[n] = i

def get_adjacency_matrix(A):
  for s,d in zip(edges_cites["source"], edges_cites["dest"]):
      r,c = node_dict[s], node_dict[d]
      A[r][c] = 1
  return A 

n_nodes = len(node_content["node"])

A = np.zeros(shape= (n_nodes, n_nodes))
A = get_adjacency_matrix(A)

graph_shape = A.shape
I = np.eye(graph_shape[0], graph_shape[1])

A_hat = A + I 
D = np.sum(A_hat, axis= 0)
D_hat = np.diag(D)
D_norm = fractional_matrix_power(D_hat, -0.5)

H = np.asarray(node_content['word_vector'])

# A_norm = D_norm * A * D_norm
A_norm = D_norm @ A_hat @ D_norm

input_data = A_norm @ H

#  RELU(D**-0.5 * A * D**-0.5 * H * W)
#  D**−1A now corresponds to taking the average of neighboring node features.
# In practice, dynamics get more interesting when we use a symmetric normalization, i.e. 
# D**−1/2AD**−1/2 (as this no longer amounts to mere averaging of neighboring nodes)

categories = np.unique(node_content["label"])
categories_dict = {}
for i,c in enumerate(categories):
    categories_dict[c] = i

label_data = []
for l in node_content["label"]:
    label = [0]*len(categories_dict)
    label[categories_dict[l]] = 1
    label_data.append(label)

input_shape = input_data[0].shape
batch_size = 100
output_size = len(categories_dict)

GCN_input = tf.placeholder(dtype= tf.float64, shape= (None, *(input_shape)))
labels= tf.placeholder(dtype= tf.float64, shape= (None, output_size))

dense1 = tf.layers.dense(GCN_input, 512)
dense1 = tf.maximum(dense1, 0.2* dense1)

dense2 = tf.layers.dense(dense1, 256)
dense2 = tf.maximum(dense2, 0.2* dense2)

logits = tf.layers.dense(dense2, 7)
output = tf.nn.sigmoid(logits)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= logits, labels= labels))

train = tf.train.AdamOptimizer().minimize(loss)

epoches = 100
batch_size = 100
train_input_data, test_input_data = input_data[:1000], input_data[1000:]
train_label_data, test_label_data = label_data[:1000], label_data[1000:]
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for e in range(epoches):
  for i in range(int(len(train_input_data)/batch_size)):
    batch_input = train_input_data[i:i+batch_size]
    batch_label = train_label_data[i:i+batch_size]
    # loss,_ = sess.run([loss, train], feed_dict= {GCN_input:batch_input, labels:batch_label})
    error, _ = sess.run([loss, train], feed_dict= {GCN_input:batch_input, labels:batch_label})
    if e%10 == 0:
      print(error)

i=2
out = sess.run(output, feed_dict= {GCN_input:[test_input_data[i]]})
print(np.argmax(test_label_data[i]) == np.argmax(out))
