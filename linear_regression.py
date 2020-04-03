import neural_network
import data_retriever

training_data = data_retriever.get_data()

net = neural_network.Network([1, 1])

data_size = len(training_data)

net.SGD(training_data, 500, data_size, 0.02)

slope = net.weights[0][0][0]
intercept = net.biases[0][0][0]

print("Equation: y = {0}x {1} {2}".format(slope, "+" if intercept > 0 else "-", intercept))

