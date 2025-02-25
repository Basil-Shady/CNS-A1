import numpy as np
import math

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
def ann(input_layer, weights, biases):
    hidden_layer = []
    for i in range(len(weights[0])):
        weighted_sum = 0
        for j in range(len(input_layer)):
            weighted_sum += input_layer[j] * weights[0][i][j]
        weighted_sum += biases[0][i]
        activation = tanh(weighted_sum)
        hidden_layer.append(activation)

    output_layer = []
    for i in range(len(weights[1])):
        weighted_sum = 0
        for j in range(len(hidden_layer)):
            weighted_sum += hidden_layer[j] * weights[1][i][j]
        weighted_sum += biases[1][i]
        activation = tanh(weighted_sum)
        output_layer.append(activation)

    return output_layer

if __name__ == '__main__':  
    input_layer = [0.05, 0.1]

    weights = [
        [[0.15, 0.2], [0.25, 0.3]],
        [[0.4, 0.45], [0.5, 0.55]],
    ]
    biases = [[0.35, 0.35], [0.6, 0.65]]
    output_layer = ann(input_layer, weights, biases)
    print("output = ",output_layer)
