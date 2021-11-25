import numpy as np

class Softmax:
  # A standard fully-connected layer with softmax activation.

  def __init__(self, input_len, nodes):
    # We divide by input_len to reduce the variance of our initial values
    self.weights = np.random.normal(0, 1/input_len,(input_len, nodes))
    self.biases = np.zeros(nodes)
    print(self.weights.shape)

  def forward(self, input):
    '''
    Performs a forward pass of the softmax layer using the given input.
    Returns a 1d numpy array containing the respective probability values.
    - input can be any array with any dimensions.
    '''
    self.last_input_shape = input.shape
    input = input.flatten()

    self.last_input = input
    input_len, nodes = self.weights.shape

    totals = np.dot(input, self.weights) + self.biases
    self.last_totals = totals
    exp = np.exp(totals)
    return exp / np.sum(exp, axis=0)

  def backprop(self, gradients, lr):
      for i, gradient in enumerate(gradients):
            if gradient == 0:
                continue
      
            t_exp = np.exp(self.last_totals)

            S = np.sum(t_exp)
      
            # On cherche le gradient de i spécifiquement, donc 
            # f(x) = (e^x)*S^-1
            # df(x)/dx. [df(x1)/dx1, df(x2)/x2, etc]
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
      

            #TODO sur une feuille de papier
            # Gradients of totals against weights/biases/input
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights
            # Gradients of loss against totals
            d_L_d_t = d_out_d_t * gradient
            # Gradients of loss against weights/biases/input
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t
      
            # Update weights / biases
            self.weights -= lr * d_L_d_w
            self.biases -= lr * d_L_d_b
            return d_L_d_inputs.reshape(self.last_input_shape)




