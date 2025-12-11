import numpy as np

class NeuralNetworkV0:
    
    def __init__(self, activation_dims):
        
        self.activation_dims = activation_dims
        self.layers = len(activation_dims)
        self.network = {}
        self.gradient = {}
        self.output = np.zeros(activation_dims[-1])
        self.create_network()
        
    def create_network(self):
        
        for layer in range(self.layers):
            self.network[layer+1] = {}
            
            if (layer+1 == 1):
  
                self.network[layer+1]["activations"] = self.sigmoid(np.empty(self.activation_dims[layer]).reshape(-1,1)) 
                
            else:
                self.network[layer+1]["activations"] = np.random.rand(self.activation_dims[layer]).reshape(-1,1)
                self.network[layer+1]["weights"] = np.random.uniform(-0.5, 0.5, size = self.activation_dims[layer]*self.activation_dims[layer-1]).reshape(self.activation_dims[layer], self.activation_dims[layer-1])
                self.network[layer+1]["biases"] = np.random.uniform(-0.5, 0.5, size = self.activation_dims[layer]).reshape(-1,1)
            
    
    def create_gradient(self):
        
        g = {}
        for layer in range(2, self.layers+1):
            g[layer] = {}
            
            g[layer]["weights"] = np.zeros(self.network[layer]["weights"].shape)
            g[layer]["biases"] = np.zeros(self.network[layer]["biases"].shape)

        return g
       
    def calculate_z(self, a, w, b):

        """caculate z from z = a*w + b"""

        return np.dot(w, a) + b

    def sigmoid(self, x):
    
        """return value of sigmoid function of x"""

        return 1/(1+np.exp(-x))

    def cost_function_derivative(self, a, y):
        
        """return value of the cost function derivative of a and y"""

        return 2*(a-y)

    def sigmoid_derivative(self, x):
        
        """reuturn value of sigmoid derivative of x"""

        return (1/(1+np.exp(-x))**2)*np.exp(-x)

    def forward(self, data):
        
        """calculate activations in each layer"""
        
        for layer in range(1, self.layers+1):
            
            if (layer == 1):
  
                # self.network[layer]["activations"] = self.sigmoid(data[index].reshape(-1,1))
                self.network[layer]["activations"] = (data.reshape(-1,1))/255
                
            else:
    
                self.network[layer]["activations"] = self.sigmoid(self.calculate_z(self.network[layer-1]["activations"], self.network[layer]["weights"], self.network[layer]["biases"]))

        return self.network[self.layers]["activations"]
        
    def back(self, l, g, e = None):

        """compute backpropagation and return gradient of the network"""
        
        n = self.network
        
        if (l == 1):
            return g
        else:
            z = self.calculate_z(n[l-1]["activations"], n[l]["weights"], n[l]["biases"])
            da_by_dz = self.sigmoid_derivative(z)

            if (l == len(n)):
                dc_by_da = self.cost_function_derivative(n[l]["activations"], e)
            else:
                dc_by_da = np.sum(g[l+1]["biases"]*n[l+1]["weights"], axis = 0).reshape(-1,1)
                # dc_by_da = np.dot(n[l+1]["weights"], g[l+1["biases"]])

            g[l]["weights"] = n[l-1]["activations"].reshape(-1)*dc_by_da*da_by_dz
            g[l]["biases"] = dc_by_da*da_by_dz
            g = self.back(l-1,g)
            return g

    
    def add_gradient(self, g1, g2):

        """return added gradient"""

        w = "weights"
        b = "biases"
        l = len(g1)
        
        for i in range(2, l+2, 1):
            g1[i][w] += g2[i][w]
            g1[i][b] += g2[i][b]

        return g1

    def average_gradient(self, g, batch_size):

        """averaging the gradient with batch size and return averaged gradient"""

        w = "weights"
        b = "biases"
        for i in range(2, self.layers + 1, 1):
            g[i][w]/=batch_size
            g[i][b]/=batch_size

        return g

    def graident_descent(self, g, lr):

        """compute graident descent"""
        
        w = "weights"
        b = "biases"
        for i in range(2, self.layers + 1, 1):
            self.network[i][w] -= lr*g[i][w]
            self.network[i][b] -= lr*g[i][b]
            
    
    def calculate_loss(self, pred, true):
        true_output = np.zeros(len(pred))
        true_output[true] = 1

        mse = ((true_output - pred.squeeze())**2).sum() / len(pred)

        return mse