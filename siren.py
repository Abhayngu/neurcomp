from __future__ import print_function
import argparse
import torch as th
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()
    #

    def init_weights(self):
        with th.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
            #
        #
    #

    def forward(self, input):
        return th.sin(self.omega_0 * self.linear(input))
    #
#

class ResidualSineLayer(nn.Module):
    def __init__(self, features, bias=True, ave_first=False, ave_second=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0

        self.features = features
        self.linear_1 = nn.Linear(features, features, bias=bias)
        self.linear_2 = nn.Linear(features, features, bias=bias)

        self.weight_1 = .5 if ave_first else 1
        self.weight_2 = .5 if ave_second else 1

        self.init_weights()
    #

    def init_weights(self):
        with th.no_grad():
            self.linear_1.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)
            self.linear_2.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)
        #
    #

    def forward(self, input):
        sine_1 = th.sin(self.omega_0 * self.linear_1(self.weight_1*input))
        sine_2 = th.sin(self.omega_0 * self.linear_2(sine_1))
        return self.weight_2*(input+sine_2)



# The function is used to calculate the number of neurons in each hidden layer

# Parameters:
# opt contains options value
# target_size is the number of points when some compression_ratio is applied
def compute_num_neurons(opt,target_size):
    # Input and output layers neurons
    d_in = opt.d_in
    d_out = opt.d_out

    # For sample let's say n_layers = 3, neurons = 5
    def network_size(neurons):
        # Adding input layer
        layers = [d_in]
        # layers = [3]

        # Adding hidden layers
        layers.extend([neurons]*opt.n_layers)
        # layers = [3, 5, 5, 5]

        # Adding output layer
        layers.append(d_out)
        # layers = [3, 5, 5, 5, 1]

        # Number of steps 
        n_layers = len(layers)-1
        # n_layers = 5 - 1 = 4

        n_params = 0

        # np.arange(4) = [0, 1, 2, 3]
        for ndx in np.arange(n_layers):

            # number of neurons in below layer
            layer_in = layers[ndx]

            # number of neurons in above layer
            layer_out = layers[ndx+1]

            # max number of neurons in both the layer
            og_layer_in = max(layer_in,layer_out)

            # if below layer is the input layer 
            # or the above layer is the output layer
            if ndx==0 or ndx==(n_layers-1):
                # Adding weight corresponding to every neuron for every input neuron
                # Adding bias for every neuron in the above layer
                n_params += ((layer_in+1)*layer_out)
            
            else:

                # If the layer is residual then proceed as follows as there will be more weights if residual layer is included
                if opt.is_residual:

                    # for this example, is_shortcut is always going to be false as the below and above layer will be having same number of neurons
                    is_shortcut = layer_in != layer_out

                    if is_shortcut:
                        n_params += (layer_in*layer_out)+layer_out
                    
                    
                    n_params += (layer_in*og_layer_in)+og_layer_in
                    n_params += (og_layer_in*layer_out)+layer_out

                # if the layer is non residual then simply add number of weights and biases as follows
                else:
                    n_params += ((layer_in+1)*layer_out)
                #
            #
        #

        return n_params
    
    # by default initialising number of neurons in a hidden layer by 16
    min_neurons = 16

    # while the network_size by using min_neurons
    # is less than the target size keep increasing the min_neurons
    while network_size(min_neurons) < target_size:
        min_neurons+=1

    # Decrease min_neurons by 1 as it must have not worked for the last min_neuron value
    min_neurons-=1

    return min_neurons
#

class FieldNet(nn.Module):
    def __init__(self, opt):
        super(FieldNet, self).__init__()

        self.d_in = opt.d_in
        self.layers = [self.d_in]
        self.layers.extend(opt.layers)
        self.d_out = opt.d_out
        self.layers.append(self.d_out)
        self.n_layers = len(self.layers)-1
        self.w0 = opt.w0
        self.is_residual = opt.is_residual

        self.net_layers = nn.ModuleList()
        for ndx in np.arange(self.n_layers):
            layer_in = self.layers[ndx]
            layer_out = self.layers[ndx+1]
            if ndx != self.n_layers-1:
                if not self.is_residual:
                    self.net_layers.append(SineLayer(layer_in,layer_out,bias=True,is_first=ndx==0))
                    continue
                #

                if ndx==0:
                    self.net_layers.append(SineLayer(layer_in,layer_out,bias=True,is_first=ndx==0))
                else:
                    self.net_layers.append(ResidualSineLayer(layer_in,bias=True,ave_first=ndx>1,ave_second=ndx==(self.n_layers-2)))
                #
            else:
                final_linear = nn.Linear(layer_in,layer_out)
                with th.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6 / (layer_in)) / 30.0, np.sqrt(6 / (layer_in)) / 30.0)
                self.net_layers.append(final_linear)
            #
        #
    #

    def forward(self,input):
        batch_size = input.shape[0]
        out = input
        for ndx,net_layer in enumerate(self.net_layers):
            out = net_layer(out)
        #
        return out
    #
#
