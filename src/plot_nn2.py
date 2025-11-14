# Source - https://stackoverflow.com/a
# Posted by Oliver Wilken, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-14, License - CC BY-SA 4.0

from matplotlib import pyplot
from math import cos, sin, atan

import library


class Neuron():
    def __init__(self, x, y, placeholder=False):
        self.x = x
        self.y = y
        self.placeholder = placeholder

    def draw(self, neuron_radius):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False, edgecolor="none" if self.placeholder else 'black')
        pyplot.gca().add_patch(circle)
        if self.placeholder:
            pyplot.gca().annotate("[...]", xy=(self.x, self.y), fontsize=20, ha="center")


class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        has_placeholder: bool = number_of_neurons % 2 == 1
        for iteration in range(number_of_neurons):
            placeholder = True if has_placeholder and iteration == (number_of_neurons - 1) / 2 else False
            
            neuron = Neuron(x, self.y, placeholder=placeholder)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        print(len(neurons))
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2):
        placeholder = True if neuron1.placeholder or neuron2.placeholder else False
        color = "none" if placeholder else 'gray'

        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment), color=color)
        pyplot.gca().add_line(line)

    def draw(self, layerType=0):
        for i, neuron in enumerate(self.neurons):
            if len(self.neurons) / 2 == i:
                pyplot.gca().add_line
                print("half")

            neuron.draw( self.neuron_radius )

            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron)
        # write Text
        x_text = -5.0#self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == -1:
            pyplot.text(x_text, self.y, 'Input Layer', fontsize = 20)
        elif layerType == 0:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize = 20)
        else:
            pyplot.text(x_text, self.y, 'Hidden Layer '+str(layerType), fontsize = 20)

class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer + 1
        self.layers = []
        self.layertype = 0

    def add_layer(self, number_of_neurons ):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer)
        self.layers.append(layer)

    def draw(self):
        pyplot.figure(figsize=(14, 8))
        for i in range( len(self.layers) ):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw( i )
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title( 'Neural Network architecture', fontsize=20)
        pyplot.savefig(library.get_target_image(__file__), dpi=300)

class DrawNN():
    def __init__( self, neural_network: list[int] ):
        neural_network.reverse()
        self.neural_network = neural_network

    def draw( self ):
        widest_layer = max( self.neural_network )
        network = NeuralNetwork( widest_layer)
        for l in self.neural_network:
            network.add_layer(l)
        network.draw()

network = DrawNN( [11,11,10] )
network.draw()