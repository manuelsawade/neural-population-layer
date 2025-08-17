from data.mnist import MNIST
from networks import NeuralNetwork
from trainer import Trainer


class MNISTTraining:
    def __init__(self, epochs, batch_size, learning_rate, train_noise, test_noise, **kwargs):
        self.data_set = MNIST()
        self.input_dim = 28 * 28
        self.output_dim = 10
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_noise = train_noise
        self.test_noise = test_noise
        self.kwargs = kwargs
    
    @classmethod
    def train_stack(self, stack):
        trainer = Trainer(
            model=NeuralNetwork(layers=stack),
            data_set=self.data_set,
            training_noise=self.train_noise,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate)

        output = {}
        output['train_noise'] = self.train_noise
        for k, val in self.kwargs.items():
            output[k] = val

        trainer.train(epochs=10)
        trainer.test(noise=0.2, summary=output)