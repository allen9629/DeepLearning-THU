""" Optimizer Class """

import numpy as np

class SGD():
    def __init__(self, learningRate, weightDecay):
        self.learningRate = learningRate
        self.weightDecay = weightDecay
    
    def step(self, model):
        layers = model.layerList
        for layer in layers:
            if layer.trainable:
                layer.diff_W = -self.learningRate * layer.grad_W - self.weightDecay* self.learningRate* layer.W
                layer.diff_b = -self.learningRate * layer.grad_b
                
                layer.W += layer.diff_W
                layer.b += layer.diff_b