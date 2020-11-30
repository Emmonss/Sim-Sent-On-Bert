from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np


def load(model_path, custom_object=None):
    new_model = NLUModel()
    new_model.model = load_model(model_path,custom_objects=custom_object)

class NLUModel:
    def __init__(self):
        self.model = None

    def predict(self,x):
        assert self.model == None, "model is None"
        return self.model.predict(x)

    def save(self,save_path):
        assert self.model == None, "model is None"
        self.model.save(save_path)

