from sklearn_extensions.extreme_learning_machines.random_layer import RandomLayer, MLPRandomLayer
from sklearn_extensions.extreme_learning_machines import elm
import numpy as np
from scipy.linalg import pinv2
from sklearn.utils import as_float_array
from sklearn.utils.extmath import safe_sparse_dot
#creating Genetic EML class
class GeneticELMRegressor(elm.BaseELM, elm.RegressorMixin):   
    
    def __init__(self, hidden_layer=MLPRandomLayer(random_state=0), regressor=None):        
        super(GeneticELMRegressor, self).__init__(hidden_layer, regressor)
        self.coefs_ = None
        self.fitted_ = False
        self.hidden_activations_ = None
    #genetic algorithm fitness function using RMSE
    def fitness(self, y):
        if self.regressor is None:
            self.coefs_ = safe_sparse_dot(pinv2(self.hidden_activations_), y)
        else:
            self.regressor.fit(self.hidden_activations_, y)

        self.fitted_ = True

    def fit(self, X, y):
        # fit random hidden layer and compute the hidden layer activations
        self.hidden_activations_ = self.hidden_layer.fit_transform(X)

        # solve the regression from hidden activations to outputs
        self.fitness(as_float_array(y, copy=True))

        return self
    #crossover to generate new population and then evaluate fitness
    def crossover(self):
        if self.regressor is None:
            preds = safe_sparse_dot(self.hidden_activations_, self.coefs_)
        else:
            preds = self.regressor.predict(self.hidden_activations_)

        return preds

    def predict(self, X):
        if not self.fitted_:
            raise ValueError("ELMRegressor not fitted")

        # compute hidden layer activations
        self.hidden_activations_ = self.hidden_layer.transform(X)

        # compute output predictions for new hidden activations
        predictions = self.crossover()

        return predictions
