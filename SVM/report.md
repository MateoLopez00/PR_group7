# Suport Vector Machine 
Optimization Problem of SVM 

$$
\min \;\; \frac{1}{2}\lVert w\rVert^2 + C \sum_{i=1}^n \xi_i
$$

Hyperparameters to be optimised: C to configure the weight of the minimisation of error distance, kernel function and gamma 

## Optimization for Hyperparameters

Source: 
- https://medium.com/@prayushshrestha89/tuning-svm-hyperparameters-making-your-classifier-shine-like-a-pro-8673639ddb16
- https://medium.com/@sibelasgarova/gridsearchcv-and-randomsearchcv-in-svm-cda2dad8e434
- https://scikit-learn.org/stable/modules/svm.html
- The plots were created with ChatGPT

### Kernel 
1. Linear: $$ \langle x, x' \rangle $$
2. RBF radial basis function kernel: $$ \exp\!\big(-\gamma \, \lVert x_i - x_j \rVert^2\big) $$

#### C 
C is the weight for misclassification of training examples. A high C, means that misclassifications are highly weighted. If C is to high, this leads to overfitting the model. 

#### Gamma RBF only

Decides how close model fits training data, so how much influence a single training example has. Large gamma means, that examples have to be close to affect each other (problem of overfitting if gamma is to high, model might learn the noise in data).

#### RandomizedSearchCV 
To optimise the hyperparameters kernel, gamma and C RandomizedSearchCV was used.
RandomizedSearchCV optimizes the parameters using cross-validation with n_iter random combinations from the given ranges for C, gamma and the kernel types.

I chose RandomizedSearchCV instead of GridSearchCV because GridSearchCV evaluates every possible combination of hyperparameters, which is very time-consuming. RandomizedSearchCV tests only a fixed number of random combinations, making it faster while still giving good results.

I also had to use a subset of the training data to optimize the parameters, because with the RBF kernel the runtime on the full dataset was not feasible.


## Result 

I ran the RandomizedSearchCV algorithm twice on different subsets of 15,000 samples, using the following parameter ranges:

```
C_range = [0.1, 1, 10, 100, 1000] 
gamma_range = [1.0, 0.1, 0.01, 0.001, 0.0001] 
kernels = ['linear', 'rbf'] 
param_dist = [ 
    {'kernel': ['linear'], 'C': C_range}, 
    {'kernel': ['rbf'], 'C': C_range, 'gamma': gamma_range} 
    ]
```
For the second run, I removed gamma = 1.0 due to poor performance in the initial results.
The best performance was achieved with C = 1000, gamma = 0.01, and kernel = 'rbf'.
Since the configuration C = 100, gamma = 0.01, and kernel = 'rbf' achieved identical accuracy, I selected C = 100 for the final model to reduce the risk of overfitting. 

