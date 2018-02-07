import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from csv import reader
from sklearn.preprocessing import StandardScaler
from random import randrange
import operator


def linear_grad_func(theta, x, y):
    # compute gradient
    grad = np.dot((linear_val_func(theta, x) - y).T, np.c_[np.ones(x.shape[0]), x])
    grad = grad / x.shape[0]

    return grad


def linear_val_func(theta, x):
    # forwarding
    return np.dot(np.c_[np.ones(x.shape[0]), x], theta.T)


def linear_cost_func(theta, x, y):
    # compute cost (loss)
    y_hat = linear_val_func(theta, x)
    cost = np.mean((y_hat - y) ** 2)
    return cost


def linear_grad_desc(theta, X_train, Y_train, lr, max_iter, tolerance):
    cost = linear_cost_func(theta, X_train, Y_train)
    RMSE_iter = []
    RMSE_iter.append(np.sqrt(np.sum((linear_val_func(theta, X_train) - Y_train) ** 2) / Y_train.shape[0]))
    cost_change = 1
    i = 1

    while cost_change > tolerance and i < max_iter:
        pre_cost = cost
        # compute gradient
        grad = linear_grad_func(theta, X_train, Y_train)

        # update gradient
        theta = theta - lr * grad

        # compute loss
        cost = linear_cost_func(theta, X_train, Y_train)
        RMSE_iter.append(np.sqrt(np.sum((linear_val_func(theta, X_train) - Y_train) ** 2) / Y_train.shape[0]))
        cost_change = abs(cost - pre_cost)
        i += 1

    return theta, RMSE_iter


def load_dataset(filename):
    '''Loads an example of market basket transactions from a provided csv file.

    Returns: A list (database) of lists (transactions). Each element of a transaction is
    an item.
    '''
    with open(filename, 'r') as dest_f:
        data_iter = reader(dest_f, delimiter=',', quotechar='"')
        data = [data for data in data_iter]
        data_array = np.asarray(data)

    return data_array


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def linear_regression(dataset, n_folds, lr, max_iter, tolerance):
    # split dataset into training and testing
    dataset_split = cross_validation_split(dataset, n_folds)
    RMSE_train = []
    RMSE_test = []
    SSE_train = []
    SSE_test = []

    for i in range(n_folds):
        test = np.array(dataset_split[i])
        train = list(dataset_split)
        train.pop(i)
        train = np.array(reduce(operator.add, train))

        # Normalize X_Train
        X_train = train[:, :-1]
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)

        # Get the mean and std to normalize the test dataset
        X_test = test[:, :-1]
        X_test = scaler.transform(X_test)

        Y_train = train[:, -1]
        Y_test = test[:, -1]

        Y_train = Y_train[:, None]
        Y_test = Y_test[:, None]

        # Linear regression
        #  Initialize the weights for the gradient descent algorithm to all zeros
        # theta = np.zeros((1, X_train.shape[1] + 1))
        theta = np.random.rand(1, X_train.shape[1] + 1)

        fitted_theta, RMSE_iter = linear_grad_desc(theta, X_train, Y_train, lr, max_iter, tolerance)

        #
        if i == 0:
            plt.plot(range(len(RMSE_iter)), RMSE_iter)
            plt.xlabel('Iteration')
            plt.ylabel('RMSE')

        RMSE_test.append(np.sqrt(np.sum((linear_val_func(fitted_theta, X_test) - Y_test) ** 2) / Y_test.shape[0]))
        RMSE_train.append(np.sqrt(np.sum((linear_val_func(fitted_theta, X_train) - Y_train) ** 2) / Y_train.shape[0]))
        SSE_test.append(np.sum((linear_val_func(fitted_theta, X_test) - Y_test) ** 2))
        SSE_train.append(np.sum((linear_val_func(fitted_theta, X_train) - Y_train) ** 2))
        print('Train RMSE: {}'.format(RMSE_train[i]))
        print('Test RMSE: {}'.format(RMSE_test[i]))
    print('Overall Mean Train RMSE: {}'.format(np.sum(RMSE_train) * 1. / len(RMSE_train)))
    print('Overall Mean Test RMSE: {}'.format(np.sum(RMSE_test) * 1. / len(RMSE_test)))
    print('Overall Mean Train SSE: {}'.format(np.sum(SSE_train) * 1. / len(SSE_train)))
    print('Overall Mean Test SSE: {}'.format(np.sum(SSE_test) * 1. / len(SSE_test)))
    print('std of train SSE: {}'.format(np.std(np.array(SSE_train), axis=0)))
    print('std of test SSE: {}'.format(np.std(np.array(SSE_test), axis=0)))


def sklearn_linear_regression(dataset, n_folds):
    # split dataset into training and testing

    X_train = X[:-20, :]
    X_test = X[-20:, :]

    Y_train = Y[:-20, None]
    Y_test = Y[-20:, None]

    # Linear regression
    regressor = linear_model.LinearRegression()
    regressor.fit(X_train, Y_train)
    print('Coefficients: {}'.format(regressor.coef_))
    print('Intercept: {}'.format(regressor.intercept_))
    print('MSE:{}'.format(np.mean((regressor.predict(X_test) - Y_test) ** 2)))


def main():
    dataset = load_dataset("housing.csv")
    dataset = dataset.astype(float)

    print('Housing dataset Linear Regression')
    linear_regression(dataset, n_folds=10, lr=0.0004, max_iter=1000, tolerance=0.005)
    print ('')

    dataset = load_dataset("yachtData.csv")
    dataset = dataset.astype(float)
    print('Yacht dataset Linear Regression')
    linear_regression(dataset, n_folds=10, lr=0.001, max_iter=1000, tolerance=0.001)
    print ('')

    dataset = load_dataset("concreteData.csv")
    dataset = dataset.astype(float)
    print('Concrete dataset Linear Regression')
    linear_regression(dataset, n_folds=10, lr=0.0007, max_iter=1000, tolerance=0.0001)
    print ('')


#    print('sklearn Linear Regression Example')
#    sklearn_linear_regression(dataset, n_folds=10)

if __name__ == "__main__":
    main()