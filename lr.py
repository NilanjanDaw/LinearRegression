# @Author: nilanjan
# @Date:   2018-08-05T19:02:51+05:30
# @Email:  nilanjandaw@gmail.com
# @Filename: lr.py
# @Last modified by:   nilanjan
# @Last modified time: 2018-08-19T20:03:52+05:30
# @copyright: Nilanjan Daw


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import math
import argparse

iterations = 500000
learning_rate = 0.0001
lambda_regularizer = 4500

error_rate = 0.00000001


def normalisation(data):
    return (data - data.min()) / (data.max() - data.min())


def plot(data):

    plt.plot(data["F1"], data["target"], 'mo', markersize=2, alpha=0.6, label="F1")

    plt.ylabel('target', fontsize=18)
    plt.xlabel('F1', fontsize=16)
    plt.show()

    plt.plot(data["F6"], data["target"], 'mo', markersize=2, alpha=0.6, label="F6")
    plt.ylabel('target', fontsize=18)
    plt.xlabel('F6', fontsize=16)
    plt.show()

    plt.plot(data["base_time"], data["target"], 'mo', markersize=2, alpha=0.6, label="base_time")
    plt.ylabel('target', fontsize=18)
    plt.xlabel('base_time', fontsize=16)
    plt.show()


def read_data(file_path, basis=None):
    if os.path.isfile(file_path):
        data = pd.read_csv(file_path, skiprows=[0], names=["page_likes", "page_checkin", "daily_crowd",
                                                           "page_category", "F1", "F2", "F3", "F4", "F5", "F6",
                                                           "F7", "F8", "c1", "c2", "c3", "c4", "c5", "base_time",
                                                           "post_length", "share_count", "promotion",
                                                           "h_target", "post_day", "basetime_day", "target"])

        # plot(data)

        y_train = data.loc[:, "target"]
        basetime_day = pd.get_dummies(data["basetime_day"], prefix='basetime_day')
        post_day = pd.get_dummies(data["post_day"], prefix="post_day")

        data = pd.concat([data, basetime_day], axis=1)
        data = pd.concat([data, post_day], axis=1)

        data.drop(['basetime_day'], axis=1, inplace=True)
        data.drop(["post_day"], axis=1, inplace=True)

        data = normalisation(data)
        data.drop(["target"], axis=1, inplace=True)
        y_train = y_train.values
        x_train = data.values
        if basis == 'sigmoid':
            x_train = 1 / (1 + np.exp(x_train))
        if basis == 'normal':
            variance = x_train.std() ** 2
            mean = x_train.mean()
            x_train = -((x_train - mean) ** 2) / (2 * variance)
            x_train = 1 / math.sqrt(2 * math.pi * variance) * np.exp(x_train)
        number_of_samples, _ = np.shape(x_train)
        bias = np.ones((number_of_samples, 1))
        x_train = np.hstack((bias, x_train))

        return x_train, y_train
    else:
        print("file not found")


def read_data_test(file_path, basis=None):
    if os.path.isfile(file_path):
        data = pd.read_csv(file_path, skiprows=[0], names=["page_likes", "page_checkin", "daily_crowd",
                                                           "page_category", "F1", "F2", "F3", "F4", "F5", "F6",
                                                           "F7", "F8", "c1", "c2", "c3", "c4", "c5", "base_time",
                                                           "post_length", "share_count", "promotion",
                                                           "h_target", "post_day", "basetime_day"])

        basetime_day = pd.get_dummies(data["basetime_day"], prefix='basetime_day')
        post_day = pd.get_dummies(data["post_day"], prefix="post_day")

        data = pd.concat([data, basetime_day], axis=1)
        data = pd.concat([data, post_day], axis=1)

        data.drop(['basetime_day'], axis=1, inplace=True)
        data.drop(["post_day"], axis=1, inplace=True)
        x_test = normalisation(data)

        x_test[np.isnan(x_test)] = 0.0
        x_test = x_test.values
        number_of_tests, _ = np.shape(x_test)
        if basis == "sigmoid":
            x_test = 1 / (1 + np.exp(x_test))
        elif basis == 'normal':
            variance = x_test.std() ** 2
            mean = x_test.mean()
            x_test = -((x_test - mean) ** 2) / (2 * variance)
            x_test = 1 / math.sqrt(2 * math.pi * variance) * np.exp(x_test)
        bias = np.ones((number_of_tests, 1))
        x_test = np.hstack((bias, x_test))

        return x_test
    else:
        print("file not found")


def loss_function(x, y, weight):

    number_of_samples, _ = np.shape(x)
    predict = np.dot(x, weight) - y
    cost = np.sum(predict ** 2) + lambda_regularizer * np.dot(weight.T, weight)
    cost = cost / (number_of_samples)
    return cost, predict


def gradient_descent(x_train, y_train, x_validate, y_validate,
                     iterations, learning_rate):

    number_of_samples, number_of_features = np.shape(x_train)
    print("Number of train samples: {}, number of train features: {}".format(
        number_of_samples, number_of_features))

    weight = np.ones(number_of_features)
    best_validation_loss = math.inf
    best_weight = None
    best_train_loss = math.inf
    for i in range(0, iterations):
        train_loss, predict = loss_function(x_train, y_train, weight)
        validation_loss, _ = loss_function(x_validate, y_validate, weight)
        if i % 100 == 0:
            print("iteration {}\t| Train_MSE {}\t| Validation_MSE {} ".format(
                  str(i), str(train_loss), str(validation_loss)))

        gradient = np.dot(predict, x_train) + lambda_regularizer * weight
        gradient = 2 * learning_rate * gradient / number_of_samples
        weight = weight - gradient
        if abs(best_train_loss - train_loss) < error_rate:
            break
        if validation_loss > best_validation_loss:
            break
        best_weight = weight
        best_train_loss = train_loss
        best_validation_loss = validation_loss

    print("final train loss: {} | iteraions: {}".format(train_loss, i))

    return best_weight


def p_loss_function(x, y, weight, p):

    number_of_samples, _ = np.shape(x)
    predict = np.dot(x, weight) - y

    np.power(weight, p / 2)
    norm = np.dot(weight.T, weight)
    norm = norm ** (1 / p)
    cost = np.sum(predict ** 2) + lambda_regularizer * norm
    cost = cost / (number_of_samples)
    return cost, predict


def p_norm_gradient_descent(x_train, y_train, x_validate, y_validate,
                            iterations, learning_rate, p):

    number_of_samples, number_of_features = np.shape(x_train)
    print("Number of train samples: {}, number of train features: {}".format(
        number_of_samples, number_of_features))

    weight = np.ones(number_of_features)
    best_validation_loss = math.inf
    best_weight = None
    best_train_loss = math.inf
    for i in range(0, iterations):
        train_loss, predict = p_loss_function(x_train, y_train, weight, p)
        validation_loss, _ = p_loss_function(x_validate, y_validate, weight, p)
        if i % 100 == 0:
            print("iteration {}\t| Train_MSE {}\t| Validation_MSE {} ".format(
                  str(i), str(train_loss), str(validation_loss)))
        regularizer = weight.copy()
        np.power(regularizer, p / 2)
        norm = np.dot(weight.T, weight)
        norm = norm ** ((1 / p) - 1)
        norm = norm * np.power(weight.copy(), p - 1)
        gradient = np.dot(predict, x_train) + lambda_regularizer * norm
        gradient = 2 * learning_rate * gradient / number_of_samples
        weight = weight - gradient
        if abs(best_train_loss - train_loss) < error_rate:
            break
        if validation_loss > best_validation_loss:
            break
        best_weight = weight
        best_train_loss = train_loss
        best_validation_loss = validation_loss

    print("final loss: {} | iteraions: {}".format(train_loss, i))

    return best_weight


def test(x_test, weight):
    number_of_samples, number_of_features = np.shape(x_test)
    print("Number of test samples: {}, number of test features: {}".format(
        number_of_samples, number_of_features))
    predict = np.zeros(number_of_samples)

    for i in range(0, number_of_samples):
        for j in range(0, number_of_features):
            predict[i] = predict[i] + x_test[i][j] * weight[j]

    return predict


def main():
    global learning_rate, lambda_regularizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--p-norm', dest='p', type=int)
    parser.add_argument('--basis', dest='basis', type=str,
                        choices=['sigmoid', 'normal'])
    parser.add_argument('--lambda', dest='lambda_rate', type=float)
    parser.add_argument('--alpha', dest='learning_rate', type=float)
    parser.add_argument('--train', dest='train', type=str)
    parser.add_argument('--test', dest='test', type=str)
    args = parser.parse_args()

    learning_rate = args.learning_rate
    lambda_regularizer = args.lambda_rate
    p = args.p

    x_train, y_train = read_data(args.train, basis=args.basis)
    x_validate = x_train[25600:, :]
    x_train = x_train[:25600, :]

    y_validate = y_train[25600:]
    y_train = y_train[:25600]

    number_of_samples, number_of_features = np.shape(x_train)
    if p is None:
        weight = gradient_descent(x_train, y_train, x_validate, y_validate,
                                  iterations, learning_rate)
    else:
        weight = p_norm_gradient_descent(x_train, y_train, x_validate, y_validate,
                                         iterations, learning_rate, p)

    print("weight:")
    print(weight)

    x_test = read_data_test(args.test, basis=args.basis)
    test_data = test(x_test, weight)

    i = 0
    with open('submission.csv', 'w') as file:
        fieldNames = ["Id", "target"]
        writer = csv.DictWriter(file, fieldNames)
        writer.writeheader()
        for row in test_data:
            writer.writerow({"Id": str(i), "target": str(row)})
            i = i + 1

    print("done")


if __name__ == '__main__':
    main()
