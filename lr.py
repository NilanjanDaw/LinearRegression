# @Author: nilanjan
# @Date:   2018-08-05T19:02:51+05:30
# @Email:  nilanjandaw@gmail.com
# @Filename: lr.py
# @Last modified by:   nilanjan
# @Last modified time: 2018-08-18T20:06:29+05:30
# @copyright: Nilanjan Daw


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import math

iterations = 500000
learning_rate = 0.0003
lambda_regularizer = 80000

error_rate = 0.00000001


def normalisation(data):
    return (data - data.min()) / (data.max() - data.min())


def plot(data):
    fig = plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(data["post_day"], data["page_likes"], 'bo',
             markersize=2, alpha=0.6, label="post_day")
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(data["daily_crowd"], data["page_likes"], 'ro',
             markersize=2, alpha=0.6, label="daily_crowd")
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(data["share_count"], data["page_likes"], 'go',
             markersize=2, alpha=0.6, label="share_count")
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(data["page_checkin"], data["page_likes"], 'yo',
             markersize=2, alpha=0.6, label="page_checkin")
    plt.legend()
    # plt.plot(data["page_likes"], data["promotion"], 'mo', markersize=2, alpha=0.6, label="page_likes")

    # plt.show()

    fig = plt.figure()
    plt.subplot(2, 4, 1)
    plt.plot(data["F1"], data["page_likes"], 'bo',
             markersize=2, alpha=0.6, label="F1")
    plt.legend()
    plt.subplot(2, 4, 2)
    plt.plot(data["F2"], data["page_likes"], 'ro',
             markersize=2, alpha=0.6, label="F2")
    plt.legend()
    plt.subplot(2, 4, 3)
    plt.plot(data["F3"], data["page_likes"], 'go',
             markersize=2, alpha=0.6, label="F3")
    plt.legend()
    plt.subplot(2, 4, 4)
    plt.plot(data["F4"], data["page_likes"], 'yo',
             markersize=2, alpha=0.6, label="F4")
    plt.legend()

    plt.subplot(2, 4, 5)
    plt.plot(data["F5"], data["page_likes"], 'bo',
             markersize=2, alpha=0.6, label="F5")
    plt.legend()
    plt.subplot(2, 4, 6)
    plt.plot(data["F6"], data["page_likes"], 'ro',
             markersize=2, alpha=0.6, label="F6")
    plt.legend()
    plt.subplot(2, 4, 7)
    plt.plot(data["F7"], data["page_likes"], 'go',
             markersize=2, alpha=0.6, label="F7")
    plt.legend()
    plt.subplot(2, 4, 8)
    plt.plot(data["F8"], data["page_likes"], 'yo',
             markersize=2, alpha=0.6, label="F8")
    plt.legend()
    # plt.plot(data["page_likes"], data["promotion"], 'mo', markersize=2, alpha=0.6, label="page_likes")
    fig = plt.figure()
    plt.subplot(2, 4, 1)
    plt.plot(data["c1"], data["page_likes"], 'bo',
             markersize=2, alpha=0.6, label="c1")
    plt.legend()
    plt.subplot(2, 4, 2)
    plt.plot(data["c2"], data["page_likes"], 'ro',
             markersize=2, alpha=0.6, label="c2")
    plt.legend()
    plt.subplot(2, 4, 3)
    plt.plot(data["c3"], data["page_likes"], 'go',
             markersize=2, alpha=0.6, label="c3")
    plt.legend()
    plt.subplot(2, 4, 4)
    plt.plot(data["c4"], data["page_likes"], 'yo',
             markersize=2, alpha=0.6, label="c4")
    plt.legend()

    plt.subplot(2, 4, 5)
    plt.plot(data["c5"], data["page_likes"], 'bo',
             markersize=2, alpha=0.6, label="c5")
    plt.legend()

    plt.show()


# def one_hot_encode():
#     day = {"monday": 0, "tuesday": 1, "wednesday": 2,
#            "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
#


def read_data(file_path):
    if os.path.isfile(file_path):
        data = pd.read_csv(file_path, skiprows=[0], names=["page_likes", "page_checkin", "daily_crowd",
                                                           "page_category", "F1", "F2", "F3", "F4", "F5", "F6",
                                                           "F7", "F8", "c1", "c2", "c3", "c4", "c5", "base_time",
                                                           "post_length", "share_count", "promotion",
                                                           "h_target", "post_day", "basetime_day", "target"])
        # data["basetime_day"] = list(map(day_to_num, data["basetime_day"]))
        # data["post_day"] = list(map(day_to_num, data["post_day"]))
        data.sample(frac=1).reset_index(drop=True)
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
        number_of_samples, _ = np.shape(x_train)
        bias = np.ones((number_of_samples, 1))
        x_train = np.hstack((bias, x_train))

        return x_train, y_train
    else:
        print("file not found")


def read_data_test(file_path):
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

        x_test[np.isnan(x_test)] = 0
        number_of_tests, _ = np.shape(x_test)
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


def sigmoid_loss_function(x, y, weight):

    number_of_samples, _ = np.shape(x)
    predict = np.dot(x, weight) - y
    cost = np.sum(predict ** 2) + lambda_regularizer * np.dot(weight.T, weight)
    cost = cost / (number_of_samples)
    return cost, predict


def basis_sigmoid_gradient_descent(x_train, y_train, x_validate, y_validate,
                                   iterations, learning_rate, p):

    number_of_samples, number_of_features = np.shape(x_train)
    print("Number of train samples: {}, number of train features: {}".format(
       number_of_samples, number_of_features))

    weight = np.ones(number_of_features)
    best_validation_loss = math.inf
    best_weight = None
    best_train_loss = math.inf
    for i in range(0, iterations):
        train_loss, predict = sigmoid_loss_function(x_train, y_train, weight)
        validation_loss, _ = sigmoid_loss_function(x_validate, y_validate, weight)
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

    x_train, y_train = read_data("train.csv")

    x_validate = x_train[25600:, :]
    x_train = x_train[:25600, :]

    y_validate = y_train[25600:]
    y_train = y_train[:25600]

    number_of_samples, number_of_features = np.shape(x_train)

    # weight = gradient_descent(x_train, y_train, x_validate, y_validate,
                              # iterations, learning_rate)
    p = 6
    weight = p_norm_gradient_descent(x_train, y_train, x_validate, y_validate,
                                     iterations, learning_rate, p)

    x_test = read_data_test('test.csv')
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
