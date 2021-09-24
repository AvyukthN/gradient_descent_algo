import matplotlib.pyplot as plt
import numpy as np

class GradientDescent:
    def __init__(self, optimization_type='linear_regression', _data=[]):
        self.data = _data

    def partial_deriv_cost_function(self, parameters, count):
        predicted = [(parameters[0] * self.data['x'][i]) + parameters[1] for i in range(len(self.data['x']))]
        costs0 = [predicted[i] - self.data['y'][i] for i in range(len(self.data['y']))]
        costs1 = [(predicted[i] - self.data['y'][i]) * self.data['x'][i] for i in range(len(self.data['y']))]

        cost_theta0 = sum(costs0) / len(costs0)
        cost_theta1 = sum(costs1) / len(costs1)

        """
        theta0_vals = [i for i in range(len(100))]
        theta1_vals = [i for i in range(len(100))]

        points = []
        for i in range(len(theta0_vals)):
            for j in range(len(theta1_vals)):
                cost = theta0_vals[i] *  
                points.append([theta0_vals[i], theta1_vals[j]])
        
        plt.plot(self.data['x'], costs0)
        plt.plot([parameters[1]], )
        plt.savefig('./linreg_progression/cost_function-sucrose_lab_data/theta0/cost_function{}'.format(count))
        plt.clf()
        plt.plot(self.data['x'], costs1)
        plt.savefig('./linreg_progression/cost_function-sucrose_lab_data/theta1/cost_function{}'.format(count))
        plt.clf()

        """
        return cost_theta0, cost_theta1

    def train_on_batch(self, training_data=[], lr=0.1):
        theta0 = 0
        cost = 1
        last_cost = 0
        cost_theta0 = 0
        cost_theta1 = 0
        theta1 = 0
        theta0 = 0
        count = 0

        while abs(cost - last_cost) > (10 ** -7):
            last_cost = abs(cost_theta0 + cost_theta1)
            cost_theta0, cost_theta1 = self.partial_deriv_cost_function((theta1, theta0), count)
            temp_theta0 = theta0 - (lr * cost_theta0)
            temp_theta1 = theta1 - (lr * cost_theta1)

            theta0 = temp_theta0
            theta1 = temp_theta1

            cost = abs(cost_theta0 + cost_theta1)

            print('[ theta0 {} theta1 {} ]\n[ cost_theta0 {} cost_theta1 {} ]\n\n'.format(theta0, theta1, cost_theta0, cost_theta1))

            print(cost - last_cost)
            
            """
            plt.scatter(x=self.data['x'], y=self.data['y']) 
            x_pred = self.data['x']
            y_pred = [(x_pred[i] * theta1) + theta0 for i in range(len(x_pred))]
            plt.plot(x_pred, y_pred)
            plt.xlabel('sucrose solution concenration')
            plt.ylabel('change in mass')
            plt.savefig('./linreg_progression/sucrose_lab_data/model{}'.format(count))
            
            count += 1
            """

            count += 1

        return theta0, theta1, cost