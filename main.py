from gradient_descent import GradientDescent
import pandas as pd

def linear_data_generation():
    return {'x': [0, 0.2, 0.4, 0.6, 0.8, 1], 'y': [0.04, 1.26, 2.64, 3.13, 4.6, 5.89]}
    # return {'x': [0, 1, 2, 3, 4, 5], 'y': [0, 1, 2, 3, 4, 5]}

    """
    df = pd.read_csv('Advertising.csv')
    x = df['TV']
    y = df['Radio']
    
    return {'x': list(x), 'y': list(y)}
    """


if __name__ == '__main__':
    gd = GradientDescent(optimization_type='linear_regression', _data=linear_data_generation())

    params = gd.train_on_batch()
    print(params)

    print('y = {}x + {}'.format(round(params[1], 2), round(params[0], 5)))