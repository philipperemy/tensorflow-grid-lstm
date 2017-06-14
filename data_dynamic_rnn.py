import numpy as np

# [
# [
# [
# [           (i,j-1)
# [
# [  (i-1,j)  (i,j)
# [
# [

D = 140
N = 62


def validator(o):
    x_o = []
    x_hor_i = []
    x_vert_i = []
    d, n = o.shape
    for i in range(d):
        for j in range(n):
            x_o.append(o[i, j])
            x_hor_i.append(o[i - 1, j])
            x_vert_i.append(o[i, j - 1])

    cor1 = np.corrcoef(x_o, x_hor_i)[0, 1]
    cor2 = np.corrcoef(x_o, x_vert_i)[0, 1]
    cor3 = np.corrcoef(x_hor_i, x_vert_i)[0, 1]
    # print(cor1)
    # print(cor2)
    # print(cor3)
    return cor1, cor2, cor3


def gen_grid_data(d=D, n=5, rho=0.8):
    o = np.random.standard_normal((d, n))
    for i in range(d):
        for j in range(n):
            # o[i, j] = o[i - 1, j] * rho + np.sqrt(1 - rho ** 2) * np.random.standard_normal()
            o[i, j] = (rho / 2) * o[i - 1, j] \
                      + (rho / 2) * o[i, j - 1] \
                      + np.sqrt(1 - (rho / 2) ** 2) * np.random.standard_normal()
    # o = (o - np.mean(o)) / np.std(o)
    return o


"""

x1 = rnorm(10000)
x2 = rnorm(10000)

a = 0.5
b = 0.5

x3 = a * x1 + b * x2 + sqrt(1-(a^2 + b^2)) * rnorm(10000)

cor(x3, x1)
cor(x3, x2)
cor(x1, x2)

> cor(x3, x1)
[1] 0.5049171
> cor(x3, x2)
[1] 0.508686
> cor(x1, x2)
[1] 0.009731089
"""


def gen_grid_data_2(d=D, n=N, rho_1=0.3, rho_2=0.3):
    # here it's less accurate because the data deviate from N(0, 1)
    o = np.random.standard_normal((d, n))
    for i in range(d):
        for j in range(n):
            o[i, j] = rho_1 * o[i - 1, j] \
                      + rho_2 * o[i, j - 1] \
                      + np.sqrt(1 - (rho_1 ** 2 + rho_2 ** 2)) * np.random.standard_normal()
    # o = (o - np.mean(o)) / np.std(o)
    return o


def per_stock_gen(d=D, n=N):
    c1 = 0.0
    c2 = 0.0
    c3 = 1.0
    while c3 > 0.12 or c1 < 0.3 or c2 < 0.3:
        stock_data = gen_grid_data_2(d, n)
        # print('min   =', np.min(stock_data))
        # print('max   =', np.max(stock_data))
        # print('mean  =', np.mean(stock_data))
        # print('std   =', np.std(stock_data))
        # print('shape =', stock_data.shape)
        c1, c2, c3 = validator(stock_data)
    return stock_data


def next_batch(bs=10, d=D, n=N):
    x = []
    for i in range(bs):
        x.append(per_stock_gen(d, n))
    x = np.array(x)
    # print(x.shape)

    y = np.roll(x, shift=-1, axis=2)
    y[:, :, -1] = 0.0
    # np.set_printoptions(precision=1)
    # print('x=')
    # print(np.matrix(x[0]))
    # print('y=')
    # print(np.matrix(y[0]))

    """
    y = np.roll(x, shift=-1, axis=2)
    np.set_printoptions(precision=1)
    print('x=')
    print(np.matrix(x[0]))
    print('y=')
    print(np.matrix(y[0]))
    """

    """
        time of day ->
        days |
             v
    x=
    [[ 1.4  0.2 -0.9 -0.2 -0.4  1.7  0.7  2.4  1.2 -0.1]
     [-0.5  0.1 -0.5  1.6  0.1  1.7  2.2  1.5  0.6 -1. ]
     [-1.1 -1.5 -0.7 -1.1 -1.2  0.9  1.4  2.1  1.7  1.7]
     [ 0.2 -0.2  0.5  0.5 -0.7  0.9  0.5  1.5  1.1  0.6]
     [-0.5 -0.2 -0.6 -0.4  1.3  0.5 -0.7  0.2  0.9  0.1]]
    y=
    [[ 0.2 -0.9 -0.2 -0.4  1.7  0.7  2.4  1.2 -0.1  0. ]
     [ 0.1 -0.5  1.6  0.1  1.7  2.2  1.5  0.6 -1.   0. ]
     [-1.5 -0.7 -1.1 -1.2  0.9  1.4  2.1  1.7  1.7  0. ]
     [-0.2  0.5  0.5 -0.7  0.9  0.5  1.5  1.1  0.6  0. ]
     [-0.2 -0.6 -0.4  1.3  0.5 -0.7  0.2  0.9  0.1  0. ]]
    """

    return x, y


if __name__ == '__main__':
    output = next_batch(bs=10, d=5, n=10)
    # output = next_batch(bs=2)
    np.save('data.npy', output)
