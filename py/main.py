import numpy as np

nSamples = 200
nFeatures = 7
nAttributes = 1

pha = 0.5
alpha = np.array([0.000001, 0.000005, 0.00001, 0.00005])

X_test = np.random.randn(nSamples, nFeatures) # One row for one sample
Y_test = np.random.randn(nSamples, nAttributes)
# print(X_test)

print(np.invert(X_test))


def lossFunction(x_batch, y_batch, batch_length, beta, pha):
    Losses = np.zeros([batch_length])
    for i in np.arange(batch_length):
        Losses[i] = X_test


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
