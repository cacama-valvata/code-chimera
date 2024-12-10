import numpy as np
import matplotlib.pyplot as plt


# Idea from  https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
def softplus_np(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def relu(x):
    # Why doesn't this work?
    # return np.max(0.0, x)
    return x * (x > 0)


fig, ax = plt.subplots(2, 3, figsize=(8, 6))

x = np.linspace(-5, 5, 101)

ax[0][0].plot(x, softplus_np(x))
ax[0][0].set_title("SoftPlus : log(1+x)")

ax[0][1].plot(x, relu(x))
ax[0][1].set_title("Relu : max(0, x)")

plt.suptitle("Activation Functions")

plt.tight_layout()

plt.show()
