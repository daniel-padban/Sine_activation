import matplotlib.pyplot as plt
import numpy as np

train_x = np.arange(0,100,0.01)
test_x = range(100,200)
train_y_sin = np.sin(train_x)
test_y_sin = np.sin(test_x)

fig, ax = plt.subplots(1, 1, figsize=(40, 4))
ax.grid(True)
ax.axhline(0, linestyle='dashed',color='black')
ax.legend()
ax.plot(train_x,train_y_sin, label = 'train')
ax.plot(test_x,test_y_sin, label = 'test')

plt.show()
