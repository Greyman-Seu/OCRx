import ipdb
import os
import matplotlib.pyplot as plt
import numpy as np

# workdir = '/home/chuan/captcha/data/raw/fake_pic_train/'
workdir = '/home/chuan/dataset/captcha/train'
files = os.listdir(workdir)
labels, colors = [], []
for file in files:
    if file.endswith('.png'):
        _, label, color = file.split('_')
        labels.extend(label)
        colors.extend(color[:-4])

# deal with labels
vocab = {}
for char in labels:
    if char in vocab:
        vocab[char] += 1
    else:
        vocab[char] = 1
keys = np.array(list(vocab.keys()))
values = np.array(list(vocab.values()))
sorted_index = np.argsort(values)
print(keys[sorted_index], values[sorted_index], sep='\n')

plt.scatter(range(len(vocab)), values)
plt.savefig('labels.png')

# deal with colors
colors = set(colors)
print(colors)
