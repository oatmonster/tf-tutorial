import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import skimage
import os
import random


def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []

    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

train_data_directory = "./Training/"
test_data_directory = "./Testing/"

images, labels = load_data(train_data_directory)

# signs = [random.choice(images) for i in range(4)]

# for i in range(len(signs)):
#     plt.subplot(1, 4, i + 1)
#     plt.axis('off')
#     plt.imshow(signs[i])
#     plt.subplots_adjust(wspace = 0.5)
# plt.show()

unique_lables = set(labels)

plt.figure(figsize = (15, 15))

i = 1

for label in unique_lables:
    image = images[labels.index(label)]
    plt.subplot(8, 8, i)
    plt.axis('off')
    plt.title("label {0} ({1})".format(label, labels.count(label)))
    i += 1
    plt.imshow(image)
plt.show()