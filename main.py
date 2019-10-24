import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import transform
from skimage.color import rgb2gray
import os
import random

# ============================================
# Setup
# ============================================

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

images28 = [transform.resize(image, (28, 28)) for image in images]

images28 = np.array(images28)

images28 = rgb2gray(images28)

# ============================================
# Training
# ============================================

# Initialize placeholders
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

# Optimizer
train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(201):
    print("EPOCH", i)
    _, accuracy_val = sess.run([train_op, accuracy], feed_dict = {x: images28, y: labels})
    if i % 10 == 0:
        print("Loss: ", loss)
    print("DONE WITH EPOCH")

# ============================================
# Testing
# ============================================

test_images, test_labels = load_data(test_data_directory)

test_images = [transform.resize(image, (28, 28)) for image in test_images]

test_images28 = rgb2gray(np.array(test_images))

predicted = sess.run([correct_pred], feed_dict = {x: test_images28})[0]

match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

accuracy = match_count / len(test_labels)

print("Accuracy: {:.3f}".format(accuracy))

# ============================================
# Display Basic Stats
# ============================================

# signs = [random.choice(images) for i in range(4)]

# for i in range(len(signs)):
#     plt.subplot(1, 4, i + 1)
#     plt.axis('off')
#     plt.imshow(signs[i])
#     plt.subplots_adjust(wspace = 0.5)
# plt.show()

# ============================================
# Display Labels
# ============================================

# unique_lables = set(labels)

# plt.figure(figsize = (15, 15))

# i = 1

# for label in unique_lables:
#     image = images[labels.index(label)]
#     plt.subplot(8, 8, i)
#     plt.axis('off')
#     plt.title("label {0} ({1})".format(label, labels.count(label)))
#     i += 1
#     plt.imshow(image)
# plt.show()