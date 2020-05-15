import idx2numpy as reader
import os
from PIL import Image
import numpy as np

train_dir = 'C:/datasets/emnist/train/'
test_dir = 'C:/datasets/emnist/test/'
num_classes = 62
train_images = reader.convert_from_file('./emnist/emnist-byclass-train-images-idx3-ubyte')
train_labels = reader.convert_from_file('./emnist/emnist-byclass-train-labels-idx1-ubyte')
test_images = reader.convert_from_file('./emnist/emnist-byclass-test-images-idx3-ubyte')
test_labels = reader.convert_from_file('./emnist/emnist-byclass-test-labels-idx1-ubyte')

train_nums, img_size, _ = np.shape(train_images)
test_nums = np.shape(test_labels)[0]

# do not set the dataset_dir in working directory!
index = np.zeros([num_classes],'int32')
for i in range(train_nums):
    im = Image.fromarray(train_images[i])
    img_file = str(index[train_labels[i]]) + '.jpg'
    index[train_labels[i]] = index[train_labels[i]] + 1
    store_dir = train_dir + str(train_labels[i])
    if not os.path.isdir(store_dir):
        os.mkdir(store_dir)
    im.save(os.path.join(store_dir, img_file), quality=95, subsampling=0)

index = np.zeros([num_classes],'int32')
for i in range(test_nums):
    im = Image.fromarray(test_images[i])
    img_file = str(index[test_labels[i]]) + '.jpg'
    index[test_labels[i]] = index[test_labels[i]] + 1
    store_dir = test_dir + str(test_labels[i])
    if not os.path.isdir(store_dir):
        os.mkdir(store_dir)
    im.save(os.path.join(store_dir, img_file), quality=95, subsampling=0)






