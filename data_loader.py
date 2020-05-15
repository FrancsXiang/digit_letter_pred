import tensorflow as tf
import numpy as np
import os
import math
num_classes = 62
total_need = 400000.0
class_imgs = math.ceil(total_need / num_classes)

def get_files(file_dir):
    labels = []
    images = []
    for i in range(num_classes):
        counter = 0
        seen = []
        max_index = len(os.listdir(file_dir + str(i)))
        class_need = min(class_imgs,max_index)
        while counter < class_need:
            index = int(np.random.random() * max_index)
            if index in seen:
                continue
            labels.append(i)
            images.append(os.path.join(file_dir + str(i) + '/', str(index) + '.jpg'))
            counter = counter + 1
            seen.append(index)
    return images, labels

def get_batch(image, label, image_H, image_W, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=1)
    image = tf.image.resize_image_with_crop_or_pad(image, image_H, image_W)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, label_batch