import os
import model
import numpy as np
import tensorflow as tf
from PIL import Image

eval_dir = 'C:/datasets/emnist/test/'
check_point_path = './log/model/'

IMG_W = 28
IMG_H = 28
CHANNELS = 1
BATCH_SIZE = 1
NUM_CLASSES = 62
class_names = {}

def get_one_image_name():
    sub_folder = str(np.random.randint(0,62)) + '/'
    fetch_dir = eval_dir + sub_folder
    return os.path.join(fetch_dir, str(np.random.randint(0,len(os.listdir(fetch_dir)))) + '.jpg')

def pre_process():
    global class_names
    for i in range(0,10):
        class_names[i] = chr(i + 48)
    for i in range(10,36):
        class_names[i] = chr(i + 65 - 10)
    for i in range(36,62):
        class_names[i] = chr(i + 97 - 36)

def run_single_test():
    global class_names
    file_name = get_one_image_name()
    image_contents = tf.read_file(file_name, 'r')
    image = tf.image.decode_jpeg(image_contents, channels=1)
    image = tf.image.resize_image_with_crop_or_pad(image, IMG_H, IMG_W)
    image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, [1, IMG_H, IMG_W, CHANNELS])
    logits, _ = model.inference(image, BATCH_SIZE, NUM_CLASSES, training=False)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        if os.path.exists(os.path.join(check_point_path, 'checkpoint')):
            saver.restore(sess, tf.train.latest_checkpoint(check_point_path))
        else:
            print('There is no checkpoint!')
            return
        res = sess.run(logits)
        print('prediction result: %c' % class_names[np.argmax(res,1)[0]])
    img = Image.open(file_name, 'r')
    img.show()

pre_process()
run_single_test()
