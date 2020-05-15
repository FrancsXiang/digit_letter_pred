import tensorflow as tf
from matplotlib import pyplot as plt
from data_loader import *

train_dir = 'C:/datasets/emnist/train/'
test_dir = 'C:/datasets/emnist/test/'
IMG_SIZE = 28
BATCH_SIZE = 32

image_lists, label_lists = get_files(train_dir)
image_batch, label_batch = get_batch(image_lists, label_lists, IMG_SIZE, IMG_SIZE, BATCH_SIZE, BATCH_SIZE * 4)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() :
                img,label = sess.run([image_batch,label_batch])
                for j in np.arange(BATCH_SIZE):
                    print('label:%d'%(label[j]))
                    img = img.reshape([BATCH_SIZE,IMG_SIZE,IMG_SIZE])
                    print(img)
                    plt.imshow(img[j,:,:])
                    plt.show()
                break
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)


