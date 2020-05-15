import os
import math
import model
import tensorflow as tf
import data_loader

N_CLASSES = 62
IMG_W = 28
IMG_H = 28
BATCH_SIZE = 32
CAPACITY = 4 * BATCH_SIZE
MAX_STEPS = math.ceil(data_loader.total_need / BATCH_SIZE)
LEARNING_RATE = 1e-4

def run_training():
    train_dir = 'C:/datasets/emnist/train/'
    logs_summary_dir = './log/summary/train/'
    check_point_path = './log/model/'

    train, train_labels = data_loader.get_files(train_dir)
    train_batch, train_label_batch = data_loader.get_batch(train, train_labels, IMG_H, IMG_W, BATCH_SIZE, CAPACITY)
    train_logits, _ = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.training(train_loss, LEARNING_RATE)
    train_acc = model.evaluation(train_logits, train_label_batch)

    summery_op = tf.summary.merge_all()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(logs_summary_dir, graph=sess.graph, session=sess)
        saver = tf.train.Saver(max_to_keep=1)
        if os.path.exists(os.path.join(check_point_path,'checkpoint')):
            saver.restore(sess,tf.train.latest_checkpoint(check_point_path))
        else:
            sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for step in range(MAX_STEPS):
                if coord.should_stop(): break
                _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

                if step % 50 == 0:
                    print('The training loss and acc respectively: %.2f %.2f' % (tra_loss, tra_acc))
                    summary_total = sess.run(summery_op)
                    train_writer.add_summary(summary_total, global_step=step)

                if step % 2000 == 0 or (step + 1) == MAX_STEPS:
                    saver.save(sess, check_point_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print('training done!')
        finally:
            coord.request_stop()
    coord.join(threads)

run_training()