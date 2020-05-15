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

eval_log_dir = './log/summary/test/'
eval_dir = 'C:/datasets/emnist/test/'
check_point_path = './log/model/'
eva, eva_labels = data_loader.get_files(eval_dir)
eval_batch, eval_label_batch = data_loader.get_batch(eva, eva_labels, IMG_H, IMG_W, BATCH_SIZE, CAPACITY)
eval_logits = model.inference(eval_batch, BATCH_SIZE, N_CLASSES, training=False)
eval_acc = model.evaluation(eval_logits, eval_label_batch)
eval_loss = model.losses(eval_logits, eval_label_batch)
summery_op = tf.summary.merge_all()


def run_testing():
    TOTAL_ACC_SUM = 0
    TOTAL_LOSS_SUM = 0
    with tf.Session() as sess:
        eval_writer = tf.summary.FileWriter(eval_log_dir, sess.graph, session=sess)
        saver = tf.train.Saver()
        if os.path.exists(os.path.join(check_point_path, 'checkpoint')):
            saver.restore(sess, tf.train.latest_checkpoint(check_point_path))
        else:
            print('There is no checkpoint!')
            return
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for step in range(MAX_STEPS):
                if coord.should_stop(): break
                test_acc, test_loss = sess.run([eval_acc, eval_loss])
                if step % 100 == 0:
                    print('step %d The testing acc and loss respectively: %.2f %.2f' % (step, test_acc, test_loss))
                    summary_total = sess.run(summery_op)
                    eval_writer.add_summary(summary_total, global_step=step)
                TOTAL_ACC_SUM += test_acc
                TOTAL_LOSS_SUM += test_loss
        except tf.errors.OutOfRangeError:
            print('testing done!')
        finally:
            coord.request_stop()
    coord.join(threads)
    print('The average testing acc and loss respectively:%.2f %.2f' %
          ((TOTAL_ACC_SUM / MAX_STEPS), (TOTAL_LOSS_SUM / MAX_STEPS)))

run_testing()