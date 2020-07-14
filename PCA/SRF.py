import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from PCA.PixelHop2 import readImgToNumpy_pairs


def PlotImage(image):
    """
	PlotImage: Give a normalized image matrix which can be used with implot, etc.
	Maps to [0, 1]
	"""
    im = image.astype(float)
    return (im - np.min(im)) / (np.max(im) - np.min(im))


def SRM(imgs):
    filter1 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    filter2 = [[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]]
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 1, -2, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    q = [4.0, 12.0, 2.0]
    filter1 = np.asarray(filter1, dtype=float) / 4
    filter2 = np.asarray(filter2, dtype=float) / 12
    filter3 = np.asarray(filter3, dtype=float) / 2
    filters = [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]]
    filters = np.einsum('klij->ijlk', filters)
    filters = tf.Variable(filters, dtype=tf.float32)
    imgs = np.array(imgs, dtype=float)
    input = tf.Variable(imgs, dtype=tf.float32)
    op = tf.nn.conv2d(input, filters, strides=[1, 1, 1, 1], padding='SAME')

    q = [4.0, 12.0, 2.0]
    filter1 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    filter2 = [[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]]
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 1, -2, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    filter1 = np.asarray(filter1, dtype=float) / q[0]
    filter2 = np.asarray(filter2, dtype=float) / q[1]
    filter3 = np.asarray(filter3, dtype=float) / q[2]
    filters = [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]]
    filters = np.einsum('klij->ijlk', filters)
    filters = filters.flatten()
    initializer_srm = tf.constant_initializer(filters)

    def truncate_2(x):
        neg = ((x + 2) + abs(x + 2)) / 2 - 2
        return -(-neg + 2 + abs(- neg + 2)) / 2 + 2

    op2 = slim.conv2d(input, 3, [5, 5], trainable=False, weights_initializer=initializer_srm,
                      activation_fn=None, padding='SAME', stride=1)
    op2 = truncate_2(op2)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        re = (sess.run(op))
        res = np.round(re[0])
        res[res > 2] = 2
        res[res < -2] = -2

        res2 = sess.run(op2)
        # print(sum(sum(sum(sum(res2>2)))))
    ress2 = np.array(res2, dtype=float)
    ress = np.array([res], dtype=float)
    # input = tf.Variable(ress, dtype=tf.float32)
    # op = tf.nn.conv2d(input, filters_coocurr, strides=[1, 1, 1, 1], padding='SAME')
    # with tf.Session() as sess:
    #     sess.run(tf.initialize_all_variables())
    #     res = (sess.run(op))
    return ress, ress2


if __name__ == '__main__':

    ori_images, stego_images = readImgToNumpy_pairs('/mnt/zhengwen/image_steganalysis/new_100/ori',
                                      '/mnt/zhengwen/image_steganalysis/new_100/SUNI_08', mode=1)
    ori_images = ori_images.astype("double")
    stego_images = stego_images.astype("double")
    ori_noise, ori_noise_trun = SRM(ori_images)
    np.save('ori_noise_trun.npy', ori_noise_trun)
    stego_noise, stego_noise_trun = SRM(stego_images)
    np.save('stego_noise_trun.npy', stego_noise_trun)
