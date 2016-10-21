import tensorflow as tf

import skimage
import skimage.io
import skimage.transform

import os
from glob import glob

def readims(path):
    pattern = os.path.join(im_dir, "*.png")
    c = skimage.io.ImageCollection(glob(pattern))
    all_images = c.concatenate()
    return all_images

im_dir = '/Users/ful6ru04/Documents/MATLAB/Segmentation/spine2'
xs = tf.placeholder(tf.float32,[None,512,512])
ims = readims(im_dir)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
im = sess.run(xs,feed_dict={xs:ims})
sess.close()

'''
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc

data = np.zeros((512,512,1,300))
for i in range(300):
    data[:,:,0,i] = misc.imread(im_dir+'spine2-'+str(i+1)+'.png')

img_tf = tf.Variable(data)
print img_tf.get_shape().as_list()
sess = tf.Session()
sess.run(tf.initialize_all_variables())
im = sess.run(img_tf)

fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(np.squeeze(im[:,:,:,0]))
fig.add_subplot(1,2,2)
plt.imshow(np.squeeze(im[:,:,:,1]))
plt.show()

sess.close()
'''
