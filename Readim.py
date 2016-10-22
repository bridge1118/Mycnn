import tensorflow as tf

import skimage
import skimage.io
import skimage.transform

import os
from glob import glob

import numpy as np

class MyImages:
    
    def __ini__(self):
        pass
        
    def build(self,im_dir,batch_size,ext='png'):
        ext = '*.'+ext
        self.train_urls = os.path.join(im_dir,'train',ext)
        self.label_urls = os.path.join(im_dir,'label',ext)
        self.train_imgs = self.readims(self.train_urls)
        self.label_imgs = self.readims(self.label_urls)
        self.label_imgs = self.label_parsing(self.label_imgs)
        self.start = 0
        self.end = batch_size
        self.batch_size = batch_size
        
    def readims(self,path):
        #pattern = os.path.join(path, "*.png")
        #c = skimage.io.ImageCollection(glob(pattern))
        c = skimage.io.ImageCollection(glob(path))
        all_images = c.concatenate()
        
        if all_images.ndim == 3:
            all_images = all_images[:,:,:,np.newaxis]
        all_images.astype(float)
        return all_images
        
    def label_parsing(self,images):
        images[np.nonzero(images)] = 1
        return images
    def nextBatch(self):
        size = self.train_imgs.shape[0]
        batch_images = None
        
        if self.end > self.start:
            batch_images = self.train_imgs[self.start:self.end,:,:,:]
            batch_labels = self.label_imgs[self.start:self.end,:,:,:]
        else:
            tmp1 = self.train_imgs[self.start:size-1,:,:,:]
            tmp2 = self.train_imgs[0:self.end,:,:,:]
            batch_images = np.concatenate((tmp1,tmp2),axis=0)
            
            tmp3 = self.label_imgs[self.start:size-1,:,:,:]
            tmp4 = self.label_imgs[0:self.end,:,:,:]
            batch_labels = np.concatenate((tmp3,tmp4),axis=0)
            
        batch_images = np.resize(batch_images,(self.batch_size,500,500,1))
        batch_labels = np.resize(batch_labels,(self.batch_size,500,500,1))
        
        self.start = (self.start + self.batch_size) % size
        self.end = (self.end + self.batch_size) % size
        
        return batch_images, batch_labels
'''
im_dir = '/Users/ful6ru04/Documents/TensorFlow workspace/Mycnn/SPINE_data'
xs = tf.placeholder(tf.float32,[10,512,512,1])
myImages = MyImages()
myImages.build(im_dir,'png')
ims = myImages.nextBatch(10)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
im = sess.run(xs,feed_dict={xs:ims})
sess.close()
'''