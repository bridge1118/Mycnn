import skimage
import skimage.io
import skimage.transform

import os
from glob import glob

import numpy as np

class MyImages:
    
    def __ini__(self):
        pass
        
    def build(self,im_dir,batch_size,new_height=None,new_width=None,ext='png'):
        
        ext = '*.'+ext
        
        self.train_urls = os.path.join(im_dir,'train',ext)
        self.label_urls = os.path.join(im_dir,'label',ext)
        
        self.train_imgs = self.readims(self.train_urls)
        self.label_imgs = self.readims(self.label_urls)
        self.label_imgs = self.label_parsing(self.label_imgs)
        
        self.start = 0
        self.end = batch_size
        self.batch_size = batch_size
        
        self.new_height=new_height
        self.new_width =new_width
        
    def readims(self,path):
        c = skimage.io.ImageCollection(glob(path))
        all_images = c.concatenate()
        
        if all_images.ndim == 3:
            all_images = all_images[:,:,:,np.newaxis]
        all_images = np.uint8(all_images)
        return all_images
        
    def label_parsing(self,images):
        images[np.nonzero(images)] = 1
        return images
        
    def nextBatch(self):
        size = self.train_imgs.shape[0]
        batch_images = None
        batch_labels = None
        
        if self.end > self.start:
            batch_images = self.train_imgs[self.start:self.end,:,:,:]
            batch_labels = self.label_imgs[self.start:self.end,:,:,:]
        else:
            tmp1 = self.train_imgs[self.start:size,:,:,:]
            tmp2 = self.train_imgs[0:self.end,:,:,:]
            batch_images = np.concatenate((tmp1,tmp2),axis=0)
            
            tmp3 = self.label_imgs[self.start:size,:,:,:]
            tmp4 = self.label_imgs[0:self.end,:,:,:]
            batch_labels = np.concatenate((tmp3,tmp4),axis=0)
            
        if not self.new_height==None and not self.new_width==None:
            batch_images = np.resize(batch_images,(self.batch_size,
                                                   self.new_height,
                                                   self.new_width,1))
            batch_labels = np.resize(batch_labels,(self.batch_size,
                                                   self.new_height,
                                                   self.new_width,1))
        
        self.start = (self.start + self.batch_size) % size
        self.end = (self.end + self.batch_size) % size
        
        return batch_images, batch_labels
