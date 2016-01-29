# Author: Scott Workman
# Date  : 1/29/2016

import glob
import caffe
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from collections import defaultdict

class LoadImage:
  def __init__(self, mean):
    self.mean = mean

  def __call__(self, file):
    im = caffe.io.load_image(file)
    im = caffe.io.resize_image(im, (256,256))

    # channel swap for pre-trained (RGB -> BGR)
    im = im[:, :, [2,1,0]]
    
    # make channels x height x width
    im = im.swapaxes(0,2).swapaxes(1,2)

    # convert to uint8
    im = (255*im).astype(np.uint8, copy=False) 
    
    # subtract mean & resize
    caffe_input = im - self.mean
    caffe_input = caffe_input.transpose((1,2,0))
    caffe_input = caffe.io.resize_image(caffe_input, (227,227))
    caffe_input = caffe_input.transpose((2,0,1))
    caffe_input = caffe_input.reshape((1,)+caffe_input.shape)

    return caffe_input

def load_images(files, mean):

  proc = LoadImage(mean)
  pool = multiprocessing.Pool(processes=20)
  results = pool.map(proc, files) 

  if len(files) == 1:
    return results[0]
  else:
    return np.vstack(results)

def compute_features(net, data_blob, blobs, images):

  features = defaultdict(list) 

  kw = {data_blob: images, 'blobs': blobs}
  out = net.forward_all(**kw)
    
  for blob_name in blobs:
    features[blob_name].append(out[blob_name])

  return features

def load_net(deploy_file, model_file):
  caffe.set_mode_gpu()
  net = caffe.Net(deploy_file, model_file, caffe.TEST)
  return net

def load_mean(mean_file):
  blob = caffe.io.caffe_pb2.BlobProto()
  f = open(mean_file,'rb')
  blob.ParseFromString(f.read())
  f.close()
  means = caffe.io.blobproto_to_array(blob)
  means = means[0]
  return means

def compute_similarity(a_feats, g_feat):
  ret = []
  for a_feat in a_feats:
    val = np.linalg.norm(g_feat-a_feat)
    ret.append(val)
  return np.vstack(ret)
  
  
if __name__ == '__main__': 
 
  image_dir = './images/'
  model_dir = '../models/'

  query_im = image_dir + '60949863@N02_7984662477_43.533763_-89.290620.jpg'
  region_im = image_dir + 'region_43.529763_-89.294620_43.537763_-89.286620.jpg'
  aerial_ims = glob.glob('%saerial/*.jpg' % image_dir)
  
  #
  # setup nets
  #
 
  weights_file = model_dir + 'cvplaces/cvplaces.caffemodel'
  g_deploy_file = model_dir + 'cvplaces/ground_deploy.net'
  a_deploy_file = model_dir  + 'cvplaces/aerial_deploy.net'
  g_mean_file = model_dir + 'mean_files/places205CNN_mean.binaryproto'
  a_mean_file = model_dir  + 'mean_files/aerial_mean.binaryproto'

  g_net = load_net(g_deploy_file, weights_file)
  g_mean = load_mean(g_mean_file)
  a_net = load_net(a_deploy_file, weights_file)
  a_mean = load_mean(a_mean_file)
 
  #
  # process images 
  #

  g_data_blob = 'data_g'
  a_data_blob = 'data_a'
  g_blobs = ['fc8_g'] 
  a_blobs = ['fc8_a'] 

  print "processing images"
  features = compute_features(g_net, g_data_blob, g_blobs, load_images([query_im], g_mean))
  g_feat = features['fc8_g'] 

  features = compute_features(a_net, a_data_blob, a_blobs, load_images(aerial_ims, a_mean))
  a_feats = np.vstack(features['fc8_a'])

  print "computing scores"
  scores = np.squeeze(compute_similarity(a_feats, g_feat))

  #
  # visualize result 
  # 

  sorted_inds = np.argsort(scores)
  im_neighbor1 = caffe.io.load_image(aerial_ims[sorted_inds[0]]) 
  im_neighbor2 = caffe.io.load_image(aerial_ims[sorted_inds[1]]) 
  im_neighbor3 = caffe.io.load_image(aerial_ims[sorted_inds[2]]) 
  im_neighbor4 = caffe.io.load_image(aerial_ims[sorted_inds[3]]) 

  im_query = caffe.io.load_image(query_im)
  im_region = caffe.io.load_image(region_im)
  
  print "visualizing"
  fig = plt.figure(1)
  fig.suptitle('Query Image')
  plt.imshow(im_query, aspect='auto')
  plt.axis('off')

  fig = plt.figure(2)
  fig.suptitle('Region of Interest')
  plt.imshow(im_region, aspect='auto')
  plt.axis('off')

  fig = plt.figure(3)
  fig.suptitle('4 Nearest Neighbors')
  ax1 = fig.add_subplot(221)
  ax1.imshow(im_neighbor1, aspect='auto')
  plt.axis('off')
  ax2 = fig.add_subplot(222)
  ax2.imshow(im_neighbor2, aspect='auto')
  plt.axis('off')
  ax3 = fig.add_subplot(223)
  ax3.imshow(im_neighbor3, aspect='auto')
  plt.axis('off')
  ax4 = fig.add_subplot(224)
  ax4.imshow(im_neighbor4, aspect='auto')
  plt.axis('off')

  plt.show()
