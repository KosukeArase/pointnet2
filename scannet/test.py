import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
import itertools
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import scannet_dataset
# import show3d_balls


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--model', default='pointnet2_sem_seg', help='Model name [default: pointnet2_sem_seg.py]')

parser.add_argument('--dataset', default='s3dis', choices=['scannet', 's3dis'], help='Dataset name [default: scannet.py]')
parser.add_argument('--output_dir', default='result', help='Log dir [default: log]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
FLAGS = parser.parse_args()

MODEL_PATH = FLAGS.model_path
OUTPUT_PATH = FLAGS.output_dir
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
MODEL = importlib.import_module(FLAGS.model) # import network module
NUM_CLASSES = 21
DATA_PATH = os.path.join(ROOT_DIR,'data','{}_data_pointnet2'.format(FLAGS.dataset))

TEST_DATASET = scannet_dataset.ScannetDatasetVirtualScan(root=DATA_PATH, npoints=NUM_POINT, split='test', dataset=FLAGS.dataset)


if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)


def get_model(batch_size, num_point):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, _ = MODEL.placeholder_inputs(batch_size, num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES)
            # loss = MODEL.get_loss(pred, labels_pl, end_points)
            saver = tf.train.Saver()
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred} # ,
               # 'loss': loss}
        return sess, ops

def inference(sess, ops, pc, batch_size):
    ''' pc: BxNx3 array, return BxN pred '''
    assert pc.shape[0]%batch_size == 0
    num_batches = pc.shape[0]/batch_size
    logits = np.zeros((pc.shape[0], pc.shape[1], NUM_CLASSES))
    for i in range(num_batches):
        feed_dict = {ops['pointclouds_pl']: pc[i*batch_size:(i+1)*batch_size,...],
                     ops['is_training_pl']: False}
        batch_logits = sess.run(ops['pred'], feed_dict=feed_dict)
        logits[i*batch_size:(i+1)*batch_size,...] = batch_logits
    return np.argmax(logits, 2)

if __name__=='__main__':
    # import matplotlib.pyplot as plt
    # cmap = plt.cm.get_cmap("hsv", 4)
    # cmap = np.array([cmap(i) for i in range(10)])[:,:3]

    N = len(TEST_DATASET)
    pcs = np.empty([N, 8, NUM_POINT, 3])
    preds = np.empty([N, 8, NUM_POINT])
    gts = np.empty([N, 8, NUM_POINT])

    sess, ops = get_model(batch_size=1, num_point=NUM_POINT)

    for i, (data_idx, view_idx) in enumerate(itertools.product(range(len(TEST_DATASET)), range(8))):
        pc, gt, _ = TEST_DATASET[(data_idx, view_idx)]
        pred = inference(sess, ops, np.expand_dims(pc, 0), batch_size=1) 
        pred = pred.squeeze()

        pcs[data_idx, view_idx] = pc
        preds[data_idx, view_idx] = pred
        gts[data_idx, view_idx] = gt

        # gt = cmap[seg, :]
        # pred = cmap[segp, :]
        # show3d_balls.showpoints(ps, gt, pred, ballradius=8)

    result = {'pcs': pcs, 'preds': preds, 'gts': gts}

    experiment_name = MODEL_PATH.split('/')[-2]
    result_file = os.path.join(OUTPUT_PATH, experiment_name + '.pkl')

    print('Save result as {}'.format(result_file))
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)
