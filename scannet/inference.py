import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
import itertools
import random
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

import provider
import tf_util
import pc_util
import scannet_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--model', default='pointnet2_sem_seg', help='Model name [default: pointnet2_sem_seg.py]')
parser.add_argument('--num_classes', default=13, help='Number of classes')
parser.add_argument('--dataset', default='s3dis', choices=['scannet', 's3dis'], help='Dataset name [default: scannet.py]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--whole', action='store_true', help='Use whole scan (not virtual scan)')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

MODEL_PATH = FLAGS.model_path
OUTPUT_PATH = os.path.join(os.path.dirname(MODEL_PATH), "result")
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
print("Save output to {}".format(OUTPUT_PATH))

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu

MODEL = importlib.import_module(FLAGS.model)  # import network module

LOG_FOUT = open(os.path.join(OUTPUT_PATH, 'log_inference.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

# Shapenet official train/test split
DATA_PATH = os.path.join(ROOT_DIR, 'data', '{}_data_pointnet2'.format(FLAGS.dataset))

if FLAGS.whole:
    print('Use whole scan data')
    TEST_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', dataset=FLAGS.dataset, num_classes=FLAGS.num_classes)
    TEST_DATASET_WHOLE_SCENE = scannet_dataset.ScannetDatasetWholeScene(root=DATA_PATH, npoints=NUM_POINT, split='test', dataset=FLAGS.dataset, num_classes=FLAGS.num_classes)
else:
    print('Use virtual scan data')
    TEST_DATASET = scannet_dataset.ScannetDatasetVirtualScan(root=DATA_PATH, npoints=NUM_POINT, split='test', dataset=FLAGS.dataset, num_classes=FLAGS.num_classes)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def main():
    print("Start loading data")
    test_dataset_whole_scene = TEST_DATASET_WHOLE_SCENE.load_data()
    print("Finish loading data")
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, borders_pl, _ = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            print("--- Get model")
            pred_class, pred_border, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, FLAGS.num_classes)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'borders_pl': borders_pl,
               'is_training_pl': is_training_pl,
               'pred_class': pred_class,
               'pred_border': pred_border}

        acc = inference_whole_scene(sess, ops, test_dataset_whole_scene)


def get_whole(dataset, data_methods, idx):
    scene_points_list = dataset["scene_points_list"]
    semantic_labels_list = dataset["semantic_labels_list"]
    borders_list = dataset["borders_list"]
    instance_ids_list = dataset["instance_ids_list"]
    # virtual_smpidx = dataset["virtual_smpidx"]

    scene_point = scene_points_list[idx].copy()
    semantic_label = semantic_labels_list[idx].copy()
    instance_id = instance_ids_list[idx].copy()
    borders = borders_list[idx].copy()
    ps, seg, border, _, nx, ny, instance_id = data_methods.sample(scene_point, semantic_label, borders, instance_id)

    return ps, seg, border, None, nx, ny, instance_id


# evaluate on whole scenes to generate numbers provided in the paper
def inference_whole_scene(sess, ops, dataset):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    num_data = len(TEST_DATASET_WHOLE_SCENE)

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION WHOLE SCENE----' % (EPOCH_CNT))

    all_data = []
    class_accs = []
    border_accs = []

    for data_ind in range(num_data):
        batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 3))
        batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
        batch_instance = np.zeros((BATCH_SIZE, NUM_POINT))
        batch_border = np.zeros((BATCH_SIZE, NUM_POINT, 1))

        data, label, border, _, nx, ny, instance = get_whole(dataset, TEST_DATASET_WHOLE_SCENE, data_ind)
        n_cubes = data.shape[0]
        assert n_cubes <= BATCH_SIZE

        batch_data[:n_cubes, ...] = data
        batch_label[:n_cubes, ...] = label
        batch_instance[:n_cubes, ...] = instance
        batch_border[:n_cubes, ...] = border

        aug_data = batch_data
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['is_training_pl']: is_training}
        pred_val_class, pred_val_border = sess.run([ops['pred_class'], ops['pred_border']], feed_dict=feed_dict)

        pred_val_class = np.argmax(pred_val_class, 2)  # BxN
        pred_val_border = np.round(pred_val_border)  # BxN

        class_acc = np.sum(pred_val_class[:n_cubes] == batch_label[:n_cubes]) / float(NUM_POINT * n_cubes)
        border_acc = np.sum(pred_val_border[:n_cubes] == batch_border[:n_cubes]) / float(NUM_POINT * n_cubes)
        class_accs.append(class_acc)
        border_accs.append(border_acc)

        print("class acc: {} ({}/{})".format(class_acc, np.sum(pred_val_class[:n_cubes] == batch_label[:n_cubes]), NUM_POINT * n_cubes))
        print("border acc: {} ({}/{})".format(border_acc, np.sum(pred_val_border[:n_cubes] == batch_border[:n_cubes]), NUM_POINT * n_cubes))
        print("============")

        all_data.append({"data": data[:n_cubes], "label": label, "border": border, "pred_class": pred_val_class, "pred_border": pred_val_border, "nx": nx, "ny": ny, "instance": instance})

    print("mean class acc: {}".format(sum(class_accs)/float(len(class_accs))))
    print("mean border acc: {}".format(sum(border_accs)/float(len(border_accs))))

    with open("result.pkl", "wb") as f:
        pickle.dump(all_data, f)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    main()
    LOG_FOUT.close()
