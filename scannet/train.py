import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import itertools
import random
import joblib
import time
import multiprocessing as mp
from contextlib import closing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

import provider  # noqa
import tf_util  # noqa
import pc_util  # noqa
import scannet_dataset  # noqa

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 1]')
parser.add_argument('--model', default='pointnet2_sem_seg', help='Model name [default: pointnet2_sem_seg.py]')
parser.add_argument('--dataset', default='s3dis', choices=['scannet', 's3dis'], help='Dataset name [default: s3dis]')
parser.add_argument('--num_classes', default=13, help='Number of classes')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--whole', action='store_true', help='Use whole scan (not virtual scan)')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# Shapenet official train/test split
DATA_PATH = os.path.join(ROOT_DIR, 'data', '{}_data_pointnet2'.format(FLAGS.dataset))

if FLAGS.whole:
    print('Use whole scan data')
    TRAIN_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', dataset=FLAGS.dataset, num_classes=FLAGS.num_classes)
    TEST_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', dataset=FLAGS.dataset, num_classes=FLAGS.num_classes)
    TEST_DATASET_WHOLE_SCENE = scannet_dataset.ScannetDatasetWholeScene(root=DATA_PATH, npoints=NUM_POINT, split='test', dataset=FLAGS.dataset, num_classes=FLAGS.num_classes)
else:
    print('Use virtual scan data')
    # train_batch, init_op = scannet_dataset.scannet_dataset(root=DATA_PATH, npoints=NUM_POINT, split='train', whole=FLAGS.whole, dataset=FLAGS.dataset)
    TRAIN_DATASET = scannet_dataset.ScannetDatasetVirtualScan(root=DATA_PATH, npoints=NUM_POINT, split='train', dataset=FLAGS.dataset, num_classes=FLAGS.num_classes)
    TEST_DATASET = scannet_dataset.ScannetDatasetVirtualScan(root=DATA_PATH, npoints=NUM_POINT, split='test', dataset=FLAGS.dataset, num_classes=FLAGS.num_classes)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    print("Start loading data")
    train_dataset = TRAIN_DATASET.load_data()
    test_dataset = TEST_DATASET.load_data()
    test_dataset_whole_scene = TEST_DATASET_WHOLE_SCENE.load_data()
    print("Finish loading data")
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, borders_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss
            pred_class, pred_border, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, FLAGS.num_classes, bn_decay=bn_decay)
            class_loss, border_loss = MODEL.get_loss(pred_class, pred_border, labels_pl, borders_pl, smpws_pl)
            total_loss = class_loss + border_loss
            tf.summary.scalar('total loss', total_loss)

            correct_class = tf.equal(tf.argmax(pred_class, 2), tf.to_int64(labels_pl))
            accuracy_class = tf.reduce_sum(tf.cast(correct_class, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy class', accuracy_class)

            correct_border = tf.equal(tf.cast(tf.round(pred_border), dtype=tf.int64), tf.to_int64(borders_pl))
            accuracy_border = tf.reduce_sum(tf.cast(correct_border, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy border', accuracy_border)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'borders_pl': borders_pl,
               'smpws_pl': smpws_pl,
               'is_training_pl': is_training_pl,
               'pred_class': pred_class,
               'pred_border': pred_border,
               'class_loss': class_loss,
               'border_loss': border_loss,
               'total_loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_acc = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            train_one_epoch(sess, ops, train_writer, train_dataset)
            if epoch % 5 == 0:
                acc = eval_one_epoch(sess, ops, test_writer, test_dataset)
                if FLAGS.whole:
                    acc = eval_whole_scene_one_epoch(sess, ops, test_writer, test_dataset_whole_scene)

            if acc > best_acc:
                best_acc = acc
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt" % (epoch)))
                log_string("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


# def process(args):
def process(_idx, _scene_points_list, _semantic_labels_list, _borders_list, _virtual_smpidx, _dataset, _data_methods, _send_end):
    # _idx, _scene_points_list, _semantic_labels_list, _borders_list, _virtual_smpidx, _dataset, _data_methods = args
    while True:
        if FLAGS.whole:
            try:
                _scene_point = _scene_points_list[_idx].copy()
                _semantic_label = _semantic_labels_list[_idx].copy()
                _borders = _borders_list[_idx].copy()
                _ps, _seg, _border, _smpw = _data_methods.sample(_scene_point, _semantic_label, _borders)
                break
            except Exception as e:
                print(e)
                _old_idx = _idx
                _idx = np.random.randint(len(_dataset))
                print('Data-{} is invalid. Instead, use data-{}'.format(_old_idx, _idx))
        else:
            pass
        """
                    try:
                        _data_idx, _view_idx = _idx
                        _scene_point = scene_points_list[_data_idx]
                        _semantic_label = semantic_labels_list[_data_idx]
                        _borders = borders_list[_data_idx]
                        _smpidx = virtual_smpidx[_data_idx][_view_idx][0]
                        _ps, _seg, _border, _smpw = _data_methods.sample(_scene_point, _semantic_label, _borders, _smpidx, _view_idx)
                    except Exception as e:
                        print(e)
                        _old_data_idx, _old_view_idx = _idx
                        _data_idx = np.random.randint(len(_dataset))
                        _view_idx = np.random.randint(8)
                        _idx = (_data_idx, _view_idx)
                        print('Data-{} from view-{} is invalid. Instead, use data-{} from view-{}'.format(_old_data_idx, _old_view_idx, _data_idx, _view_idx))
        """
    _dropout_ratio = np.random.random()*0.875  # 0-0.875
    _drop_idx = np.where(np.random.random((_ps.shape[0])) <= _dropout_ratio)[0]
    _ps[_drop_idx, :] = _ps[0, :]
    _seg[_drop_idx] = _seg[0]
    _border[_drop_idx] = _border[0]
    _smpw[_drop_idx] *= 0

    return _send_end.send([_ps, _seg, _border, _smpw])


def get_batch_wdp(dataset, data_methods, idxs, start_idx, end_idx):
    bsize = end_idx - start_idx
    with mp.Manager() as manager:
        scene_points_list = manager.list(dataset["scene_points_list"])
        semantic_labels_list = manager.list(dataset["semantic_labels_list"])
        borders_list = manager.list(dataset["borders_list"])
        if not FLAGS.whole:
            virtual_smpidx = manager.list(dataset["virtual_smpidx"])
        else:
            virtual_smpidx = manager.list()

        # results = joblib.Parallel(n_jobs=2, verbose=10)([joblib.delayed(process)(idxs[i+start_idx]) for i in range(bsize)])
        """
        p = mp.Pool(processes=4)
        results = p.map(process, [(idxs[i+start_idx], scene_points_list, semantic_labels_list, borders_list, virtual_smpidx, dataset, data_methods) for i in range(bsize)])
        p.close()
        print(results)
        """
        args_const = [scene_points_list, semantic_labels_list, borders_list, virtual_smpidx, dataset, data_methods]
        jobs = []
        pipe_list = []

        for i in range(bsize):
            recv_end, send_end = mp.Pipe(False)
            p = mp.Process(target=process, args=[idxs[i+start_idx]] + args_const + [send_end])
            jobs.append(p)
            pipe_list.append(recv_end)
            p.start()

        for p in jobs:
            p.join()
        results = [pipe.recv() for pipe in pipe_list]

    batch_data   = np.array([ps     for ps, _, _, _     in results])
    batch_label  = np.array([seg    for _, seg, _, _    in results])
    batch_border = np.array([border for _, _, border, _ in results])
    batch_smpw   = np.array([smpw   for _, _, _, smpw   in results])

    return batch_data, batch_label, batch_border, batch_smpw


def get_batch(dataset, data_methods, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    scene_points_list = dataset["scene_points_list"]
    semantic_labels_list = dataset["semantic_labels_list"]
    borders_list = dataset["borders_list"]
    virtual_smpidx = dataset["virtual_smpidx"]
    """
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_border = np.zeros((bsize, NUM_POINT, 1), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    """

    def process(_idx):
        if not FLAGS.whole:
            _idx = (_idx, random.randint(0, 7))
        while True:
            if FLAGS.whole:
                try:
                    _scene_point = scene_points_list[_idx].copy()
                    _semantic_label = semantic_labels_list[_idx].copy()
                    _borders = borders_list[_idx].copy()
                    _ps, _seg, _border, _smpw = data_methods.sample(_scene_point, _semantic_label, _borders)
                    break
                except Exception as e:
                    print(e)
                    _old_idx = _idx
                    _idx = np.random.randint(len(dataset))
                    print('Data-{} is invalid. Instead, use data-{}'.format(_old_idx, _idx))
            else:
                try:
                    _data_idx, _view_idx = _idx
                    _scene_point = scene_points_list[_data_idx]
                    _semantic_label = semantic_labels_list[_data_idx]
                    _borders = borders_list[_data_idx]
                    _smpidx = virtual_smpidx[_data_idx][_view_idx][0]
                    _ps, _seg, _border, _smpw = data_methods.sample(_scene_point, _semantic_label, _borders, _smpidx, _view_idx)

                except Exception as e:
                    print(e)
                    _old_data_idx, _old_view_idx = _idx
                    _data_idx = np.random.randint(len(dataset))
                    _view_idx = np.random.randint(8)
                    _idx = (_data_idx, _view_idx)
                    print('Data-{} from view-{} is invalid. Instead, use data-{} from view-{}'.format(_old_data_idx, _old_view_idx, _data_idx, _view_idx))

        return _ps, _seg, _border, _smpw

    results = joblib.Parallel(n_jobs=2)([joblib.delayed(process)(idxs[i+start_idx]) for i in range(bsize)])
        
    batch_data   = np.array([ps     for ps, _, _, _     in results])
    batch_label  = np.array([seg    for _, seg, _, _    in results])
    batch_border = np.array([border for _, _, border, _ in results])
    batch_smpw   = np.array([smpw   for _, _, _, smpw   in results])

    return batch_data, batch_label, batch_border, batch_smpw


def get_whole(dataset, data_methods, idx):
    scene_points_list = dataset["scene_points_list"]
    semantic_labels_list = dataset["semantic_labels_list"]
    borders_list = dataset["borders_list"]
    virtual_smpidx = dataset["virtual_smpidx"]

    scene_point = scene_points_list[idx].copy()
    semantic_label = semantic_labels_list[idx].copy()
    borders = borders_list[idx].copy()
    ps, seg, border, smpw = data_methods.sample(scene_point, semantic_label, borders)

    if FLAGS.whole:
        scene_point = scene_points_list[idx].copy()
        semantic_label = semantic_labels_list[idx].copy()
        borders = borders_list[idx].copy()
        ps, seg, border, smpw = data_methods.sample(scene_point, semantic_label, borders)
    else:
        data_idx, view_idx = idx
        scene_point = scene_points_list[data_idx].copy()
        semantic_label = semantic_labels_list[data_idx].copy()
        borders = borders_list[data_idx].copy()
        smpidx = virtual_smpidx[data_idx][view_idx][0].copy()
        ps, seg, border, smpw = data_methods.sample(scene_point, semantic_label, borders, smpidx, view_idx)

    return ps, seg, border, smpw


def train_one_epoch(sess, ops, train_writer, dataset):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train samples
    if FLAGS.whole:
        train_idxs = np.arange(0, len(TRAIN_DATASET))
    else:
        train_idxs = list([x for x in itertools.product(range(len(TRAIN_DATASET)), range(8))])
    np.random.shuffle(train_idxs)
    num_batches = len(train_idxs)/BATCH_SIZE

    log_string(str(datetime.now()))

    total_correct_class = 0
    total_correct_border = 0
    total_seen = 0
    loss_sum_class = 0
    loss_sum_border = 0

    for batch_idx in range(num_batches):
        print("batch: {}/{}".format(batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        s = time.time()
        print("begin get batch")
        batch_data, batch_label, batch_border, batch_smpw = get_batch_wdp(dataset, TRAIN_DATASET, train_idxs, start_idx, end_idx)
        print("end get batch", time.time() - s)
        # Augment batched point clouds by rotation
        aug_data = provider.rotate_point_cloud(batch_data)
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['borders_pl']: batch_border,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training,
                     }
        summary, step, _, loss_val_class, loss_val_border, pred_val_class, pred_val_border = sess.run(
            [ops['merged'], ops['step'], ops['train_op'], ops['class_loss'], ops['border_loss'], ops['pred_class'], ops['pred_border']],
            feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val_class = np.argmax(pred_val_class, 2)
        pred_val_border = np.round(pred_val_border)
        correct_class = np.sum(pred_val_class == batch_label)
        correct_border = np.sum(pred_val_border == batch_border)
        total_correct_class += correct_class
        total_correct_border += correct_border
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum_class += loss_val_class
        loss_sum_border += loss_val_border
        if (batch_idx+1) % 10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss (class): %f' % (loss_sum_class / 10))
            log_string('mean loss (border): %f' % (loss_sum_border / 10))
            log_string('accuracy class: %f' % (total_correct_class / float(total_seen)))
            log_string('accuracy border: %f' % (total_correct_border / float(total_seen)))
            total_correct_class = 0
            total_correct_border = 0
            total_seen = 0
            loss_sum_class = 0
            loss_sum_border = 0


#  evaluate on randomly chopped scenes
def eval_one_epoch(sess, ops, test_writer, dataset):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = len(TEST_DATASET)/BATCH_SIZE

    total_correct_cls = 0
    total_correct_border = 0
    total_seen = 0
    loss_sum_class = 0
    loss_sum_border = 0
    total_seen_class = [0 for _ in range(FLAGS.num_classes)]
    total_correct_class = [0 for _ in range(FLAGS.num_classes)]

    total_correct_vox = 0
    total_seen_vox = 0
    total_seen_class_vox = [0 for _ in range(FLAGS.num_classes)]
    total_correct_class_vox = [0 for _ in range(FLAGS.num_classes)]

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))

    labelweights = np.zeros(FLAGS.num_classes)
    labelweights_vox = np.zeros(FLAGS.num_classes)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        s = time.time()
        print("start get batch")
        batch_data, batch_label, batch_border, batch_smpw = get_batch(dataset, TEST_DATASET, test_idxs, start_idx, end_idx)
        print("end get batch", time.time() - s)
        aug_data = provider.rotate_point_cloud(batch_data)

        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['borders_pl']: batch_border,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val_class, loss_val_border, pred_val_class, pred_val_border = sess.run(
            [ops['merged'], ops['step'], ops['class_loss'], ops['border_loss'], ops['pred_class'], ops['pred_border']],
            feed_dict=feed_dict)
        test_writer.add_summary(summary, step)

        pred_val_class = np.argmax(pred_val_class, 2)  # BxN
        correct_class = np.sum((pred_val_class == batch_label) & (batch_smpw > 0))
        total_correct_cls += correct_class

        pred_val_border = np.round(pred_val_border)  # BxN
        correct_border = np.sum((pred_val_border == batch_border) & (batch_smpw[..., None] > 0))
        total_correct_border += correct_border

        total_seen += np.sum(batch_smpw > 0)
        loss_sum_class += loss_val_class
        loss_sum_border += loss_val_border
        tmp, _ = np.histogram(batch_label, range(FLAGS.num_classes+1))
        labelweights += tmp

        for l in range(FLAGS.num_classes):
            total_seen_class[l] += np.sum((batch_label == l) & (batch_smpw > 0))
            total_correct_class[l] += np.sum((pred_val_class == l) & (batch_label == l) & (batch_smpw > 0))

        for b in range(batch_label.shape[0]):
            _, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(  # Point to voxel index (a point for each voxel)
                aug_data[b, batch_smpw[b, :] > 0, :],
                np.concatenate(
                    (np.expand_dims(batch_label[b, batch_smpw[b, :] > 0], 1), np.expand_dims(pred_val_class[b, batch_smpw[b, :] > 0], 1)),
                    axis=1
                ),
                res=0.02
            )
            total_correct_vox += np.sum((uvlabel[:, 0] == uvlabel[:, 1]) & (uvlabel[:, 0] > 0))
            total_seen_vox += np.sum(uvlabel[:, 0] > 0)
            tmp, _ = np.histogram(uvlabel[:, 0], range(FLAGS.num_classes+1))
            labelweights_vox += tmp
            for l in range(FLAGS.num_classes):
                total_seen_class_vox[l] += np.sum(uvlabel[:, 0] == l)
                total_correct_class_vox[l] += np.sum((uvlabel[:, 0] == l) & (uvlabel[:, 1] == l))

    log_string('eval mean loss (class): %f' % (loss_sum_class / float(num_batches)))
    log_string('eval mean loss (border): %f' % (loss_sum_border / float(num_batches)))
    log_string('eval point accuracy vox: %f' % (total_correct_vox / float(total_seen_vox)))
    log_string('eval point avg class acc vox: %f' % (np.mean(np.array(total_correct_class_vox)/(np.array(total_seen_class_vox, dtype=np.float)+1e-6))))
    log_string('eval point accuracy (class): %f' % (float(total_correct_cls) / float(total_seen)))
    log_string('eval point accuracy (border): %f' % (float(total_correct_border) / float(total_seen)))
    log_string('eval point avg class acc: %f' % (np.mean(np.array(total_correct_class)/(np.array(total_seen_class, dtype=np.float)+1e-6))))
    labelweights_vox = labelweights_vox.astype(np.float32)/np.sum(labelweights_vox.astype(np.float32))
    caliweights = np.ones(FLAGS.num_classes)  # np.array([0.388,0.357,0.038,0.033,0.017,0.02,0.016,0.025,0.002,0.002,0.002,0.007,0.006,0.022,0.004,0.0004,0.003,0.002,0.024,0.029])
    log_string('eval point calibrated average acc: %f' % (np.average(np.array(total_correct_class)/(np.array(total_seen_class,dtype=np.float)+1e-6),weights=caliweights)))
    per_class_str = 'vox based --------'
    for l in range(FLAGS.num_classes):
        per_class_str += 'class %d weight: %f, acc: %f; ' % (l, labelweights_vox[l], total_correct_class[l]/float(total_seen_class[l]))
    log_string(per_class_str)
    EPOCH_CNT += 1
    return total_correct_cls/float(total_seen)


# evaluate on whole scenes to generate numbers provided in the paper
def eval_whole_scene_one_epoch(sess, ops, test_writer, dataset):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    num_batches = len(TEST_DATASET_WHOLE_SCENE)

    total_correct_cls = 0
    total_correct_border = 0
    total_seen = 0
    loss_sum_class = 0
    loss_sum_border = 0
    total_seen_class = [0 for _ in range(FLAGS.num_classes)]
    total_correct_class = [0 for _ in range(FLAGS.num_classes)]

    total_correct_vox = 0
    total_seen_vox = 0
    total_seen_class_vox = [0 for _ in range(FLAGS.num_classes)]
    total_correct_class_vox = [0 for _ in range(FLAGS.num_classes)]

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION WHOLE SCENE----' % (EPOCH_CNT))

    labelweights = np.zeros(FLAGS.num_classes)
    labelweights_vox = np.zeros(FLAGS.num_classes)
    is_continue_batch = False

    extra_batch_data = np.zeros((0, NUM_POINT, 3))
    extra_batch_label = np.zeros((0, NUM_POINT))
    extra_batch_border = np.zeros((0, NUM_POINT, 1))
    extra_batch_smpw = np.zeros((0, NUM_POINT))

    for batch_idx in range(num_batches):
        if not is_continue_batch:
            # batch_data, batch_label, batch_border, batch_smpw = TEST_DATASET_WHOLE_SCENE[batch_idx]
            batch_data, batch_label, batch_border, batch_smpw = get_whole(dataset, TEST_DATASET_WHOLE_SCENE, batch_idx)
            batch_data = np.concatenate((batch_data, extra_batch_data), axis=0)
            batch_label = np.concatenate((batch_label, extra_batch_label), axis=0)
            batch_border = np.concatenate((batch_border, extra_batch_border), axis=0)
            batch_smpw = np.concatenate((batch_smpw, extra_batch_smpw), axis=0)
        else:
            # batch_data_tmp, batch_label_tmp, batch_border_tmp, batch_smpw_tmp = TEST_DATASET_WHOLE_SCENE[batch_idx]
            batch_data_tmp, batch_label_tmp, batch_border_tmp, batch_smpw_tmp = get_whole(dataset, TEST_DATASET_WHOLE_SCENE, batch_idx)
            batch_data = np.concatenate((batch_data, batch_data_tmp), axis=0)
            batch_label = np.concatenate((batch_label, batch_label_tmp), axis=0)
            batch_border = np.concatenate((batch_border, batch_border_tmp), axis=0)
            batch_smpw = np.concatenate((batch_smpw, batch_smpw_tmp), axis=0)
        if batch_data.shape[0] < BATCH_SIZE:
            is_continue_batch = True
            continue
        elif batch_data.shape[0] == BATCH_SIZE:
            is_continue_batch = False
            extra_batch_data = np.zeros((0, NUM_POINT, 3))
            extra_batch_label = np.zeros((0, NUM_POINT))
            extra_batch_border = np.zeros((0, NUM_POINT, 1))
            extra_batch_smpw = np.zeros((0, NUM_POINT))
        else:
            is_continue_batch = False
            extra_batch_data = batch_data[BATCH_SIZE:, :, :]
            extra_batch_label = batch_label[BATCH_SIZE:, :]
            extra_batch_border = batch_border[BATCH_SIZE:, :, :]
            extra_batch_smpw = batch_smpw[BATCH_SIZE:, :]
            batch_data = batch_data[:BATCH_SIZE, :, :]
            batch_label = batch_label[:BATCH_SIZE, :]
            batch_border = batch_border[:BATCH_SIZE, :, :]
            batch_smpw = batch_smpw[:BATCH_SIZE, :]

        aug_data = batch_data
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['borders_pl']: batch_border,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val_class, loss_val_border, pred_val_class, pred_val_border = sess.run(
            [ops['merged'], ops['step'], ops['class_loss'], ops['border_loss'], ops['pred_class'], ops['pred_border']],
            feed_dict=feed_dict)
        test_writer.add_summary(summary, step)

        pred_val_class = np.argmax(pred_val_class, 2)  # BxN
        correct_class = np.sum((pred_val_class == batch_label) & (batch_smpw > 0))
        total_correct_cls += correct_class
        pred_val_border = np.round(pred_val_border)  # BxN
        correct_border = np.sum((pred_val_border == batch_border) & (batch_smpw[..., None] > 0))
        total_correct_border += correct_border

        total_seen += np.sum(batch_smpw > 0)
        loss_sum_class += loss_val_class
        loss_sum_border += loss_val_border
        tmp, _ = np.histogram(batch_label, range(FLAGS.num_classes+1))
        labelweights += tmp
        for l in range(FLAGS.num_classes):
            total_seen_class[l] += np.sum((batch_label == l) & (batch_smpw > 0))
            total_correct_class[l] += np.sum((pred_val_class == l) & (batch_label == l) & (batch_smpw > 0))

        for b in range(batch_label.shape[0]):
            _, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(
                aug_data[b, batch_smpw[b, :] > 0, :],
                np.concatenate(
                    (np.expand_dims(batch_label[b, batch_smpw[b, :] > 0], 1), np.expand_dims(pred_val_class[b, batch_smpw[b, :] > 0], 1)),
                    axis=1
                ),
                res=0.02)
            total_correct_vox += np.sum((uvlabel[:, 0] == uvlabel[:, 1]) & (uvlabel[:, 0] > 0))
            total_seen_vox += np.sum(uvlabel[:, 0] > 0)
            tmp, _ = np.histogram(uvlabel[:, 0], range(FLAGS.num_classes+1))
            labelweights_vox += tmp
            for l in range(FLAGS.num_classes):
                total_seen_class_vox[l] += np.sum(uvlabel[:, 0] == l)
                total_correct_class_vox[l] += np.sum((uvlabel[:, 0] == l) & (uvlabel[:, 1] == l))

    log_string('eval whole scene mean loss (class): %f' % (loss_sum_class / float(num_batches)))
    log_string('eval whole scene mean loss (border): %f' % (loss_sum_border / float(num_batches)))
    log_string('eval whole scene point accuracy vox: %f' % (total_correct_vox / float(total_seen_vox)))
    log_string('eval whole scene point avg class acc vox: %f' % (np.mean(np.array(total_correct_class_vox)/(np.array(total_seen_class_vox, dtype=np.float)+1e-6))))
    log_string('eval whole scene point accuracy (class): %f' % (total_correct_cls / float(total_seen)))
    log_string('eval whole scene point accuracy (border): %f' % (total_correct_border / float(total_seen)))
    log_string('eval whole scene point avg class acc: %f' % (np.mean(np.array(total_correct_class)/(np.array(total_seen_class, dtype=np.float)+1e-6))))
    labelweights = labelweights.astype(np.float32)/np.sum(labelweights.astype(np.float32))
    labelweights_vox = labelweights_vox.astype(np.float32)/np.sum(labelweights_vox.astype(np.float32))
    caliweights = np.ones(FLAGS.num_classes)  # np.array([0.388,0.357,0.038,0.033,0.017,0.02,0.016,0.025,0.002,0.002,0.002,0.007,0.006,0.022,0.004,0.0004,0.003,0.002,0.024,0.029])
    caliacc = np.average(np.array(total_correct_class_vox)/(np.array(total_seen_class_vox, dtype=np.float)+1e-6), weights=caliweights)
    log_string('eval whole scene point calibrated average acc vox: %f' % caliacc)

    per_class_str = 'vox based --------'
    for l in range(FLAGS.num_classes):
        per_class_str += 'class %d weight: %f, acc: %f; ' % (l, labelweights_vox[l], total_correct_class_vox[l]/float(total_seen_class_vox[l]))
    log_string(per_class_str)
    EPOCH_CNT += 1
    return caliacc


if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()
