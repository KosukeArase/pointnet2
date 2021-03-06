### ScanNet Data

Original dataset website: <a href="http://www.scan-net.org/">http://www.scan-net.org/</a>

You can get our preprocessed data at <a href="https://shapenet.cs.stanford.edu/media/scannet_data_pointnet2.zip">here (1.72GB)</a> and refer to the code in `scannet_util.py` for data loading. Note that the virtual scan data is generated on the fly from our preprocessed data.

Some code we used for scannet preprocessing is also included in `preprocessing` folder. You have to download the original ScanNet data and make small modifications in paths in order to run them.


python train.py --num_point 8192 --log_dir ../logs/20181004_8192_2000 --max_epoch 2000 --whole
python inference.py --num_point 4096 --model_path ../logs/4096/model.ckpt --whole
