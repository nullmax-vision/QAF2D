# Data Preparation

## Dataset
**1. Download nuScenes**

Download the [nuScenes dataset](https://www.nuscenes.org/download) to `./data/nuscenes`.

## 2. Creating infos file

We modify data preparation in `MMDetection3D`, which addtionally creates 2D annotations and temporal information for training/evaluation. 
```shell
python tools/create_data_nusc.py --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes2d --version v1.0
```

Using the above code will generate `nuscenes2d_temporal_infos_{train,val}.pkl`.
We also privided the processed [train](https://github.com/exiawsh/storage/releases/download/v1.0/nuscenes2d_temporal_infos_train.pkl), [val](https://github.com/exiawsh/storage/releases/download/v1.0/nuscenes2d_temporal_infos_val.pkl) and [test](https://github.com/exiawsh/storage/releases/download/v1.0/nuscenes2d_temporal_infos_test.pkl) pkl.



* After preparation, you will be able to see the following directory structure:  

**Folder structure**

```
QAF2D
├── projects/
├── mmdetection3d/
├── tools/
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes2d_temporal_infos_train.pkl
|   |   ├── nuscenes2d_temporal_infos_val.pkl
```

In addition, you also need to rename nuscenes2d_temporal_infos_train.pkl to nuscenes2d_temporal_infos_train_stream.pkl and rename nuscenes2d_temporal_infos_val.pkl to nuscenes2d_temporal_infos_val_stream.pkl