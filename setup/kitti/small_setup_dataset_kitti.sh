#!/bin/bash

mkdir -p 'data/kitti_raw_data'

files=(2011_09_26_calib.zip
2011_09_26_drive_0001
2011_09_26_drive_0002)

for i in ${files[@]}; do
    if [ ${i:(-3)} != "zip" ]
    then
        shortname=$i'_sync.zip'
        fullname=$i'/'$i'_sync.zip'
    else
        shortname=$i
        fullname=$i
    fi
    echo "Downloading: "$shortname
    wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'$fullname -P 'data/kitti_raw_data'
    unzip -o 'data/kitti_raw_data/'$shortname -d 'data/kitti_raw_data'
    rm 'data/kitti_raw_data/'$shortname
done

wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip -P data
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip -P data
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip -P data

mkdir -p data/kitti_depth_completion
mkdir -p data/kitti_depth_completion/train_val_split
mkdir -p data/kitti_depth_completion/train_val_split/sparse_depth
mkdir -p data/kitti_depth_completion/train_val_split/ground_truth
mkdir -p data/kitti_depth_completion/validation
mkdir -p data/kitti_depth_completion/testing
mkdir -p data/kitti_depth_completion/tmp

unzip data/data_depth_velodyne.zip -d data/kitti_depth_completion/train_val_split/sparse_depth
unzip data/data_depth_annotated.zip -d data/kitti_depth_completion/train_val_split/ground_truth
unzip data/data_depth_selection.zip -d data/kitti_depth_completion/tmp

mv data/kitti_depth_completion/tmp/depth_selection/val_selection_cropped/image data/kitti_depth_completion/validation/image
mv data/kitti_depth_completion/tmp/depth_selection/val_selection_cropped/velodyne_raw data/kitti_depth_completion/validation/sparse_depth
mv data/kitti_depth_completion/tmp/depth_selection/val_selection_cropped/groundtruth_depth data/kitti_depth_completion/validation/ground_truth
mv data/kitti_depth_completion/tmp/depth_selection/val_selection_cropped/intrinsics data/kitti_depth_completion/validation/intrinsics

mv data/kitti_depth_completion/tmp/depth_selection/test_depth_completion_anonymous/image data/kitti_depth_completion/testing/image
mv data/kitti_depth_completion/tmp/depth_selection/test_depth_completion_anonymous/velodyne_raw data/kitti_depth_completion/testing/sparse_depth
mv data/kitti_depth_completion/tmp/depth_selection/test_depth_completion_anonymous/intrinsics data/kitti_depth_completion/testing/intrinsics

rm -r data/kitti_depth_completion/tmp

python setup/setup_dataset_kitti.py