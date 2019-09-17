cd src
# train
python python main.py ctdet --exp_id coco_res18 --arch detector_resnet18 --batch_size 11 --lr 5e-5 --gpus 0 --num_workers 5 --val_intervals 1
# test
python test.py ctdet --exp_id coco_res18 --arch detector_resnet18 --keep_res --resume
# flip test
python test.py ctdet --exp_id coco_res18 --arch detector_resnet18 --keep_res --resume --flip_test
# multi scale test
python test.py ctdet --exp_id coco_res18 --arch detector_resnet18 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
