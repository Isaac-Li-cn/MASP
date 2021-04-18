#! /bin/bash
python -m torch.distributed.launch --nproc_per_node 3 train.py --weights yolov5s.pt --data data/coco_c_9_2.yaml --epochs 60 --name model_9_2_all_layer_60  --batch-size 144 --device 1,2,3

