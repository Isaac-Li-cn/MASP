#! /bin/bash
python -m torch.distributed.launch --nproc_per_node 3 train.py --weights yolov5s.pt --data data/coco_c_10_2.yaml --epochs 40 --name model_10_2_all_40  --batch-size 144 --device 1,2,3


