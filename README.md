python -m src.train --epochs=10 --lr_cycles=0.6
python3 csv2yolo_parser.py --csv_path ../../data/dataset_YOLO --dataset_path ../../data/dataset_YOLO/YOLO_dataset/

python train.py --weights yolov7-tiny.pt --data "../data/dataset_YOLO/yolo_cones.yaml" --workers 6 --batch-size 8 --img 416 --cfg cfg/training/yolov7-tiny.yaml --name yolov7 --hyp data/hyp.scratch.p5.yaml --epochs 50
