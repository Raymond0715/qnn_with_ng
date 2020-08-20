# Run command

```sh
python main.py \
	--model <model/name> \
	--class_num <class/number> \
	--dataset <dataset> \
	--quantilize <Choose quantization method> \ # Will be string in near future
	--quantilize_w <weights/bits/width, e.g. 32> \
	--quantilize_x <activation/bits/width, e.g. 32> \
	--weight_decay <weight/decay, e.g. 0.0005> \
	--batch_size <batch/size, e.g. 128> \
	--num_epochs <epoch/number, e.g. 250> \
	--learning_rate <learning/rate, e.g. 0.1> \
	--ckpt_dir <ckpt/dir, e.g. resnet20> \
	--log_dir <log/dir, e.g. log_dir> \
	--log_file <log/file, e.g. log_file.txt>

# e.g.
python main.py --model resnet20 --class_num 10 --dataset cifar10 --quantilize ste --quantilize_w 1 --quantilize_x 1 --weight_decay 0 --log_dir resnet20_cifar10 --log_file ng.csv

python main.py --model vgg16 --class_num 100 --dataset cifar100 --quantilize full --quantilize_w 1 --quantilize_x 1 --weight_decay 0.0005 --log_dir vgg16_cifar100 --log_file full.csv
```

