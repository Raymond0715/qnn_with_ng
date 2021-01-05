# Quantized Neural Networks on Classification tasks using `TensorFlow 2.3`

- Supported datasets: Cifar 10/100

- Supported Model: resnet 20 and vgg 16



## Dependencies

- `TensorFlow 2.3.0`
- `numpy 1.19.0`



## Run command

Clone and run following command. Specify `quantilize` for full and quantized NN. `ste` stand for "straight forward estimator" and `ng` stand for natural gradient introduced in "optimizing quantized neural network with natural gradient" .

```sh
python main.py \
	--model <model/name> \ # Must be one of resnet20 and vgg16
	--pretrain_path <pretrain/path, default = None> \
	--class_num <class/number> \
	--dataset <dataset> \ # Must be one of cifar10 and cifar100
	--quantilize <Choose quantization method> \ # Must be one of full, ste and ng
	--quantilize_w <weights/bits/width, e.g. 32> \ # Weights bits width for quantilize model
	--quantilize_x <activation/bits/width, e.g. 32> \ # Activation bits width for quantilize model
	--weight_decay <weight/decay, e.g. 0.0005> \
	--batch_size <batch/size, e.g. 128> \
	--num_epochs <epoch/number, e.g. 250> \
	--learning_rate <learning/rate, e.g. 0.1> \
	--log_dir <log/dir, e.g. log_dir> \
	--log_file <log/file, e.g. log_file.txt> \
	--ckpt_dir <ckpt/dir, e.g. ckpt> \
	--ckpt_file <ckpt/file, e.g. model>
```

example: 

```sh
python main.py \
	--model resnet20 \
	--pretrain_path resnet20_cifar10/full.h5 \
	--class_num 10 \
	--dataset cifar10 \
	--quantilize ste \
	--quantilize_w 1 \
	--quantilize_x 1 \
	--weight_decay 0.0005 \
	--log_dir resnet20_cifar10 \
	--log_file ste.csv \
	--ckpt_dir resnet20_cifar10 \
	--ckpt_file full.h5
```

Besides, you can run `python main.py -h` for help. 



## Tools

In 'tools' directory, you can run following command to plot loss and accuracy curve recorded in log file.

```sh
python curve.py
```

You need to change log file path and y limits in `curve.py` for different log file. 