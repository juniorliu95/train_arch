test environment：Python 2.7, tensorflow 1.6,CUDA 9

variable setting is in config.py。

( mkdir pretrain/inception_v4, 下载预训练模型, cp到pretrain/inception_v4/ ) 

run： python main.py --mode= --retrain= (flag details in main.py)


file structure:

-train_arch # codes in git
-ckpt  #.ckpts for fine-tune.detail definitions are in train.py
-dataset # include folders containing samples
	-mask
	-train
	-train_normal
	-test
	-test_normal
	-val
	-val_normal
-logs
-mask_ckpt #include things for mask acquirment
-model  #retrained files in each child folder
	-arch_inception_v4   
	-arch_resnet_v2_152  
	-inception_resnet_v2  
	-resnet_v2_152
	-arch_resnet_v2_101  
	-arch_resnet_v2_50   
	-inception_v4


only the last part of the network is trained in default. Heatmap is not so accurate for 123 subsampling.
