# YOLOv4_tensorflow
yolov4的tensorflow实现. <br/>
Implement yolov4 with tensorflow<br/>
持续更新<br/>
continuous update the code<br/>
## 中科院计算所二战失败考生，跪求好老师调剂收留
277118506@qq.com<br/>
rdc01234@163.com<br/>
## 使用说明
## introductions
直接执行命令, 程序就能执行<br/>
python train.py<br/>
just run the following command<br/>
python train.py<br/>
### 训练自己的数据集
### train with own dataset
./data/JPEGImages 文件夹中存放用labelme标注json文件的jpg图片和对应的json文件, 参考我给的文件夹<br/>
The jpg image and the corresponding json file which marked with 'labelme' are stored in the folder "./data/JPEGImages", just like what I do<br/>
<br/>
然后在 ./data 文件夹下执行 python 命令, 会自动产生 label 文件和 train.txt 文件<br/>
python generate_labels.py<br/>
and then, go to the folder "./data", execute the following python command, it automatically generates label files and train.txt<br/>
python generate_labels.py<br/>
<br/>
继续执行命令,得到 anchor box<br/>
python k_means.py<br/>
excute the python command, to get anchor box<br/>
python k_means.py<br/>
<br/>
打开 config.py, 将得到的 anchor box 写入到第六行，就像这样<br/>
anchors = 12,19, 19,27, 18,37, 21,38, 23,38, 26,39, 31,38, 39,44, 67,96<br/>
open config.py, write the anchor box to line 6, just like this<br/>
<br/>
所有的配置参数都在 config.py 中，你可以按照自己的实际情况来修改<br/>
all configuration parameters are in the config.py, you can modify them according to your actual situation<br/>
<br/>
配置完成,执行命令<br/>
python train.py<br/>
ok, that's all, execute the command<br/>
python train.py<br/>
<br/>
### 有关 config.py 和训练的提示
### some tips with config.py and train the model
config.py 中的 width 和 height 应该是 608，显存不够才调整为 416 的<br/>
the parameters of width and height in config.py should be 608, but i have not a powerful GPU, that is why i set them as 416<br/>
学习率不宜设置太高<br/>
learning rate do not set too high<br/>
<br/>
### 自己的设备
### my device
GPU : 1660ti (华硕猛禽)<br/>
CPU : i5 9400f<br/>
mem : 16GB<br/>
os  : ubuntu 18.04<br/>
cuda: 10.2<br/>
cudnn : 7<br/>

