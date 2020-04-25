# coding:utf-8
# 网络基本模块

import tensorflow as tf
slim = tf.contrib.slim
from src.Activation import activation as act
from src import Activation

# conv + BN + activation
def conv(inputs, out_channels, kernel_size=3, stride=1, bn=True, activation=act.LEAKY_RELU, isTrain=True):
    '''
    inputs:输入tensor
    out_channels:输出的维度
    kernel_size:卷积核大小
    stride:步长
    bn:是否使用 batch_normalization
    relu:是否使用leak_relu激活
    isTrain:是否是训练, 训练会更新 bn 的滑动平均
    return:tensor
    ...
    普通卷积:
        input : [batch, height, width, channel]
        kernel : [height, width, in_channels, out_channels]
    '''
    # 补偿边角
    if stride > 1:
        inputs = padding_fixed(inputs, kernel_size)
    
    # 这里可以自定义激活方式, 默认 relu, 可以实现空洞卷积:rate 参数
    inputs = slim.conv2d(inputs, out_channels, kernel_size, stride=stride, 
                                                padding=('SAME' if stride == 1 else 'VALID'),
                                                activation_fn=None,
                                                normalizer_fn=None)    
    if bn:
        # 如果想要提高稳定性，zero_debias_moving_mean设为True
        inputs = tf.contrib.layers.batch_norm(inputs=inputs, decay=0.9, updates_collections=None, 
                                                scale=False, is_training = isTrain)
    # 激活                                            
    inputs = Activation.activation_fn(inputs, activation)
    return inputs

# 边缘全零填充补偿卷积缺失
def padding_fixed(inputs, kernel_size):
    '''
    对tensor的周围进行全0填充
    '''
    pad_total = kernel_size - 1
    pad_start = pad_total // 2
    pad_end = pad_total - pad_start
    inputs = tf.pad(inputs, [[0,0], [pad_start, pad_end], [pad_start, pad_end], [0,0]])
    return inputs

# yolo 残差模块实现
def yolo_res_block(inputs, in_channels, res_num, isTrain=True):
    '''
    yolo的残差模块实现
    inputs:输入
    res_num:一共 res_num 个 1,3,s(残差) 模块
    '''
    # 3,1,r,1模块儿
    route = conv(inputs, in_channels*2, stride=2, activation=act.MISH, isTrain=isTrain)
    net = conv(route, in_channels, kernel_size=1, activation=act.MISH, isTrain=isTrain)
    route = conv(route, in_channels, kernel_size=1, activation=act.MISH, isTrain=isTrain)
    
    # 1,3,s模块儿
    for _ in range(res_num):
        tmp = net
        net = conv(net, in_channels, kernel_size=1, activation=act.MISH, isTrain=isTrain)
        net = conv(net, in_channels, activation=act.MISH, isTrain=isTrain)
        # 相加:shortcut 层
        net = tmp + net

    # 1,r,1模块儿
    net = conv(net, in_channels, kernel_size=1, activation=act.MISH, isTrain=isTrain)
    # 拼接:route 层
    net = tf.concat([net, route], -1)
    net = conv(net, in_channels*2, kernel_size=1, activation=act.MISH, isTrain=isTrain)
    
    return net

# 3*3 与 1*1 卷积核交错卷积实现
def yolo_conv_block(net,in_channels, a, b, isTrain=True):
    '''
    net:输入
    a:一共 a 个 1*1 与 3*3 交错卷积的模块
    b:一共 b 个 1*1 卷积模块儿
    '''
    for _ in range(a):
        out_channels = in_channels / 2
        net = conv(net, out_channels, kernel_size=1, isTrain=isTrain)
        net = conv(net, out_channels*2, isTrain=isTrain)
    
    out_channels = in_channels
    for _ in range(b):
        out_channels = out_channels / 2
        net = conv(net, out_channels, kernel_size=1, isTrain=isTrain)

    return net

# 最大池化模块
def yolo_maxpool_block(inputs):
    '''
    yolo的最大池化模块, 即 cfg 中的 SPP 模块
    inputs:[N, 19, 19, 512]
    return:[N, 19, 19, 2048]
    '''
    max_5 = tf.nn.max_pool(inputs, 5, [1,1,1,1], 'SAME')
    max_9 = tf.nn.max_pool(inputs, 9, [1,1,1,1], 'SAME')
    max_13 = tf.nn.max_pool(inputs, 13, [1,1,1,1], 'SAME')
    # 拼接
    inputs = tf.concat([max_13, max_9, max_5, inputs], -1)
    return inputs

# 上采样模块儿
def yolo_upsample_block(inputs, in_channels, route, isTrain=True):
    '''
    上采样模块儿
    inputs:主干输入
    route:以前的特征
    '''
    shape = tf.shape(inputs)
    out_height, out_width = shape[1]*2, shape[2]*2
    inputs = tf.compat.v1.image.resize_nearest_neighbor(inputs, (out_height, out_width))
    
    route = conv(route, in_channels, kernel_size=1, isTrain=isTrain)

    inputs = tf.concat([route, inputs], -1)
    return inputs

# darknet53实现
def darknet53(inputs, isTrain=True):
    '''
    inputs:[N, 416, 416, 3]
    darknet53实现
    只有52个卷积层
    '''
    # ########## 第一阶段 ############
    # 先卷积一次
    net = conv(inputs, 32, activation=act.MISH, isTrain=isTrain)

    # res1
    # [N, 608, 608, 32] => [N, 304, 304, 64]
    net = yolo_res_block(net, 32, 1, isTrain=isTrain)
    # res2
    # [N, 304, 304, 64] => [N, 152, 152, 128]
    net = yolo_res_block(net, 64, 2, isTrain=isTrain)
    # res8
    # [N, 152, 152, 128] => [N, 76, 76, 256]
    net = yolo_res_block(net, 128, 8, isTrain=isTrain)
    # 第54层特征
    # [N, 76, 76, 256]
    up_route_54 = net
    # res8
    # [N, 76, 76, 256] => [N, 38, 38, 512]
    net = yolo_res_block(net, 256, 8, isTrain=isTrain)
    # 第85层特征
    # [N, 38, 38, 512]
    up_route_85 = net
    # res4
    # [N, 38, 38, 512] => [N, 19, 19, 1024]
    net = yolo_res_block(net, 512, 4, isTrain=isTrain)
    # [N, 19, 19, 1024] => [N, 19, 19, 512]
    net = yolo_conv_block(net, 1024, 1, 1, isTrain=isTrain)
    # 池化:SPP
    # [N, 19, 19, 512] => [N, 19, 19, 2048]
    net = yolo_maxpool_block(net)
    # ########## 格外注意这里通道数由 2048 => 512
    # [N, 19, 19, 2048] => [N, 19, 19, 512]
    net = conv(net, 512, kernel_size=1, isTrain=isTrain)
    # [N, 19, 19, 512] => [N, 19, 19, 1024]
    net = conv(net, 1024, isTrain=isTrain)
    # [N, 19, 19, 1024] => [N, 19, 19, 512]
    net = yolo_conv_block(net, 1024, 0, 1, isTrain=isTrain)
    # 第116层特征, 用作最后的特征拼接
    # [N, 19, 19, 512]
    route_1 = net

    # ########## 第二阶段 ############
    # [N, 19, 19, 512] => [N, 19, 19, 256]
    net = yolo_conv_block(net, 512, 0, 1, isTrain=isTrain)
    # 上采样
    # [N, 19, 19, 256] => [N, 38, 38, 512]
    net = yolo_upsample_block(net, 256, up_route_85, isTrain=isTrain)
    # [N, 38, 38, 512] => [N, 38, 38, 256]
    net = yolo_conv_block(net, 512, 2, 1, isTrain=isTrain)
    # 第126层特征，用作最后的特征拼接
    # [N, 38, 38, 256]
    route_2 = net

    # ########## 第三阶段 ############
    # [N, 38, 38, 256] => [N, 38, 38, 128]
    net = yolo_conv_block(net, 256, 0, 1, isTrain=isTrain)
    # 上采样
    net = yolo_upsample_block(net, 128, up_route_54, isTrain=isTrain)
    net = yolo_conv_block(net, 256, 2, 1, isTrain=isTrain)
    # 第136层特征, 用作最后的特征拼接
    # [N, 76, 76, 128]
    route_3 = net

    return route_1, route_2, route_3


# YOLO 的卷积实现
def yolo_block(inputs, kernel_num, isTrain):
    '''
    yolo最后一段的卷积实现
    return:route,net, route比net少一个卷积, route用于与下一层特征进行拼接
    '''
    net = conv(inputs, kernel_num, 1, isTrain=isTrain)
    net = conv(net, kernel_num * 2, isTrain=isTrain)
    net = conv(net, kernel_num, 1, isTrain=isTrain)
    net = conv(net, kernel_num * 2, isTrain=isTrain)
    net = conv(net, kernel_num, 1, isTrain=isTrain)
    route = net
    net = conv(net, kernel_num * 2, isTrain=isTrain)
    return route, net



