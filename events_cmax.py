# 导入所需的库
import argparse  # 用于解析命令行参数
import time  # 用于时间相关操作
import numpy as np  # 用于数值计算
import scipy  # 科学计算库
import scipy.optimize as opt  # 优化算法
from scipy.ndimage.filters import gaussian_filter  # 高斯滤波
import torch  # PyTorch深度学习框架
from event_utils import *  # 事件处理工具函数
from objectives import *  # 目标函数
from warps import *  # 变形函数

def draw_objective_function(xs, ys, ts, ps, objective, warpfunc, x_range=(-200, 200), y_range=(-200, 200),
        gt=(0,0), show_gt=True, resolution=20, img_size=(180, 240)):
    """
    通过在给定范围内采样来绘制目标函数。根据分辨率的值,这可能涉及许多采样并需要一些时间。
    参数:
        xs,ys,ts,ps (numpy array) 事件的各个分量
        objective (object) 目标函数
        warpfunc (object) 变形函数
        x_range, y_range (tuple) 绘制参数的范围
        gt (tuple) 真实值
        show_gt (bool) 是否显示真实值
        resolution (float) 采样分辨率
        img_size (tuple) 图像传感器大小
    """
    width = x_range[1]-x_range[0]
    height = y_range[1]-y_range[0]
    print("Drawing objective function. Taking {} samples".format((width*height)/resolution))
    imshape = (int(height/resolution+0.5), int(width/resolution+0.5))
    img = np.zeros(imshape)
    for x in range(img.shape[1]):
       for y in range(img.shape[0]):
           params = np.array([x*resolution+x_range[0], y*resolution+y_range[0]])
           img[y,x] = -objective.evaluate_function(params, xs, ys, ts, ps, warpfunc, img_size, blur_sigma=0)
    img = cv.normalize(img, None, 0, 1.0, cv.NORM_MINMAX)
    plt.imshow(img, interpolation='bilinear', cmap='viridis')
    plt.xticks([])
    plt.yticks([])
    if show_gt:
        xloc = ((gt[0]-x_range[0])/(width))*imshape[1]
        yloc = ((gt[1]-y_range[0])/(height))*imshape[0]
        plt.axhline(y=yloc, color='r', linestyle='--')
        plt.axvline(x=xloc, color='r', linestyle='--')
    plt.show()

def optimize_contrast(xs, ys, ts, ps, warp_function, objective, optimizer=opt.fmin_bfgs, x0=None,
        numeric_grads=False, blur_sigma=None, img_size=(180, 240)):
    """
    优化一组事件的对比度
    参数:
    xs (numpy float array) 事件的x坐标分量
    ys (numpy float array) 事件的y坐标分量
    ts (numpy float array) 事件的时间戳。应该是ts-t[0]以避免精度问题
    ps (numpy float array) 事件的极性
    warp_function (function) 用于变形事件的函数
    objective (objective class object) 要优化的目标
    optimizer (function) 使用的优化器
    x0 (np array) 优化的初始猜测
    numeric_grads (bool) 如果为true,使用数值导数,否则使用解析导数(如果可用)
    img_size (tuple) 事件相机传感器的大小
    blur_sigma (float) 模糊核的大小。模糊变形事件的图像可能对优化的收敛有很大影响

    返回:
        相对于目标的变形参数的最大参数
    """
    args = (xs, ys, ts, ps, warp_function, img_size, blur_sigma)
    x0 = np.array([0,0])
    if x0 is None:
        x0 = np.zeros(warp_function.dims)
    if numeric_grads:
        argmax = optimizer(objective.evaluate_function, x0, args=args, epsilon=1, disp=False)
    else:
        argmax = optimizer(objective.evaluate_function, x0, fprime=objective.evaluate_gradient, args=args, disp=False)
    return argmax

def optimize(xs, ys, ts, ps, warp, obj, numeric_grads=True, img_size=(180, 240)):
    """
    优化一组事件的对比度。使用optimize_contrast()进行优化,但允许连续优化迭代的模糊调度。
    参数:
    xs (numpy float array) 事件的x坐标分量
    ys (numpy float array) 事件的y坐标分量
    ts (numpy float array) 事件的时间戳。应该是ts-t[0]以避免精度问题
    ps (numpy float array) 事件的极性
    warp (function) 用于变形事件的函数
    obj (objective class object) 要优化的目标
    numeric_grads (bool) 如果为true,使用数值导数,否则使用解析导数(如果可用)
    img_size (tuple) 事件相机传感器的大小

    返回:
        相对于目标的变形参数的最大参数
    """
    numeric_grads = numeric_grads if obj.has_derivative else True
    argmax_an = optimize_contrast(xs, ys, ts, ps, warp, obj, numeric_grads=numeric_grads, blur_sigma=blur, img_size=img_size)
    return argmax_an

def optimize_r2(xs, ys, ts, ps, warp, obj, numeric_grads=True, img_size=(180, 240)):
    """
    优化一组事件的对比度,以SoE损失结束。
    参数:
    xs (numpy float array) 事件的x坐标分量
    ys (numpy float array) 事件的y坐标分量
    ts (numpy float array) 事件的时间戳。应该是ts-t[0]以避免精度问题
    ps (numpy float array) 事件的极性
    warp (function) 用于变形事件的函数
    obj (objective class object) 要优化的目标
    numeric_grads (bool) 如果为true,使用数值导数,否则使用解析导数(如果可用)
    img_size (tuple) 事件相机传感器的大小

    返回:
        相对于目标的变形参数的最大参数
    """
    soe_obj = soe_objective()
    numeric_grads = numeric_grads if obj.has_derivative else True
    argmax_an = optimize_contrast(xs, ys, ts, ps, warp, obj, numeric_grads=numeric_grads, blur_sigma=None)
    argmax_an = optimize_contrast(xs, ys, ts, ps, warp, soe_obj, x0=argmax_an, numeric_grads=numeric_grads, blur_sigma=1.0)
    return argmax_an

if __name__ == "__main__":
    """
    各种目标的快速演示。
    参数:
        path 包含事件数据的h5文件的路径
        gt 事件切片的真实光流
        img_size 事件相机传感器的大小
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",default="D:/vitrua/events_contrast_maximization-master/circle_events_only.h5", help="h5 events path")
    #parser.add_argument("--path", default="D:/vitrua/dvs/test.h5",help="HDF5 file to extract")
    parser.add_argument("--gt", nargs='+', type=float, default=(0,0))
    parser.add_argument("--img_size", nargs='+', type=float, default=(180,240))
    args = parser.parse_args()

    # 读取事件数据
    xs, ys, ts, ps = read_h5_event_components(args.path)
    ts = ts-ts[0]
    gt_params = tuple(args.gt)
    img_size=tuple(args.img_size)

    # 设置事件处理范围
    start_idx = 20000
    end_idx=start_idx+15000
    blur = None

    # 绘制目标函数
    draw_objective_function(xs[start_idx:end_idx], ys[start_idx:end_idx], ts[start_idx:end_idx], ps[start_idx:end_idx], variance_objective(), linvel_warp())

    # 测试不同的目标函数
    objectives = [r1_objective(), zhu_timestamp_objective(), variance_objective(), sos_objective(), soe_objective(), moa_objective(),
            isoa_objective(), sosa_objective(), rms_objective()]
    warp = linvel_warp()
    for obj in objectives:
        # 使用数值梯度优化
        argmax = optimize(xs[start_idx:end_idx], ys[start_idx:end_idx], ts[start_idx:end_idx], ps[start_idx:end_idx], warp, obj, numeric_grads=True)
        loss = obj.evaluate_function(argmax, xs[start_idx:end_idx], ys[start_idx:end_idx], ts[start_idx:end_idx],
                ps[start_idx:end_idx], warp, img_size=img_size)
        gtloss = obj.evaluate_function(gt_params, xs[start_idx:end_idx], ys[start_idx:end_idx],
                ts[start_idx:end_idx], ps[start_idx:end_idx], warp, img_size=img_size)
        print("{}:({})={}, gt={}".format(obj.name, argmax, loss, gtloss))
        # 如果目标函数有导数,则使用解析梯度优化
        if obj.has_derivative:
            argmax = optimize(xs[start_idx:end_idx], ys[start_idx:end_idx], ts[start_idx:end_idx],
                    ps[start_idx:end_idx], warp, obj, numeric_grads=False)
            loss_an = obj.evaluate_function(argmax, xs[start_idx:end_idx], ys[start_idx:end_idx],
                    ts[start_idx:end_idx], ps[start_idx:end_idx], warp, img_size=img_size)
            print("   analytical:{}={}".format(argmax, loss_an))
