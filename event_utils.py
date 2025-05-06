# 导入所需的库
import argparse  # 用于解析命令行参数
import time  # 用于时间相关操作
import h5py  # 用于读取HDF5文件
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图
import cv2 as cv  # 用于图像处理
import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch函数库

def binary_search_h5_dset(dset, x, l=None, r=None, side='left'):
    """
    在预加载的HDF5数据集中进行二分查找(内存高效,因为整个数据集不会加载到RAM中)
    参数:
        dset: HDF5数据集
        x: 要查找的值
        l: 左边界索引
        r: 右边界索引
        side: 'left'或'right',表示在有重复值时返回最左或最右的索引
    返回:
        找到的索引位置
    """
    l = 0 if l is None else l
    r = len(dset)-1 if r is None else r
    while l <= r:
        mid = l + (r - l)//2;
        midval = dset[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r

def binary_search_h5_timestamp(hdf_path, l, r, x, side='left'):
    """
    在HDF5事件文件中查找时间戳的索引
    参数:
        hdf_path: HDF5文件路径
        l: 左边界索引
        r: 右边界索引
        x: 要查找的时间戳
        side: 'left'或'right'
    返回:
        时间戳的索引
    """
    f = h5py.File(hdf_path, 'r')
    return binary_search_h5_dset(f['events/ts'], x, l=l, r=r, side=side)

def binary_search_torch_tensor(t, l, r, x, side='left'):
    """
    在PyTorch张量中进行二分查找
    参数:
        t: PyTorch张量
        l: 左边界索引
        r: 右边界索引
        x: 要查找的值
        side: 'left'或'right'
    返回:
        找到的索引位置
    """
    if r is None:
        r = len(t)-1
    while l <= r:
        mid = l + (r - l)//2;
        midval = t[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r

def read_h5_events(hdf_path):
    """
    从HDF5文件读取事件到4xN的numpy数组(N=事件数量)
    参数:
        hdf_path: HDF5文件路径
    返回:
        包含事件的numpy数组
    """
    f = h5py.File(hdf_path, 'r')
    if 'events/x' in f:
        #legacy
        events = np.stack((f['events/x'][:], f['events/y'][:], f['events/ts'][:], np.where(f['events/p'][:], 1, -1)), axis=1)
    else:
        events = np.stack((f['events/xs'][:], f['events/ys'][:], f['events/ts'][:], np.where(f['events/ps'][:], 1, -1)), axis=1)
    return events

def read_h5_event_components(hdf_path):
    """
    从HDF5文件读取事件。返回x,y,t,p分量。
    参数:
        hdf_path: HDF5文件路径
    返回:
        事件的x,y,t,p分量
    """
    f = h5py.File(hdf_path, 'r')
    if 'events/x' in f:
        #legacy
        return (f['events/x'][:], f['events/y'][:], f['events/ts'][:], np.where(f['events/p'][:], 1, -1))
    else:
        return (f['events/xs'][:], f['events/ys'][:], f['events/ts'][:], np.where(f['events/ps'][:], 1, -1))

def plot_image(image, lognorm=False, cmap='gray'):
    """
    在matplotlib窗口中显示图像
    参数:
        image: 要显示的图像
        lognorm: 是否对图像取对数
        cmap: 颜色映射
    """
    if lognorm:
        image = np.log10(image)
        cmap='viridis'
    image = cv.normalize(image, None, 0, 1.0, cv.NORM_MINMAX)
    plt.imshow(image, cmap=cmap)
    plt.show()

def save_image(image, lognorm=False, cmap='gray', save_path="/tmp/img.jpg"):
    """
    保存图像
    参数:
        image: 要保存的图像
        lognorm: 是否对图像取对数
        cmap: 颜色映射
        save_path: 保存路径
    """
    if lognorm:
        image = np.log10(image)
        cmap='viridis'
    image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
    cv.imwrite(save_path, image)

def get_hot_event_mask(xs, ys, ps, sensor_size, num_hot = 100):
    """
    给定一组事件,生成热像素掩码
    参数:
        xs: 事件x坐标的张量
        ys: 事件y坐标的张量
        ps: 事件极性/权重的张量
        sensor_size: 事件传感器的大小
        num_hot: 移除前num_hot个热像素
    返回:
        掩码,好像素为1,热像素为0
    """
    img = events_to_image(xs, ys, ps, sensor_size)
    mask = np.ones_like(img)
    for idx in range(num_hot):
        idx = np.unravel_index(np.argmax(img), img.shape)
        mask[idx] = 0
        img[idx] = 0
    return mask

def get_voxel_grid_as_image(voxelgrid):
    """
    生成体素网格的并排bin图像
    参数:
        voxelgrid: 体素网格
    返回:
        并排显示的图像
    """
    images = []
    splitter = np.ones((voxelgrid.shape[1], 2))*np.max(voxelgrid)
    for image in voxelgrid:
        images.append(image)
        images.append(splitter)
    images.pop()
    sidebyside = np.hstack(images)
    sidebyside = cv.normalize(sidebyside, None, 0, 255, cv.NORM_MINMAX)
    return sidebyside

def plot_voxel_grid(voxelgrid, cmap='gray'):
    """
    显示体素网格的可视化(每个bin作为一幅图像)
    参数:
        voxelgrid: 体素网格
        cmap: 颜色映射
    """
    sidebyside = get_voxel_grid_as_image(voxelgrid)
    plt.imshow(sidebyside, cmap=cmap)
    plt.show()

def events_bounds_mask(xs, ys, x_min, x_max, y_min, y_max):
    """
    获取在给定边界内的事件的掩码
    参数:
        xs: 事件x坐标
        ys: 事件y坐标
        x_min,x_max: x坐标范围
        y_min,y_max: y坐标范围
    返回:
        边界内事件的掩码
    """
    mask = np.where(np.logical_or(xs<=x_min, xs>x_max), 0.0, 1.0)
    mask *= np.where(np.logical_or(ys<=y_min, ys>y_max), 0.0, 1.0)
    return mask

def clip_events_to_bounds(xs, ys, ps, bounds):
    """
    将事件裁剪到给定边界
    参数:
        xs: 事件x坐标
        ys: 事件y坐标
        ps: 事件极性
        bounds: 边界
    返回:
        裁剪后的事件坐标和极性
    """
    mask = events_bounds_mask(xs, ys, 0, bounds[1], 0, bounds[0])
    return xs*mask, ys*mask, ps*mask

def events_to_image(xs, ys, ps, sensor_size=(180, 240), interpolation=None, padding=False):
    """
    使用numpy将事件放入图像中
    参数:
        xs: 事件x坐标
        ys: 事件y坐标
        ps: 事件极性
        sensor_size: 传感器大小
        interpolation: 插值方法
        padding: 是否填充
    返回:
        事件图像
    """
    img_size = sensor_size
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        xt, yt, pt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ps)
        xt, yt, pt = xt.float(), yt.float(), pt.float()
        img = events_to_image_torch(xt, yt, pt, clip_out_of_range=True, interpolation='bilinear', padding=padding)
        img = img.numpy()
    else:
        coords = np.stack((ys, xs))
        try:
            abs_coords = np.ravel_multi_index(coords, sensor_size)
        except ValueError:
            print("Issue with input arrays! coords.shape={}, sum(coords)={}, sensor_size={}. \n minx={}, maxx={}, miny={}, maxy={}".format(coords.shape, np.sum(coords), sensor_size, np.min(xs), np.max(xs), np.min(ys), np.max(ys)))
            raise ValueError
        img = np.bincount(abs_coords, weights=ps, minlength=sensor_size[0]*sensor_size[1])
    img = img.reshape(sensor_size)
    return img

def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
    """
    使用双线性插值将x和y坐标累积到图像中
    参数:
        pxs: x坐标的整数部分
        pys: y坐标的整数部分
        dxs: x坐标的小数部分
        dys: y坐标的小数部分
        weights: 权重
        img: 输出图像
    返回:
        累积后的图像
    """
    img.index_put_((pys,   pxs  ), weights*(1.0-dxs)*(1.0-dys), accumulate=True)
    img.index_put_((pys,   pxs+1), weights*dxs*(1.0-dys), accumulate=True)
    img.index_put_((pys+1, pxs  ), weights*(1.0-dxs)*dys, accumulate=True)
    img.index_put_((pys+1, pxs+1), weights*dxs*dys, accumulate=True)
    return img

def interpolate_to_derivative_img(pxs, pys, dxs, dys, d_img, w1, w2):
    """
    使用双权重双线性插值将x和y坐标累积到图像中
    参数:
        pxs: x坐标的整数部分
        pys: y坐标的整数部分
        dxs: x坐标的小数部分
        dys: y坐标的小数部分
        d_img: 导数图像
        w1,w2: 权重
    """
    for i in range(d_img.shape[0]):
        d_img[i].index_put_((pys,   pxs  ), w1[i] * (-(1.0-dys)) + w2[i] * (-(1.0-dxs)), accumulate=True)
        d_img[i].index_put_((pys,   pxs+1), w1[i] * (1.0-dys)    + w2[i] * (-dxs), accumulate=True)
        d_img[i].index_put_((pys+1, pxs  ), w1[i] * (-dys)       + w2[i] * (1.0-dxs), accumulate=True)
        d_img[i].index_put_((pys+1, pxs+1), w1[i] * dys          + w2[i] *  dxs, accumulate=True)

def events_to_image_drv(xn, yn, pn, jacobian_xn, jacobian_yn,
        device=None, sensor_size=(180, 240), clip_out_of_range=True,
        interpolation=None, padding=True, compute_gradient=False):
    """
    将事件张量转换为图像的方法。允许双线性插值。
    参数:
        xn: 事件x坐标
        yn: 事件y坐标
        pn: 事件极性/权重
        jacobian_xn: x方向的雅可比矩阵
        jacobian_yn: y方向的雅可比矩阵
        device: 图像所在的设备
        sensor_size: 传感器/输出图像的大小
        clip_out_of_range: 如果事件超出所需图像大小,是否裁剪事件以适应图像
        interpolation: 使用哪种插值。选项=None,'bilinear'
        padding: 如果是双线性插值,是否允许填充图像以允许事件适应
        compute_gradient: 是否计算梯度
    返回:
        事件图像和导数图像(如果compute_gradient=True)
    """
    xt, yt, pt = torch.from_numpy(xn), torch.from_numpy(yn), torch.from_numpy(pn)
    xs, ys, ps, = xt.float(), yt.float(), pt.float()
    if compute_gradient:
        jacobian_x, jacobian_y = torch.from_numpy(jacobian_xn), torch.from_numpy(jacobian_yn)
        jacobian_x, jacobian_y = jacobian_x.float(), jacobian_y.float()
    if device is None:
        device = xs.device
    if padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = sensor_size

    mask = torch.ones(xs.size())
    if clip_out_of_range:
        zero_v = torch.tensor([0.])
        ones_v = torch.tensor([1.])
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    pxs = xs.floor()
    pys = ys.floor()
    dxs = xs-pxs
    dys = ys-pys
    pxs = (pxs*mask).long()
    pys = (pys*mask).long()
    masked_ps = ps*mask
    img = torch.zeros(img_size).to(device)
    interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)

    if compute_gradient:
        d_img = torch.zeros((2, *img_size)).to(device)
        w1 = jacobian_x*masked_ps
        w2 = jacobian_y*masked_ps
        interpolate_to_derivative_img(pxs, pys, dxs, dys, d_img, w1, w2)
        d_img = d_img.numpy()
    else:
        d_img = None
    return img.numpy(), d_img

def events_to_timestamp_image(xn, yn, ts, pn,
        device=None, sensor_size=(180, 240), clip_out_of_range=True,
        interpolation='bilinear', padding=True):
    """
    生成来自'Zhu19,无监督事件学习光流、深度和自运动'的平均时间戳图像的方法。
    此方法没有已知的导数。
    参数:
        xn: 事件x坐标列表
        yn: 事件y坐标列表
        ts: 事件时间戳列表
        pn: 事件极性列表
        device: 事件所在的设备
        sensor_size: 事件传感器/输出体素的大小
        clip_out_of_range: 如果事件超出所需图像大小,是否裁剪事件以适应图像
        interpolation: 使用哪种插值。选项=None,'bilinear'
        padding: 如果是双线性插值,是否允许填充图像以允许事件适应
    返回:
        img_pos: 正事件的时间戳图像
        img_neg: 负事件的时间戳图像
    """
    
    xt, yt, ts, pt = torch.from_numpy(xn), torch.from_numpy(yn), torch.from_numpy(ts), torch.from_numpy(pn)
    xs, ys, ts, ps = xt.float(), yt.float(), ts.float(), pt.float()
    zero_v = torch.tensor([0.])
    ones_v = torch.tensor([1.])
    if device is None:
        device = xs.device
    if padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = sensor_size

    mask = torch.ones(xs.size())
    if clip_out_of_range:
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    pos_events_mask = torch.where(ps>0, ones_v, zero_v)
    neg_events_mask = torch.where(ps<=0, ones_v, zero_v)
    normalized_ts = (ts-ts[0])/(ts[-1]+1e-6)
    pxs = xs.floor()
    pys = ys.floor()
    dxs = xs-pxs
    dys = ys-pys
    pxs = (pxs*mask).long()
    pys = (pys*mask).long()
    masked_ps = ps*mask

    pos_weights = normalized_ts*pos_events_mask
    neg_weights = normalized_ts*neg_events_mask
    img_pos = torch.zeros(img_size).to(device)
    img_pos_cnt = torch.ones(img_size).to(device)
    img_neg = torch.zeros(img_size).to(device)
    img_neg_cnt = torch.ones(img_size).to(device)

    interpolate_to_image(pxs, pys, dxs, dys, pos_weights, img_pos)
    interpolate_to_image(pxs, pys, dxs, dys, pos_events_mask, img_pos_cnt)
    interpolate_to_image(pxs, pys, dxs, dys, neg_weights, img_neg)
    interpolate_to_image(pxs, pys, dxs, dys, neg_events_mask, img_neg_cnt)

    img_pos, img_pos_cnt = img_pos.numpy(), img_pos_cnt.numpy()
    img_pos_cnt[img_pos_cnt==0] = 1
    img_neg, img_neg_cnt = img_neg.numpy(), img_neg_cnt.numpy()
    img_neg_cnt[img_neg_cnt==0] = 1
    return img_pos, img_neg #/img_pos_cnt, img_neg/img_neg_cnt

def events_to_zhu_timestamp_image(xn, yn, ts, pn,
        device=None, sensor_size=(180, 240), clip_out_of_range=True,
        interpolation='bilinear', padding=True):
    """
    遗留函数,使用events_to_timestamp_image代替
    """
    return events_to_timestamp_image(xn, yn, ts, pn, device=device, sensor_size=sensor_size,
            clip_out_of_range=clip_out_of_range, interpolation=interpolation)

def events_to_image_torch(xs, ys, ps,
        device=None, sensor_size=(180, 240), clip_out_of_range=True,
        interpolation=None, padding=True):
    """
    将事件张量转换为图像的方法。允许双线性插值。
    参数:
        xs: 事件x坐标张量
        ys: 事件y坐标张量
        ps: 事件极性/权重张量
        device: 图像所在的设备。如果为None,设置为事件所在的设备
        sensor_size: 图像传感器/输出图像的大小
        clip_out_of_range: 如果事件超出所需图像大小,是否裁剪事件以适应图像
        interpolation: 使用哪种插值。选项=None,'bilinear'
        padding: 如果是双线性插值,是否允许填充图像以允许事件适应
    返回:
        事件图像
    """
    if device is None:
        device = xs.device
    if interpolation == 'bilinear' and padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = list(sensor_size)

    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        zero_v = torch.tensor([0.], device=device)
        ones_v = torch.tensor([1.], device=device)
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    img = torch.zeros(img_size).to(device)
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        pxs = (xs.floor()).float()
        pys = (ys.floor()).float()
        dxs = (xs-pxs).float()
        dys = (ys-pys).float()
        pxs = (pxs*mask).long()
        pys = (pys*mask).long()
        masked_ps = ps.squeeze()*mask
        interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)
    else:
        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)
        img.index_put_((ys, xs), ps, accumulate=True)
    return img

def voxel_grids_fixed_n_torch(xs, ys, ts, ps, B, n, sensor_size=(180, 240), temporal_bilinear=True):
    """
    给定一组事件,返回具有固定事件数的体素网格列表。
    参数:
        xs: 事件x坐标列表(torch张量)
        ys: 事件y坐标列表(torch张量)
        ts: 事件时间戳列表(torch张量)
        ps: 事件极性列表(torch张量)
        B: 输出体素网格中的bin数(int)
        n: 每个体素的事件数
        sensor_size: 事件传感器/输出体素的大小
        temporal_bilinear: 事件是否应该简单地累积到体素中(更快),
                          或者适当地在时间上分布
    返回:
        voxels: 输出体素网格列表
    """
    voxels = []
    for idx in range(0, len(xs)-n, n):
        voxels.append(events_to_voxel_torch(xs[idx:idx+n], ys[idx:idx+n],
            ts[idx:idx+n], ps[idx:idx+n], B, sensor_size=sensor_size,
            temporal_bilinear=temporal_bilinear))
    return voxels

def voxel_grids_fixed_t_torch(xs, ys, ts, ps, B, t, sensor_size=(180, 240), temporal_bilinear=True):
    """
    给定一组事件,返回具有固定时间宽度的体素网格列表。
    参数:
        xs: 事件x坐标列表(torch张量)
        ys: 事件y坐标列表(torch张量)
        ts: 事件时间戳列表(torch张量)
        ps: 事件极性列表(torch张量)
        B: 输出体素网格中的bin数(int)
        t: 体素网格的时间宽度
        sensor_size: 事件传感器/输出体素的大小
        temporal_bilinear: 事件是否应该简单地累积到体素中(更快),
                          或者适当地在时间上分布
    返回:
        voxels: 输出体素网格列表
    """
    device = xs.device
    voxels = []
    np_ts = ts.cpu().numpy()
    for t_start in np.arange(ts[0].item(), ts[-1].item()-t, t):
        voxels.append(events_to_voxel_timesync_torch(xs, ys, ts, ps, B, t_start, t_start+t, np_ts=np_ts,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear))
    return voxels

def events_to_voxel_timesync_torch(xs, ys, ts, ps, B, t0, t1, device=None, np_ts=None, sensor_size=(180, 240), temporal_bilinear=True):
    """
    给定一组事件,返回t0和t1之间事件的体素网格
    参数:
        xs: 事件x坐标列表(torch张量)
        ys: 事件y坐标列表(torch张量)
        ts: 事件时间戳列表(torch张量)
        ps: 事件极性列表(torch张量)
        B: 输出体素网格中的bin数(int)
        t0: 体素网格的开始时间
        t1: 体素网格的结束时间
        device: 放置体素网格的设备。如果为空,与事件相同的设备
        np_ts: ts的numpy副本(可选)。如果未给出,将在原地创建
        sensor_size: 事件传感器/输出体素的大小
        temporal_bilinear: 事件是否应该简单地累积到体素中(更快),
                          或者适当地在时间上分布
    返回:
        voxel: t0和t1之间事件的体素
    """
    assert(t1>t0)
    if np_ts is None:
        np_ts = ts.cpu().numpy()
    if device is None:
        device = xs.device
    start_idx = np.searchsorted(np_ts, t0)
    end_idx = np.searchsorted(np_ts, t1)
    assert(start_idx < end_idx)
    voxel = events_to_voxel_torch(xs[start_idx:end_idx], ys[start_idx:end_idx],
        ts[start_idx:end_idx], ps[start_idx:end_idx], B, device, sensor_size=sensor_size,
        temporal_bilinear=temporal_bilinear)
    return voxel

def events_to_voxel_torch(xs, ys, ts, ps, B, device=None, sensor_size=(180, 240), temporal_bilinear=True):
    """
    将一组事件转换为体素网格张量,使用时间双线性插值
    参数:
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    Parameters
    ----------
    xs : list of event x coordinates (torch tensor)
    ys : list of event y coordinates (torch tensor)
    ts : list of event timestamps (torch tensor)
    ps : list of event polarities (torch tensor)
    B : number of bins in output voxel grids (int)
    device : device to put voxel grid. If left empty, same device as events
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxel: voxel of the events between t0 and t1
    """
    if device is None:
        device = xs.device
    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    bins = []
    dt = ts[-1]-ts[0]
    t_norm = (ts-ts[0])/dt*(B-1)
    zeros = torch.zeros(t_norm.size())
    for bi in range(B):
        if temporal_bilinear:
            bilinear_weights = torch.max(zeros, 1.0-torch.abs(t_norm-bi))
            weights = ps*bilinear_weights
            vb = events_to_image_torch(xs, ys,
                    weights, device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        else:
            tstart = t[0] + dt*bi
            tend = tstart + dt
            beg = binary_search_torch_tensor(t, 0, len(ts)-1, tstart) 
            end = binary_search_torch_tensor(t, 0, len(ts)-1, tend) 
            vb = events_to_image_torch(xs[beg:end], ys[beg:end],
                    ps[beg:end], device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        bins.append(vb)
    bins = torch.stack(bins)
    return bins

def events_to_neg_pos_voxel_torch(xs, ys, ts, ps, B, device=None,
        sensor_size=(180, 240), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation.
    Positive and negative events are put into separate voxel grids
    Parameters
    ----------
    xs : list of event x coordinates 
    ys : list of event y coordinates 
    ts : list of event timestamps 
    ps : list of event polarities 
    B : number of bins in output voxel grids (int)
    device : the device that the events are on
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxel_pos: voxel of the positive events
    voxel_neg: voxel of the negative events
    """
    zero_v = torch.tensor([0.])
    ones_v = torch.tensor([1.])
    pos_weights = torch.where(ps>0, ones_v, zero_v)
    neg_weights = torch.where(ps<=0, ones_v, zero_v)

    voxel_pos = events_to_voxel_torch(xs, ys, ts, pos_weights, B, device=device,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)
    voxel_neg = events_to_voxel_torch(xs, ys, ts, neg_weights, B, device=device,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)

    return voxel_pos, voxel_neg

def unpack_batched_events(events, batch_indices):
    """
    When returning events from a pytorch dataloader, it is often convenient when
    batching, to place them into a contiguous 1x1xNx4 array, where N=length of all
    B event arrays in the batch. This function unpacks the events into a Bx1xMx4 array,
    where B is the batch size, M is the length of the *longest* event array in the
    batch. The shorter event arrays are then padded with zeros.
    Parameters
    ----------
    events : 1x1xNx4 array of the events
    batch_indices : A list of the end indices of events, where one event array ends and
    the next begins. For example, if you batched two event arrays A and B of length
    200 and 700 respectively, batch_indices=[200, 900]
    Returns
    -------
    unpacked_events: Bx1xMx4 array of unpacked events
    """
    maxlen = 0
    start_idx = 0
    for b_idx in range(len(batch_indices)):
        end_idx = event_batch_indices[b_idx]
        maxlen = end_idx-start_idx if end_idx-start_dx > maxlen else maxlen

    unpacked_events = torch.zeros((len(batch_indices), 1, maxlen, 4))
    start_idx = 0
    for b_idx in range(len(batch_indices)):
        num_events = end_idx-start_idx
        unpacked_events[b_idx, 0, 0:num_events, :] = events[start_idx:end_idx, :]
        start_idx = end_idx
    return unpacked_events

def warp_events_flow_torch(xt, yt, tt, pt, flow_field, t0=None,
        batched=False, batch_indices=None):
    """
    Given events and a flow field, warp the events by the flow
    Parameters
    ----------
    xs : list of event x coordinates 
    ys : list of event y coordinates 
    ts : list of event timestamps 
    ps : list of event polarities 
    flow_field : 2D tensor containing the flow at each x,y position
    t0 : the reference time to warp events to. If empty, will use the
        timestamp of the last event
    Returns
    -------
    warped_xt: x coords of warped events
    warped_yt: y coords of warped events
    """
    if len(xt.shape) > 1:
        xt, yt, tt, pt = xt.squeeze(), yt.squeeze(), tt.squeeze(), pt.squeeze()
    if t0 is None:
        t0 = tt[-1]
    while len(flow_field.size()) < 4:
        flow_field = flow_field.unsqueeze(0)
    if len(xt.size()) == 1:
        event_indices = torch.transpose(torch.stack((xt, yt), dim=0), 0, 1)
    else:
        event_indices = torch.transpose(torch.cat((xt, yt), dim=1), 0, 1)
    #event_indices.requires_grad_ = False
    event_indices = torch.reshape(event_indices, [1, 1, len(xt), 2])

    # Event indices need to be between -1 and 1 for F.gridsample
    event_indices[:,:,:,0] = event_indices[:,:,:,0]/(flow_field.shape[-1]-1)*2.0-1.0
    event_indices[:,:,:,1] = event_indices[:,:,:,1]/(flow_field.shape[-2]-1)*2.0-1.0

    flow_at_event = F.grid_sample(flow_field, event_indices, align_corners=True) 
    
    dt = (tt-t0).squeeze()

    warped_xt = xt+flow_at_event[:,0,:,:].squeeze()*dt
    warped_yt = yt+flow_at_event[:,1,:,:].squeeze()*dt

    return warped_xt, warped_yt

def events_to_timestamp_image_torch(xs, ys, ts, ps,
        device=None, sensor_size=(180, 240), clip_out_of_range=True,
        interpolation='bilinear', padding=True, timestamp_reverse=False):
    """
    Method to generate the average timestamp images from 'Zhu19, Unsupervised Event-based Learning 
    of Optical Flow, Depth, and Egomotion'. This method does not have known derivative.
    Parameters
    ----------
    xs : list of event x coordinates 
    ys : list of event y coordinates 
    ts : list of event timestamps 
    ps : list of event polarities 
    device : the device that the events are on
    sensor_size : the size of the event sensor/output voxels
    clip_out_of_range: if the events go beyond the desired image size,
       clip the events to fit into the image
    interpolation: which interpolation to use. Options=None,'bilinear'
    padding: if bilinear interpolation, allow padding the image by 1 to allow events to fit:
    timestamp_reverse: reverse the timestamps of the events, for backward warp
    Returns
    -------
    img_pos: timestamp image of the positive events
    img_neg: timestamp image of the negative events 
    """
    if device is None:
        device = xs.device
    xs, ys, ps, ts = xs.squeeze(), ys.squeeze(), ps.squeeze(), ts.squeeze()
    if padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = sensor_size
    zero_v = torch.tensor([0.], device=device)
    ones_v = torch.tensor([1.], device=device)

    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    pos_events_mask = torch.where(ps>0, ones_v, zero_v)
    neg_events_mask = torch.where(ps<=0, ones_v, zero_v)
    epsilon = 1e-6
    if timestamp_reverse:
        normalized_ts = ((-ts+ts[-1])/(ts[-1]-ts[0]+epsilon)).squeeze()
    else:
        normalized_ts = ((ts-ts[0])/(ts[-1]-ts[0]+epsilon)).squeeze()
    pxs = xs.floor().float()
    pys = ys.floor().float()
    dxs = (xs-pxs).float() 
    dys = (ys-pys).float()
    pxs = (pxs*mask).long()
    pys = (pys*mask).long()
    masked_ps = ps*mask

    pos_weights = (normalized_ts*pos_events_mask).float()
    neg_weights = (normalized_ts*neg_events_mask).float()
    img_pos = torch.zeros(img_size).to(device)
    img_pos_cnt = torch.ones(img_size).to(device)
    img_neg = torch.zeros(img_size).to(device)
    img_neg_cnt = torch.ones(img_size).to(device)

    interpolate_to_image(pxs, pys, dxs, dys, pos_weights, img_pos)
    interpolate_to_image(pxs, pys, dxs, dys, pos_events_mask, img_pos_cnt)
    interpolate_to_image(pxs, pys, dxs, dys, neg_weights, img_neg)
    interpolate_to_image(pxs, pys, dxs, dys, neg_events_mask, img_neg_cnt)

    # Avoid division by 0
    img_pos_cnt[img_pos_cnt==0] = 1
    img_neg_cnt[img_neg_cnt==0] = 1
    img_pos = img_pos.div(img_pos_cnt)
    img_neg = img_neg.div(img_neg_cnt)
    return img_pos, img_neg #/img_pos_cnt, img_neg/img_neg_cnt

def events_to_voxel(xs, ys, ts, ps, B, sensor_size=(180, 240), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    Parameters
    ----------
    xs : list of event x coordinates 
    ys : list of event y coordinates 
    ts : list of event timestamps 
    ps : list of event polarities 
    B : number of bins in output voxel grids (int)
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxel: voxel of the events between t0 and t1
    """
    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    num_events_per_bin = len(xs)//B
    bins = []
    dt = ts[-1]-ts[0]
    t_norm = (ts-ts[0])/dt*(B-1)
    zeros = np.expand_dims(np.zeros(t_norm.shape[0]), axis=0).transpose()
    for bi in range(B):
        if temporal_bilinear:
            bilinear_weights = np.maximum(zeros, 1.0-np.abs(t_norm-bi))
            weights = ps*bilinear_weights
        else:
            beg = bi*num_events_per_bin
            end = beg + num_events_per_bin
            vb = events_to_image(xs[beg:end], ys[beg:end],
                    weights[beg:end], sensor_size=sensor_size)
        vb = events_to_image(xs, ys, weights.squeeze(), sensor_size=sensor_size, interpolation=None)
        bins.append(vb)
    bins = np.stack(bins)
    return bins

def events_to_neg_pos_voxel(xs, ys, ts, ps, B,
        sensor_size=(180, 240), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation.
    Positive and negative events are put into separate voxel grids
    Parameters
    ----------
    xs : list of event x coordinates 
    ys : list of event y coordinates 
    ts : list of event timestamps 
    ps : list of event polarities 
    B : number of bins in output voxel grids (int)
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxel_pos: voxel of the positive events
    voxel_neg: voxel of the negative events
    """
    pos_weights = np.where(ps, 1, 0)
    neg_weights = np.where(ps, 0, 1)

    voxel_pos = events_to_voxel(xs, ys, ts, pos_weights, B,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)
    voxel_neg = events_to_voxel(xs, ys, ts, neg_weights, B,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)

    return voxel_pos, voxel_neg

if __name__ == "__main__":
    """
    Quick demo of some of the voxel/event image generating functions
    in the utils library
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="h5 events path")
    args = parser.parse_args()
    events = read_h5_events(args.path)
    xs, ys, ts, ps = read_h5_event_components(args.path)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    test_loop = 100
    s=80000
    e=s+150000
    bins = 5

    start = time.time()
    for i in range(test_loop):
        xt, yt, tt, pt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ts), torch.from_numpy(ps)
        xt = xt.float().to(device)
        yt = yt.float().to(device)
        tt = (tt[:]-tt[0]).float().to(device)
        pt = pt.float().to(device)
    end = time.time()
    print("conversion to torch: time elapsed  = {:0.5f}".format((end-start)/test_loop))

    start = time.time()
    t0 = ts[len(ts)-1] #worst case
    for i in range(test_loop):
       idx = binary_search_h5_timestamp(args.path, 0, None, t0)
    end = time.time()
    print("binary search of hdf5 (idx={}): time elapsed  = {:0.5f}".format(idx, (end-start)/test_loop))

    start = time.time()
    t0 = ts[len(ts)-1] #worst case
    for i in range(test_loop):
        idx = np.searchsorted(ts, t0)
    end = time.time()
    print("binary search of np timestamps (idx={}): time elapsed  = {:0.5f}".format(idx, (end-start)/test_loop))

    start = time.time()
    for i in range(test_loop):
        events_to_image(xs, ys, ps)
    end = time.time()
    print("event-to-image, numpy: time elapsed  = {:0.5f}".format((end-start)/test_loop))

    start = time.time()
    for i in range(test_loop):
        events_to_image_torch(xt, yt, pt, device, clip_out_of_range=False, padding=False)
    end = time.time()
    print("event-to-image, vanilla: time elapsed = {:0.5f}".format((end-start)/test_loop))

    start = time.time()
    for i in range(test_loop):
        events_to_image_torch(xt, yt, pt, device, interpolation='bilinear')
    end = time.time()
    print("event-to-image, bilinear: time elapsed = {:0.5f}".format((end-start)/test_loop))

    start = time.time()
    for i in range(test_loop):
        events_to_voxel_torch(xt[s:e], yt[s:e], tt[s:e], pt[s:e], bins, device)
    end = time.time()
    print("voxel grid: time elapsed = {:0.5f}".format((end-start)/test_loop))

    start = time.time()
    for i in range(test_loop):
        events_to_voxel_timesync_torch(xt, yt, tt, pt, bins, tt[s], tt[e])
    end = time.time()
    print("voxel grid timesynced: time elapsed = {:0.5f}".format((end-start)/test_loop))

    start = time.time()
    for i in range(test_loop):
        vgs = voxel_grids_fixed_t_torch(xt, yt, tt, pt, bins, 0.1)
    end = time.time()
    tottime = (end-start)/test_loop
    print("voxel grids fixed t: time elapsed = {:0.5f}/{}={:0.5f}".format(tottime, len(vgs), tottime/len(vgs)))

    start = time.time()
    for i in range(test_loop):
        vgs = voxel_grids_fixed_n_torch(xt, yt, tt, pt, bins, len(xt)//34)
    end = time.time()
    tottime = (end-start)/test_loop
    print("voxel grids fixed n: time elapsed = {:0.5f}/{}={:0.5f}".format(tottime, len(vgs), tottime/len(vgs)))

    start = time.time()
    for i in range(test_loop):
        vgs = warp_events_flow_torch(xt, yt, tt, pt, None) 
    end = time.time()
    tottime = (end-start)/test_loop
    print("voxel grids fixed n: time elapsed = {:0.5f}/{}={:0.5f}".format(tottime, len(vgs), tottime/len(vgs)))
