# utils.py
import numpy as np

def segment_and_update_residuals(events, theta_star, T):
    """
    根据梯度阈值 T 对事件进行分簇，并更新残差事件
    :param events: 当前残差事件列表（列表元素为字典，假设包含 'x', 'y' 坐标键）
    :param theta_star: 估计的运动参数（具体结构依实际定义，此处为示例）
    :param T: 梯度阈值
    :return: （当前运动簇, 更新后的残差事件列表）
    """
    current_cluster = []
    residual_events = []
    for event in events:
        xk = np.array([event['x'], event['y']])  # 提取事件坐标
        # 此处需替换为真实的梯度计算（对应公式6），先模拟一个梯度计算函数
        gradient = compute_gradient(xk, theta_star)  # 假设的梯度计算逻辑
        gradient_norm = np.linalg.norm(gradient)  # 计算梯度幅值
        if gradient_norm > T:
            current_cluster.append(event)  # 符合阈值，归为当前簇
        else:
            residual_events.append(event)  # 否则作为残差事件
    return current_cluster, residual_events

def compute_gradient(xk, theta_star):
    """
    模拟梯度计算（需根据论文公式6替换为真实实现）
    :param xk: 事件坐标 [x, y]
    :param theta_star: 运动参数
    :return: 梯度向量（示例返回随机值，实际需按公式计算）
    """
    # 示例：随机返回一个二维向量模拟梯度，实际应基于对比度损失函数求导
    return np.random.rand(2)