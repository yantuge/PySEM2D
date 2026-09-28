import numpy as np
import os

class GLL:
    """
    封装GLL求积所需的所有信息
    """
    def __init__(self, order):
        """
        初始化 GLL 求积器
        
        参数:
            order: 多项式阶数 P，GLL点数 NGLL = P + 1
        """
        if not 2 <= order <= 8:
            raise ValueError("Order must be between 2 and 8.")
        
        self.order = order
        self.ngll = order + 1  # GLL点数
        self.points, self.weights, self.derivative_matrix = self._load_gll_from_file(self.ngll)

    def _load_gll_from_file(self, ngll):
        """
        从预计算的GLL表格文件中读取积分点、权重和导数矩阵
        文件格式 (每列对应一个GLL点):
        第1行: GLL点坐标 x
        第2行: 积分权重 w
        第3行及以后: 导数矩阵 h
        """
        # 构建文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(current_dir, 'gll_constant', f'gll_{ngll:02d}.dat')
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"GLL data file not found: {filename}")
        
        # 读取数据文件
        # 文件是 (ngll+2) x ngll 的矩阵，每列是一个GLL点的数据
        data = np.loadtxt(filename)
        
        # Python的loadtxt按行读取，得到 (ngll+2) x ngll
        data = data.T  # 现在是 ngll x (ngll+2)
        
        x = data[:, 0]     # GLL points
        w = data[:, 1]     # Weights  
        h = data[:, 2:].T  # Derivative matrix (转置很重要!!)
        
        return x, w, h
        

def lagrange_any(xi, ngll, xigll):
    """
    计算任意点xi处的拉格朗日插值函数及其一阶导数
    
    参数:
        xi: float, 参考坐标系中的点位置，范围 [-1, 1]
        ngll: int, GLL点数量
        xigll: numpy.ndarray, GLL点坐标数组
    
    返回:
        h: numpy.ndarray, 拉格朗日插值函数值，形状 (ngll,)
        hprime: numpy.ndarray, 拉格朗日插值函数导数值，形状 (ngll,)
    """
    h = np.zeros(ngll)
    hprime = np.zeros(ngll)
    
    for dgr in range(ngll):
        prod1 = 1.0
        prod2 = 1.0
        
        # 拉格朗日插值多项式
        x0 = xigll[dgr]
        for i in range(ngll):
            if i != dgr:
                x = xigll[i]
                prod1 *= (xi - x)
                prod2 *= (x0 - x)
        
        # 计算倒数避免多次除法
        prod2_inv = 1.0 / prod2
        h[dgr] = prod1 * prod2_inv
        
        # 一阶导数计算
        sum_val = 0.0
        for i in range(ngll):
            if i != dgr:
                prod3 = 1.0
                for j in range(ngll):
                    if j != dgr and j != i:
                        prod3 *= (xi - xigll[j])
                sum_val += prod3
        hprime[dgr] = sum_val * prod2_inv
    
    return h, hprime