"""
Taichi内核实现
"""
import taichi as ti
import numpy as np
from src.taichi_config import TaichiConfig

# NOTE: Taichi backend is initialized by taichi_config.TaichiConfig before this module is used
# Do NOT call ti.init() here - it will be called externally based on user configuration


class GPUSparseMatrix:
    """
    稀疏矩阵的GPU表示，支持CSR和CSC格式
    """
    def __init__(self, scipy_sparse_matrix, format='csr'):
        """
        从scipy稀疏矩阵初始化GPU稀疏矩阵
        
        参数:
            scipy_sparse_matrix: scipy.sparse matrix (csc_matrix or csr_matrix)
            format: str, 'csr' or 'csc' (default: 'csr' for better performance)
        """
        self.nrows = scipy_sparse_matrix.shape[0]
        self.ncols = scipy_sparse_matrix.shape[1]
        self.nnz = scipy_sparse_matrix.nnz
        self.format = format
        
        storage = scipy_sparse_matrix.tocsr() if format == 'csr' else scipy_sparse_matrix.tocsc()
        # A disabled damping direction has empty rows. Taichi still requires
        # positive field dimensions. The zero pointers keep this slot unused.
        capacity = max(1, storage.nnz)
        self.data = ti.field(dtype=float, shape=capacity)
        self.indices = ti.field(dtype=ti.i32, shape=capacity)
        self.indptr = ti.field(dtype=ti.i32, shape=storage.indptr.size)
        if storage.nnz:
            self.data.from_numpy(storage.data.astype(TaichiConfig.get_np_float_type()))
            self.indices.from_numpy(storage.indices.astype(np.int32))
        self.indptr.from_numpy(storage.indptr.astype(np.int32))


@ti.data_oriented
class SEMGPUKernels:
    """
    SEM求解器的GPU内核集合
    """
    def __init__(self, num_nodes, num_receivers, nt, fast_mode=False, num_sources=0, num_snapshots=0):
        """
        初始化GPU数据结构
        
        参数:
            num_nodes: int, 全局节点数
            num_receivers: int, 接收器数量
            nt: int, 时间步数
            fast_mode: bool, 是否启用快速计算模式
            num_sources: int, 震源数量（快速模式需要）
            num_snapshots: int, 快照数量（快速模式需要）
        """
        self.num_nodes = num_nodes
        self.num_receivers = num_receivers
        self.nt = nt
        self.fast_mode = fast_mode
        self.fixed_outer = ti.field(dtype=ti.i32, shape=num_nodes)
        self.num_sources = num_sources
        self.num_snapshots = num_snapshots
        
        # Wave field variables (2D vectors for x and z components)
        self.displ_now = ti.Vector.field(2, dtype=float, shape=num_nodes)
        self.displ_old_term1 = ti.Vector.field(2, dtype=float, shape=num_nodes)
        self.displ_old_term2 = ti.Vector.field(2, dtype=float, shape=num_nodes)
        self.displ_old_term3 = ti.Vector.field(2, dtype=float, shape=num_nodes)
        self.displ_now_term1 = ti.Vector.field(2, dtype=float, shape=num_nodes)
        self.displ_now_term2 = ti.Vector.field(2, dtype=float, shape=num_nodes)
        self.displ_now_term3 = ti.Vector.field(2, dtype=float, shape=num_nodes)
        self.displ_new_term1 = ti.Vector.field(2, dtype=float, shape=num_nodes)
        self.displ_new_term2 = ti.Vector.field(2, dtype=float, shape=num_nodes)
        self.displ_new_term3 = ti.Vector.field(2, dtype=float, shape=num_nodes)
        self.p_x = ti.Vector.field(2, dtype=float, shape=num_nodes)
        self.p_z = ti.Vector.field(2, dtype=float, shape=num_nodes)
        
        # Historical displacement for time derivative computation
        self.displ_prev = ti.Vector.field(2, dtype=float, shape=num_nodes)   # t-dt
        self.displ_prev2 = ti.Vector.field(2, dtype=float, shape=num_nodes)  # t-2*dt
        
        # Velocity and acceleration fields (always allocated)
        self.velocity = ti.Vector.field(2, dtype=float, shape=num_nodes)
        self.acceleration = ti.Vector.field(2, dtype=float, shape=num_nodes)
        
        # Temporary arrays for sparse matrix multiplication
        self.temp_vec = ti.field(dtype=float, shape=num_nodes)
        self.temp_result = ti.field(dtype=float, shape=num_nodes)
        
        # Performance optimization: cached displacement components
        self.displ_x = ti.field(dtype=float, shape=num_nodes)
        self.displ_z = ti.field(dtype=float, shape=num_nodes)
        
        # Additional temporary buffers for fused kernel optimization
        self.temp_Kx1 = ti.field(dtype=float, shape=num_nodes)
        self.temp_Kx2 = ti.field(dtype=float, shape=num_nodes)
        self.temp_Kx3 = ti.field(dtype=float, shape=num_nodes)
        self.temp_Kz3 = ti.field(dtype=float, shape=num_nodes)
        
        # Performance optimization: precomputed coefficients
        self.coeff_a0M = ti.field(dtype=float, shape=num_nodes)
        self.coeff_a1_2Cx = ti.field(dtype=float, shape=num_nodes)
        self.coeff_a1_2Cz = ti.field(dtype=float, shape=num_nodes)
        self.coeff_a1_Cx_Cz = ti.field(dtype=float, shape=num_nodes)
        self.coeff_a2M = ti.field(dtype=float, shape=num_nodes)
        self.coeff_2a1M = ti.field(dtype=float, shape=num_nodes)
        self.coeff_Cxx_minus_a2M = ti.field(dtype=float, shape=num_nodes)
        self.coeff_Cxz_minus_a2M = ti.field(dtype=float, shape=num_nodes)
        self.coeff_Czz_minus_a2M = ti.field(dtype=float, shape=num_nodes)
        
        # Diagonal matrices (stored as vectors)
        self.M = ti.field(dtype=float, shape=num_nodes)
        self.Cx = ti.field(dtype=float, shape=num_nodes)
        self.Cz = ti.field(dtype=float, shape=num_nodes)
        self.Cxx = ti.field(dtype=float, shape=num_nodes)
        self.Cxz = ti.field(dtype=float, shape=num_nodes)
        self.Czz = ti.field(dtype=float, shape=num_nodes)
        
        # Precomputed inverse matrices
        self.A1_inv = ti.field(dtype=float, shape=num_nodes)
        self.A2_inv = ti.field(dtype=float, shape=num_nodes)
        self.A3_inv = ti.field(dtype=float, shape=num_nodes)
        self.AL_inv = ti.field(dtype=float, shape=num_nodes)
        
        # Receiver data
        self.receiver_indices = ti.field(dtype=ti.i32, shape=num_receivers)
        
        # Seismogram data (displacement, velocity, acceleration - always allocated)
        self.seismograms_disp = ti.Vector.field(2, dtype=float, shape=(num_receivers, nt))
        self.seismograms_vel = ti.Vector.field(2, dtype=float, shape=(num_receivers, nt))
        self.seismograms_acc = ti.Vector.field(2, dtype=float, shape=(num_receivers, nt))
        
        # Sparse stiffness matrices (will be initialized later)
        self.sparse_matrices = {}
        
        # Fast mode specific fields
        if fast_mode:
            # Source time functions (preloaded for fast mode)
            # Shape: (num_sources, nt)
            self.source_stf = ti.field(dtype=float, shape=(num_sources, nt))
            
            # Source parameters
            self.source_indices = ti.field(dtype=ti.i32, shape=num_sources)
            self.source_sin_angles = ti.field(dtype=float, shape=num_sources)
            self.source_cos_angles = ti.field(dtype=float, shape=num_sources)
            self.source_time_delays = ti.field(dtype=ti.i32, shape=num_sources)
            
            # Autotune fields for sparse matrix-vector multiplication optimization
            #self.nnz_row_6x = ti.field(dtype=ti.i32, shape=num_nodes)
            #self.nnz_row_4x = ti.field(dtype=ti.i32, shape=num_nodes)
            #self.strategy_6x = ti.field(dtype=ti.i32, shape=num_nodes)
            #self.strategy_4x = ti.field(dtype=ti.i32, shape=num_nodes)
            
            # Snapshot storage
            if num_snapshots > 0:
                # Store displacement, velocity, acceleration snapshots
                # Shape: (num_snapshots, num_nodes, 2)
                self.snapshot_disp = ti.Vector.field(2, dtype=float, shape=(num_snapshots, num_nodes))
                self.snapshot_vel = ti.Vector.field(2, dtype=float, shape=(num_snapshots, num_nodes))
                self.snapshot_acc = ti.Vector.field(2, dtype=float, shape=(num_snapshots, num_nodes))
                
                # Snapshot metadata
                self.snapshot_timesteps = ti.field(dtype=ti.i32, shape=num_snapshots)
                self.snapshot_count = ti.field(dtype=ti.i32, shape=())  # Counter for snapshots saved
    
    def load_diagonal_matrices(self, M, Cx, Cz, Cxx, Cxz, Czz, A1_inv, A2_inv, A3_inv, AL_inv):
        """
        加载对角矩阵到GPU
        """
        self.M.from_numpy(M.astype(TaichiConfig.get_np_float_type()))
        self.Cx.from_numpy(Cx.astype(TaichiConfig.get_np_float_type()))
        self.Cz.from_numpy(Cz.astype(TaichiConfig.get_np_float_type()))
        self.Cxx.from_numpy(Cxx.astype(TaichiConfig.get_np_float_type()))
        self.Cxz.from_numpy(Cxz.astype(TaichiConfig.get_np_float_type()))
        self.Czz.from_numpy(Czz.astype(TaichiConfig.get_np_float_type()))
        self.A1_inv.from_numpy(A1_inv.astype(TaichiConfig.get_np_float_type()))
        self.A2_inv.from_numpy(A2_inv.astype(TaichiConfig.get_np_float_type()))
        self.A3_inv.from_numpy(A3_inv.astype(TaichiConfig.get_np_float_type()))
        self.AL_inv.from_numpy(AL_inv.astype(TaichiConfig.get_np_float_type()))
    
    def load_sparse_matrices(self, Kx1, Kx2, Kx3, Kx4, Kx5, Kz1, Kz2, Kz3, Kz4, Kz5):
        """
        加载稀疏刚度矩阵到GPU，使用CSR格式
        """
        self.sparse_matrices['Kx1'] = GPUSparseMatrix(Kx1, format='csr')
        self.sparse_matrices['Kx2'] = GPUSparseMatrix(Kx2, format='csr')
        self.sparse_matrices['Kx3'] = GPUSparseMatrix(Kx3, format='csr')
        self.sparse_matrices['Kx4'] = GPUSparseMatrix(Kx4, format='csr')
        self.sparse_matrices['Kx5'] = GPUSparseMatrix(Kx5, format='csr')
        self.sparse_matrices['Kz1'] = GPUSparseMatrix(Kz1, format='csr')
        self.sparse_matrices['Kz2'] = GPUSparseMatrix(Kz2, format='csr')
        self.sparse_matrices['Kz3'] = GPUSparseMatrix(Kz3, format='csr')
        self.sparse_matrices['Kz4'] = GPUSparseMatrix(Kz4, format='csr')
        self.sparse_matrices['Kz5'] = GPUSparseMatrix(Kz5, format='csr')
    
    def load_receivers(self, receiver_indices):
        """
        加载接收器信息到GPU
        """
        self.receiver_indices.from_numpy(receiver_indices.astype(np.int32))
    
    def load_moment_tensor_sources(self, source_arrays, source_elements, connectivity):
        """
        加载矩震源数据到GPU
        
        参数:
            source_arrays: numpy.ndarray, 形状 (num_sources, 2, NGLL, NGLL) 的震源分布数组
            source_elements: numpy.ndarray, 形状 (num_sources,) 的震源所在单元索引
            connectivity: numpy.ndarray, 形状 (num_elements, NGLL, NGLL) 的单元连接关系
        """
        # 保存震源数组（作为numpy数组，在add_moment_tensor_source中使用）
        self.mt_source_arrays = source_arrays.astype(TaichiConfig.get_np_float_type())
        self.mt_source_elements = source_elements.astype(np.int32)
        self.mt_connectivity = connectivity.astype(np.int32)
        
        print(f"    Loaded {len(source_elements)} moment tensor source(s)")
        print(f"    Connectivity shape: {connectivity.shape}")
        
    def precompute_coefficients(self, a0, a1, a2):
        """
        预计算时间积分常数项
        
        参数:
            a0: float, 时间积分常数 1/dt^2
            a1: float, 时间积分常数 1/(2*dt)
            a2: float, 时间积分常数 2*a0
        """
        # 预计算常用的常数组合
        M_np = self.M.to_numpy()
        Cx_np = self.Cx.to_numpy()
        Cz_np = self.Cz.to_numpy()
        Cxx_np = self.Cxx.to_numpy()
        Cxz_np = self.Cxz.to_numpy()
        Czz_np = self.Czz.to_numpy()
        
        self.coeff_a0M.from_numpy((a0 * M_np).astype(TaichiConfig.get_np_float_type()))
        self.coeff_a1_2Cx.from_numpy((a1 * 2.0 * Cx_np).astype(TaichiConfig.get_np_float_type()))
        self.coeff_a1_2Cz.from_numpy((a1 * 2.0 * Cz_np).astype(TaichiConfig.get_np_float_type()))
        self.coeff_a1_Cx_Cz.from_numpy((a1 * (Cx_np + Cz_np)).astype(TaichiConfig.get_np_float_type()))
        self.coeff_a2M.from_numpy((a2 * M_np).astype(TaichiConfig.get_np_float_type()))
        self.coeff_2a1M.from_numpy((2.0 * a1 * M_np).astype(TaichiConfig.get_np_float_type()))
        self.coeff_Cxx_minus_a2M.from_numpy((Cxx_np - a2 * M_np).astype(TaichiConfig.get_np_float_type()))
        self.coeff_Cxz_minus_a2M.from_numpy((Cxz_np - a2 * M_np).astype(TaichiConfig.get_np_float_type()))
        self.coeff_Czz_minus_a2M.from_numpy((Czz_np - a2 * M_np).astype(TaichiConfig.get_np_float_type()))
    
    @ti.kernel
    def cache_displacement_components(self):
        """
        缓存位移分量到独立的标量场
        
        """
        for i in range(self.num_nodes):
            self.displ_x[i] = self.displ_now[i][0]
            self.displ_z[i] = self.displ_now[i][1]
    
    @ti.kernel
    def sparse_matvec(self, data: ti.template(), indices: ti.template(), 
                      indptr: ti.template(), x: ti.template(), 
                      result: ti.template(), nrows: ti.i32):
        """
        稀疏矩阵-向量乘法: result = A @ x
        默认使用CSR格式
        """
        # CSR格式实现：每个线程独立计算一行，无需原子操作
        for i in range(nrows):
            row_start = indptr[i]
            row_end = indptr[i + 1]
            
            # Accumulate dot product for row i
            sum_val = 0.0
            for k in range(row_start, row_end):
                col_idx = indices[k]
                value = data[k]
                sum_val += value * x[col_idx]
            
            result[i] = sum_val
    
    @ti.kernel
    def sparse_matvec_6x_fused(self,
                                data1: ti.template(), indices1: ti.template(), indptr1: ti.template(),
                                data2: ti.template(), indices2: ti.template(), indptr2: ti.template(),
                                data3: ti.template(), indices3: ti.template(), indptr3: ti.template(),
                                data4: ti.template(), indices4: ti.template(), indptr4: ti.template(),
                                data5: ti.template(), indices5: ti.template(), indptr5: ti.template(),
                                data6: ti.template(), indices6: ti.template(), indptr6: ti.template(),
                                x_vec: ti.template(), z_vec: ti.template(),
                                result1: ti.template(), result2: ti.template(), result3: ti.template(),
                                result4: ti.template(), result5: ti.template(), result6: ti.template(),
                                nrows: ti.i32):
        """
        融合6次稀疏矩阵-向量乘法, 一次性完成以下计算：
        result1 = Kx1 @ x_vec
        result2 = Kx2 @ z_vec
        result3 = Kx3 @ x_vec
        result4 = Kz1 @ z_vec
        result5 = Kz2 @ x_vec
        result6 = Kz3 @ z_vec

        """
        for i in range(nrows):
            # 预取所有行指针, 减少全局内存访问
            start1, end1 = indptr1[i], indptr1[i + 1]
            start2, end2 = indptr2[i], indptr2[i + 1]
            start3, end3 = indptr3[i], indptr3[i + 1]
            start4, end4 = indptr4[i], indptr4[i + 1]
            start5, end5 = indptr5[i], indptr5[i + 1]
            start6, end6 = indptr6[i], indptr6[i + 1]
            
            # 初始化所有累加器
            sum1, sum2, sum3, sum4, sum5, sum6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            # Matrix 1: Kx1 @ x_vec
            for k in range(start1, end1):
                sum1 += data1[k] * x_vec[indices1[k]]
            
            # Matrix 2: Kx2 @ z_vec
            for k in range(start2, end2):
                sum2 += data2[k] * z_vec[indices2[k]]
            
            # Matrix 3: Kx3 @ x_vec
            for k in range(start3, end3):
                sum3 += data3[k] * x_vec[indices3[k]]
            
            # Matrix 4: Kz1 @ z_vec
            for k in range(start4, end4):
                sum4 += data4[k] * z_vec[indices4[k]]
            
            # Matrix 5: Kz2 @ x_vec
            for k in range(start5, end5):
                sum5 += data5[k] * x_vec[indices5[k]]
            
            # Matrix 6: Kz3 @ z_vec
            for k in range(start6, end6):
                sum6 += data6[k] * z_vec[indices6[k]]
            
            # 写回结果, 批量写入
            result1[i] = sum1
            result2[i] = sum2
            result3[i] = sum3
            result4[i] = sum4
            result5[i] = sum5
            result6[i] = sum6
    
    @ti.kernel
    def sparse_matvec_4x_fused(self,
                                data1: ti.template(), indices1: ti.template(), indptr1: ti.template(),
                                data2: ti.template(), indices2: ti.template(), indptr2: ti.template(),
                                data3: ti.template(), indices3: ti.template(), indptr3: ti.template(),
                                data4: ti.template(), indices4: ti.template(), indptr4: ti.template(),
                                x_vec: ti.template(), z_vec: ti.template(),
                                result1: ti.template(), result2: ti.template(),
                                result3: ti.template(), result4: ti.template(),
                                nrows: ti.i32):
        """
        融合4次稀疏矩阵-向量乘法, 一次性完成以下计算：
        result1 = Kx4 @ x_vec
        result2 = Kx5 @ x_vec
        result3 = Kz4 @ z_vec
        result4 = Kz5 @ z_vec
        """
        for i in range(nrows):
            # 预取所有行指针
            start1, end1 = indptr1[i], indptr1[i + 1]
            start2, end2 = indptr2[i], indptr2[i + 1]
            start3, end3 = indptr3[i], indptr3[i + 1]
            start4, end4 = indptr4[i], indptr4[i + 1]
            
            # 初始化所有累加器
            sum1, sum2, sum3, sum4 = 0.0, 0.0, 0.0, 0.0
            
            # Matrix 1: Kx4 @ x_vec
            for k in range(start1, end1):
                sum1 += data1[k] * x_vec[indices1[k]]
            
            # Matrix 2: Kx5 @ x_vec
            for k in range(start2, end2):
                sum2 += data2[k] * x_vec[indices2[k]]
            
            # Matrix 3: Kz4 @ z_vec
            for k in range(start3, end3):
                sum3 += data3[k] * z_vec[indices3[k]]
            
            # Matrix 4: Kz5 @ z_vec
            for k in range(start4, end4):
                sum4 += data4[k] * z_vec[indices4[k]]
            
            # 写回结果
            result1[i] = sum1
            result2[i] = sum2
            result3[i] = sum3
            result4[i] = sum4
    
    @ti.kernel
    def update_displ_new_term1_x(self, a0: float, a1: float, a2: float,
                                   Kx1_result: ti.template()):
        """
        更新X方向term1: displ_new_term1[:, 0]
        """
        for i in range(self.num_nodes):
            # Use precomputed coefficients to reduce arithmetic operations
            self.displ_new_term1[i][0] = self.A1_inv[i] * (
                self.M[i] * self.p_x[i][0] 
                - Kx1_result[i]
                - self.coeff_Cxx_minus_a2M[i] * self.displ_now_term1[i][0]
                - (self.coeff_a0M[i] - self.coeff_a1_2Cx[i]) * self.displ_old_term1[i][0]
            )
    
    @ti.kernel
    def update_displ_new_term2_x(self, a0: float, a1: float, a2: float,
                                   Kx2_result: ti.template()):
        """
        更新X方向term2: displ_new_term2[:, 0]
        """
        for i in range(self.num_nodes):
            # Use precomputed coefficients to reduce arithmetic operations
            self.displ_new_term2[i][0] = self.A2_inv[i] * (
                - Kx2_result[i]
                - self.coeff_Cxz_minus_a2M[i] * self.displ_now_term2[i][0]
                - (self.coeff_a0M[i] - self.coeff_a1_Cx_Cz[i]) * self.displ_old_term2[i][0]
            )
    
    @ti.kernel
    def update_displ_new_term3_x(self, a0: float, a1: float, a2: float,
                                   Kx3_result: ti.template()):
        """
        更新X方向term3: displ_new_term3[:, 0]
        """
        for i in range(self.num_nodes):
            # Use precomputed coefficients to reduce arithmetic operations
            self.displ_new_term3[i][0] = self.A3_inv[i] * (
                self.M[i] * self.p_x[i][1]
                - Kx3_result[i]
                - self.coeff_Czz_minus_a2M[i] * self.displ_now_term3[i][0]
                - (self.coeff_a0M[i] - self.coeff_a1_2Cz[i]) * self.displ_old_term3[i][0]
            )
    
    @ti.kernel
    def update_p_x_0(self, a1: float, Kx4_result: ti.template()):
        """
        更新p_x[:, 0]
        """
        for i in range(self.num_nodes):
            # Use local variable to avoid repeated global memory access
            p_x_old = self.p_x[i][0]
            self.p_x[i][0] = self.AL_inv[i] * (
                - Kx4_result[i]
                - self.Cx[i] * p_x_old
                + self.coeff_2a1M[i] * p_x_old
            )
    
    @ti.kernel
    def update_p_x_1(self, a1: float, Kx5_result: ti.template()):
        """
        更新p_x[:, 1]
        """
        for i in range(self.num_nodes):
            # Use local variable to avoid repeated global memory access
            p_x_old = self.p_x[i][1]
            self.p_x[i][1] = self.AL_inv[i] * (
                - Kx5_result[i]
                - self.Cz[i] * p_x_old
                + self.coeff_2a1M[i] * p_x_old
            )
    
    @ti.kernel
    def update_displ_new_term1_z(self, a0: float, a1: float, a2: float,
                                   Kz1_result: ti.template()):
        """
        更新Z方向term1: displ_new_term1[:, 1]
        """
        for i in range(self.num_nodes):
            # Use precomputed coefficients to reduce arithmetic operations
            self.displ_new_term1[i][1] = self.A1_inv[i] * (
                self.M[i] * self.p_z[i][0]
                - Kz1_result[i]
                - self.coeff_Cxx_minus_a2M[i] * self.displ_now_term1[i][1]
                - (self.coeff_a0M[i] - self.coeff_a1_2Cx[i]) * self.displ_old_term1[i][1]
            )
    
    @ti.kernel
    def update_displ_new_term2_z(self, a0: float, a1: float, a2: float,
                                   Kz2_result: ti.template()):
        """
        更新Z方向term2: displ_new_term2[:, 1]
        """
        for i in range(self.num_nodes):
            # Use precomputed coefficients to reduce arithmetic operations
            self.displ_new_term2[i][1] = self.A2_inv[i] * (
                - Kz2_result[i]
                - self.coeff_Cxz_minus_a2M[i] * self.displ_now_term2[i][1]
                - (self.coeff_a0M[i] - self.coeff_a1_Cx_Cz[i]) * self.displ_old_term2[i][1]
            )
    
    @ti.kernel
    def update_displ_new_term3_z(self, a0: float, a1: float, a2: float,
                                   Kz3_result: ti.template()):
        """
        更新Z方向term3: displ_new_term3[:, 1]
        """
        for i in range(self.num_nodes):
            # Use precomputed coefficients to reduce arithmetic operations
            self.displ_new_term3[i][1] = self.A3_inv[i] * (
                self.M[i] * self.p_z[i][1]
                - Kz3_result[i]
                - self.coeff_Czz_minus_a2M[i] * self.displ_now_term3[i][1]
                - (self.coeff_a0M[i] - self.coeff_a1_2Cz[i]) * self.displ_old_term3[i][1]
            )
    
    @ti.kernel
    def update_p_z_0(self, a1: float, Kz4_result: ti.template()):
        """
        更新p_z[:, 0]
        """
        for i in range(self.num_nodes):
            # Use local variable to avoid repeated global memory access
            p_z_old = self.p_z[i][0]
            self.p_z[i][0] = self.AL_inv[i] * (
                - Kz4_result[i]
                - self.Cx[i] * p_z_old
                + self.coeff_2a1M[i] * p_z_old
            )
    
    @ti.kernel
    def update_p_z_1(self, a1: float, Kz5_result: ti.template()):
        """
        更新p_z[:, 1]
        """
        for i in range(self.num_nodes):
            # Use local variable to avoid repeated global memory access
            p_z_old = self.p_z[i][1]
            self.p_z[i][1] = self.AL_inv[i] * (
                - Kz5_result[i]
                - self.Cz[i] * p_z_old
                + self.coeff_2a1M[i] * p_z_old
            )
    
    @ti.kernel
    def update_history_and_combine(self):
        """
        更新历史项并合成总位移
        """
        for i in range(self.num_nodes):
            # Homogeneous displacement at the exterior face of each PML.
            if self.fixed_outer[i]:
                self.displ_new_term1[i] = ti.Vector.zero(float, 2)
                self.displ_new_term2[i] = ti.Vector.zero(float, 2)
                self.displ_new_term3[i] = ti.Vector.zero(float, 2)
                self.p_x[i] = ti.Vector.zero(float, 2)
                self.p_z[i] = ti.Vector.zero(float, 2)
            # Update history for PML
            self.displ_old_term1[i] = self.displ_now_term1[i]
            self.displ_old_term2[i] = self.displ_now_term2[i]
            self.displ_old_term3[i] = self.displ_now_term3[i]
            
            self.displ_now_term1[i] = self.displ_new_term1[i]
            self.displ_now_term2[i] = self.displ_new_term2[i]
            self.displ_now_term3[i] = self.displ_new_term3[i]
            
            # Update historical displacement for time derivatives
            self.displ_prev2[i] = self.displ_prev[i]
            self.displ_prev[i] = self.displ_now[i]
            
            # Combine total displacement
            self.displ_now[i] = (self.displ_now_term1[i] + 
                                self.displ_now_term2[i] + 
                                self.displ_now_term3[i])
    
    @ti.kernel
    def compute_velocity(self, dt: float):
        """
        计算速度场: v(t) = [u(t) - u(t-dt)] / dt
        """
        for i in range(self.num_nodes):
            self.velocity[i] = (self.displ_now[i] - self.displ_prev[i]) / dt
    
    @ti.kernel
    def compute_acceleration(self, dt: float):
        """
        计算加速度场: a(t) = [u(t) - 2*u(t-dt) + u(t-2*dt)] / dtdt
        """
        dt2 = dt * dt
        for i in range(self.num_nodes):
            self.acceleration[i] = (self.displ_now[i] - 2.0 * self.displ_prev[i] + self.displ_prev2[i]) / dt2
    
    @ti.kernel
    def add_source(self, src_idx: ti.i32, stf_val: float,
                   sin_angle: float, cos_angle: float, dt: float):
        """
        添加单个震源贡献
        
        参数:
            src_idx: 震源节点索引
            stf_val: 当前时间步的震源时间函数值
            sin_angle: sin(force_angle)
            cos_angle: cos(force_angle)
            dt: 时间步长
        """
        # 计算力向量的各分量
        force_x = sin_angle * stf_val
        force_z = cos_angle * stf_val
        
        self._apply_nodal_force(src_idx, force_x, force_z)

    @ti.func
    def _apply_nodal_force(self, node_idx, fx, fz):
        # Drive the mixed split equation and keep its recombined value in sync.
        # In the physical domain A2_inv equals dt**2 / mass.
        if not self.fixed_outer[node_idx]:
            for c in ti.static(range(2)):
                du = self.A2_inv[node_idx] * ti.Vector([fx, fz])[c]
                self.displ_now_term2[node_idx][c] += du
                self.displ_now[node_idx][c] += du

    def add_moment_tensor_source(self, ispec, source_array, stf_val, dt):
        """
        添加矩震源贡献, 矩震源将力分布到震源所在单元的所有GLL节点
        
        参数:
            ispec: int, 震源所在单元索引
            source_array: numpy.ndarray, 形状 (2, NGLL, NGLL) 的震源分布数组
            stf_val: float, 当前时间步的震源时间函数值
            dt: float, 时间步长
        """
        self._add_element_force(self.mt_connectivity[ispec], source_array, stf_val)

    @ti.kernel
    def _add_element_force(self, nodes: ti.types.ndarray(), weights: ti.types.ndarray(),
                           stf: float):
        for j, i in ti.ndrange(nodes.shape[0], nodes.shape[1]):
            self._apply_nodal_force(nodes[j, i], weights[0, i, j] * stf,
                                    weights[1, i, j] * stf)

    @ti.kernel
    def record_seismograms_disp(self, it: ti.i32):
        """
        记录位移地震图
        """
        for rec_id in range(self.num_receivers):
            node_idx = self.receiver_indices[rec_id]
            self.seismograms_disp[rec_id, it] = self.displ_now[node_idx]
    
    @ti.kernel
    def record_seismograms_vel(self, it: ti.i32):
        """
        记录速度地震图
        """
        for rec_id in range(self.num_receivers):
            node_idx = self.receiver_indices[rec_id]
            self.seismograms_vel[rec_id, it] = self.velocity[node_idx]
    
    @ti.kernel
    def record_seismograms_acc(self, it: ti.i32):
        """
        记录加速度地震图
        """
        for rec_id in range(self.num_receivers):
            node_idx = self.receiver_indices[rec_id]
            self.seismograms_acc[rec_id, it] = self.acceleration[node_idx]
    
    @ti.kernel
    def extract_displ_component(self, component: ti.i32, output: ti.template()):
        """
        提取位移的某个分量用于稀疏矩阵乘法
        
        参数:
            component: 0 for x, 1 for z
            output: output field
        """
        for i in range(self.num_nodes):
            output[i] = self.displ_now[i][component]
    
    def get_seismograms_displacement(self):
        """
        从GPU获取位移地震图数据
        
        返回:
            list of numpy arrays, each of shape (2, nt)
        """
        seismo_data = self.seismograms_disp.to_numpy()
        # Shape: (num_receivers, nt, 2) -> transpose to (num_receivers, 2, nt)
        seismo_data = seismo_data.transpose(0, 2, 1)
        
        # Convert to list of arrays for compatibility with receivers
        return [seismo_data[i] for i in range(self.num_receivers)]
    
    def get_seismograms_velocity(self):
        """
        从GPU获取速度地震图数据
        
        返回:
            list of numpy arrays, each of shape (2, nt)
        """
        seismo_data = self.seismograms_vel.to_numpy()
        seismo_data = seismo_data.transpose(0, 2, 1)
        return [seismo_data[i] for i in range(self.num_receivers)]
    
    def get_seismograms_acceleration(self):
        """
        从GPU获取加速度地震图数据
        
        返回:
            list of numpy arrays, each of shape (2, nt)
        """
        seismo_data = self.seismograms_acc.to_numpy()
        seismo_data = seismo_data.transpose(0, 2, 1)
        return [seismo_data[i] for i in range(self.num_receivers)]
    
    # 向后兼容
    def get_seismograms(self):
        """
        返回位移地震图
        
        返回:
            numpy array of shape (num_receivers, 2, nt)
        """
        seismo_data = self.seismograms_disp.to_numpy()
        return seismo_data.transpose(0, 2, 1)
    
    def get_displacement(self):
        """
        从GPU获取当前位移场
        
        返回:
            numpy array of shape (num_nodes, 2)
        """
        return self.displ_now.to_numpy()

    def get_velocity(self):
        """
        从GPU获取当前速度场
        
        返回:
            numpy array of shape (num_nodes, 2)
        """
        return self.velocity.to_numpy()
    
    def get_acceleration(self):
        """
        从GPU获取当前加速度场
        
        返回:
            numpy array of shape (num_nodes, 2)
        """
        return self.acceleration.to_numpy()
    
    # ==================== 融合kernels ====================
    
    @ti.kernel
    def update_displ_terms_x_fused(self, 
                                     Kx1_result: ti.template(),
                                     Kx2_result: ti.template(),
                                     Kx3_result: ti.template()):
        """
        融合X方向的三个term更新
        """
        for i in range(self.num_nodes):
            # Term 1 X
            self.displ_new_term1[i][0] = self.A1_inv[i] * (
                self.M[i] * self.p_x[i][0] 
                - Kx1_result[i]
                - self.coeff_Cxx_minus_a2M[i] * self.displ_now_term1[i][0]
                - (self.coeff_a0M[i] - self.coeff_a1_2Cx[i]) * self.displ_old_term1[i][0]
            )
            
            # Term 2 X
            self.displ_new_term2[i][0] = self.A2_inv[i] * (
                - Kx2_result[i]
                - self.coeff_Cxz_minus_a2M[i] * self.displ_now_term2[i][0]
                - (self.coeff_a0M[i] - self.coeff_a1_Cx_Cz[i]) * self.displ_old_term2[i][0]
            )
            
            # Term 3 X
            self.displ_new_term3[i][0] = self.A3_inv[i] * (
                self.M[i] * self.p_x[i][1]
                - Kx3_result[i]
                - self.coeff_Czz_minus_a2M[i] * self.displ_now_term3[i][0]
                - (self.coeff_a0M[i] - self.coeff_a1_2Cz[i]) * self.displ_old_term3[i][0]
            )
    
    @ti.kernel
    def update_displ_terms_z_fused(self,
                                     Kz1_result: ti.template(),
                                     Kz2_result: ti.template(),
                                     Kz3_result: ti.template()):
        """
        融合Z方向的三个term更新
        """
        for i in range(self.num_nodes):
            # Term 1 Z
            self.displ_new_term1[i][1] = self.A1_inv[i] * (
                self.M[i] * self.p_z[i][0]
                - Kz1_result[i]
                - self.coeff_Cxx_minus_a2M[i] * self.displ_now_term1[i][1]
                - (self.coeff_a0M[i] - self.coeff_a1_2Cx[i]) * self.displ_old_term1[i][1]
            )
            
            # Term 2 Z
            self.displ_new_term2[i][1] = self.A2_inv[i] * (
                - Kz2_result[i]
                - self.coeff_Cxz_minus_a2M[i] * self.displ_now_term2[i][1]
                - (self.coeff_a0M[i] - self.coeff_a1_Cx_Cz[i]) * self.displ_old_term2[i][1]
            )
            
            # Term 3 Z
            self.displ_new_term3[i][1] = self.A3_inv[i] * (
                self.M[i] * self.p_z[i][1]
                - Kz3_result[i]
                - self.coeff_Czz_minus_a2M[i] * self.displ_now_term3[i][1]
                - (self.coeff_a0M[i] - self.coeff_a1_2Cz[i]) * self.displ_old_term3[i][1]
            )
    
    @ti.kernel
    def update_p_fields_fused(self,
                               Kx4_result: ti.template(),
                               Kx5_result: ti.template(),
                               Kz4_result: ti.template(),
                               Kz5_result: ti.template()):
        """
        融合p_x和p_z的所有分量更新
        """
        for i in range(self.num_nodes):
            # p_x[0]
            p_x_0_old = self.p_x[i][0]
            self.p_x[i][0] = self.AL_inv[i] * (
                - Kx4_result[i]
                - self.Cx[i] * p_x_0_old
                + self.coeff_2a1M[i] * p_x_0_old
            )
            
            # p_x[1]
            p_x_1_old = self.p_x[i][1]
            self.p_x[i][1] = self.AL_inv[i] * (
                - Kx5_result[i]
                - self.Cz[i] * p_x_1_old
                + self.coeff_2a1M[i] * p_x_1_old
            )
            
            # p_z[0]
            p_z_0_old = self.p_z[i][0]
            self.p_z[i][0] = self.AL_inv[i] * (
                - Kz4_result[i]
                - self.Cx[i] * p_z_0_old
                + self.coeff_2a1M[i] * p_z_0_old
            )
            
            # p_z[1]
            p_z_1_old = self.p_z[i][1]
            self.p_z[i][1] = self.AL_inv[i] * (
                - Kz5_result[i]
                - self.Cz[i] * p_z_1_old
                + self.coeff_2a1M[i] * p_z_1_old
            )
    
    # ==================== Fast Mode Methods ====================
    
    def load_source_data(self, stf_list, src_node_indices, sin_angles, cos_angles, time_delays):
        """
        快速模式预加载震源数据到GPU
        
        Parameters:
            stf_list: list of numpy arrays, 每个震源的时间函数
            src_node_indices: numpy array, 震源节点索引
            sin_angles: numpy array, sin(force_angle)
            cos_angles: numpy array, cos(force_angle)
            time_delays: numpy array, 时间延迟（单位为时间步）
        """
        if not self.fast_mode:
            raise RuntimeError("load_source_data can only be called in fast mode")
        
        # Load source STF data
        for src_idx, stf in enumerate(stf_list):
            # Ensure STF has correct length
            if len(stf) < self.nt:
                # Pad with zeros
                stf_padded = np.zeros(self.nt, dtype=TaichiConfig.get_np_float_type())
                stf_padded[:len(stf)] = stf
                stf = stf_padded
            elif len(stf) > self.nt:
                # Truncate
                stf = stf[:self.nt]
            
            # Copy to GPU
            for it in range(self.nt):
                self.source_stf[src_idx, it] = stf[it]
        
        # Load source parameters
        self.source_indices.from_numpy(src_node_indices.astype(np.int32))
        self.source_sin_angles.from_numpy(sin_angles.astype(TaichiConfig.get_np_float_type()))
        self.source_cos_angles.from_numpy(cos_angles.astype(TaichiConfig.get_np_float_type()))
        self.source_time_delays.from_numpy(time_delays.astype(np.int32))
    
    @ti.kernel
    def save_snapshot(self, snapshot_idx: ti.i32, it: ti.i32):
        """
        保存当前波场快照
        
        参数:
            snapshot_idx: 快照索引
            it: 当前时间步
        """
        # Record timestep
        self.snapshot_timesteps[snapshot_idx] = it
        
        # Copy wavefields
        for i in range(self.num_nodes):
            self.snapshot_disp[snapshot_idx, i] = self.displ_now[i]
            self.snapshot_vel[snapshot_idx, i] = self.velocity[i]
            self.snapshot_acc[snapshot_idx, i] = self.acceleration[i]
    
    def get_snapshot_data(self, snapshot_idx):
        """
        获取指定快照的数据
        
        返回:
            dict with keys: 'displacement', 'velocity', 'acceleration', 'timestep'
        """
        if not self.fast_mode or self.num_snapshots == 0:
            raise RuntimeError("get_snapshot_data can only be called in fast mode with snapshots enabled")
        
        timestep = self.snapshot_timesteps.to_numpy()[snapshot_idx]
        disp = self.snapshot_disp.to_numpy()[snapshot_idx]  # Shape: (num_nodes, 2)
        vel = self.snapshot_vel.to_numpy()[snapshot_idx]
        acc = self.snapshot_acc.to_numpy()[snapshot_idx]
        
        return {
            'timestep': int(timestep),
            'displacement': disp,
            'velocity': vel,
            'acceleration': acc
        }
    
    def run_full_time_loop_batch(self, nt, dt, snapshot_interval, enable_snapshot, moment_source_step=None):
        """
        在快速模式下执行完整时间循环
        
        参数:
            nt: int, 时间步数
            dt: float, 时间步长
            snapshot_interval: int, 快照输出间隔
            enable_snapshot: bool, 是否启用快照
        """
        if not self.fast_mode:
            raise RuntimeError("run_full_time_loop_batch can only be called in fast mode")
        
        # Get sparse matrix references
        Kx1 = self.sparse_matrices['Kx1']
        Kx2 = self.sparse_matrices['Kx2']
        Kx3 = self.sparse_matrices['Kx3']
        Kx4 = self.sparse_matrices['Kx4']
        Kx5 = self.sparse_matrices['Kx5']
        Kz1 = self.sparse_matrices['Kz1']
        Kz2 = self.sparse_matrices['Kz2']
        Kz3 = self.sparse_matrices['Kz3']
        Kz4 = self.sparse_matrices['Kz4']
        Kz5 = self.sparse_matrices['Kz5']
        
        snapshot_counter = 0
        first_snapshot_step = max(snapshot_interval, 3) if enable_snapshot else nt + 1
        
        # Main time loop
        for it in range(nt):
            # Step 1: Update displacement terms
            self.cache_displacement_components()
            
            # Fused sparse matvec for displacement terms (6 matrices in 1 kernel call)
            #self.sparse_matvec(Kx1.data, Kx1.indices, Kx1.indptr, 
            #                  self.displ_x, self.temp_Kx1, Kx1.nrows)
            #self.sparse_matvec(Kx2.data, Kx2.indices, Kx2.indptr, 
            #                  self.displ_z, self.temp_Kx2, Kx2.nrows)
            #self.sparse_matvec(Kx3.data, Kx3.indices, Kx3.indptr, 
            #                  self.displ_x, self.temp_Kx3, Kx3.nrows)
            #self.sparse_matvec(Kz1.data, Kz1.indices, Kz1.indptr, 
            #                  self.displ_z, self.temp_result, Kz1.nrows)
            #self.sparse_matvec(Kz2.data, Kz2.indices, Kz2.indptr, 
            #                  self.displ_x, self.temp_vec, Kz2.nrows)
            #self.sparse_matvec(Kz3.data, Kz3.indices, Kz3.indptr, 
            #                  self.displ_z, self.temp_Kz3, Kz3.nrows)
            # 优化：10次sparse_matvec调用到2次融合调用 (算法保存，以便回退)
            self.sparse_matvec_6x_fused(
                Kx1.data, Kx1.indices, Kx1.indptr,
                Kx2.data, Kx2.indices, Kx2.indptr,
                Kx3.data, Kx3.indices, Kx3.indptr,
                Kz1.data, Kz1.indices, Kz1.indptr,
                Kz2.data, Kz2.indices, Kz2.indptr,
                Kz3.data, Kz3.indices, Kz3.indptr,
                self.displ_x, self.displ_z,
                self.temp_Kx1, self.temp_Kx2, self.temp_Kx3,
                self.temp_result, self.temp_vec, self.temp_Kz3,
                Kx1.nrows
            )
            
            # Clear result arrays for atomic operations
            #self.zero_fields_6(self.temp_Kx1, self.temp_Kx2, self.temp_Kx3,
            #                  self.temp_result, self.temp_vec, self.temp_Kz3)
            #self.spmv_6x_autotune(
            #    Kx1.data, Kx1.indices, Kx1.indptr,
            #    Kx2.data, Kx2.indices, Kx2.indptr,
            #    Kx3.data, Kx3.indices, Kx3.indptr,
            #    Kz1.data, Kz1.indices, Kz1.indptr,
            #    Kz2.data, Kz2.indices, Kz2.indptr,
            #    Kz3.data, Kz3.indices, Kz3.indptr,
            #    self.displ_x, self.displ_z,
            #    self.temp_Kx1, self.temp_Kx2, self.temp_Kx3,
            #    self.temp_result, self.temp_vec, self.temp_Kz3,
            #    self.strategy_6x, Kx1.nrows
            #    )
            
            # Fused update
            self.update_displ_terms_x_fused(self.temp_Kx1, self.temp_Kx2, self.temp_Kx3)
            self.update_displ_terms_z_fused(self.temp_result, self.temp_vec, self.temp_Kz3)
            
            # Step 2: Update auxiliary fields
            # Fused sparse matvec for auxiliary fields
            #self.sparse_matvec(Kx4.data, Kx4.indices, Kx4.indptr, 
            #                  self.displ_x, self.temp_Kx1, Kx4.nrows)
            #self.sparse_matvec(Kx5.data, Kx5.indices, Kx5.indptr, 
            #                  self.displ_x, self.temp_Kx2, Kx5.nrows)
            #self.sparse_matvec(Kz4.data, Kz4.indices, Kz4.indptr, 
            #                  self.displ_z, self.temp_result, Kz4.nrows)
            #self.sparse_matvec(Kz5.data, Kz5.indices, Kz5.indptr, 
            #                  self.displ_z, self.temp_vec, Kz5.nrows)
            self.sparse_matvec_4x_fused(
                Kx4.data, Kx4.indices, Kx4.indptr,
                Kx5.data, Kx5.indices, Kx5.indptr,
                Kz4.data, Kz4.indices, Kz4.indptr,
                Kz5.data, Kz5.indices, Kz5.indptr,
                self.displ_x, self.displ_z,
                self.temp_Kx1, self.temp_Kx2,
                self.temp_result, self.temp_vec,
                Kx4.nrows
            )
            
            # Clear result arrays for atomic operations
            #self.zero_fields_4(self.temp_Kx1, self.temp_Kx2,
            #                  self.temp_result, self.temp_vec)
            #self.spmv_4x_autotune(
            #    Kx4.data, Kx4.indices, Kx4.indptr,
            #    Kx5.data, Kx5.indices, Kx5.indptr,
            #    Kz4.data, Kz4.indices, Kz4.indptr,
            #    Kz5.data, Kz5.indices, Kz5.indptr,
            #    self.displ_x, self.displ_z,
            #    self.temp_Kx1, self.temp_Kx2,
            #    self.temp_result, self.temp_vec,
            #    self.strategy_4x, Kx4.nrows
            #)
            
            self.update_p_fields_fused(self.temp_Kx1, self.temp_Kx2, 
                                       self.temp_result, self.temp_vec)
            
            # Step 3: Update history and combine
            self.update_history_and_combine()
            
            # Step 4: Add sources (using batch source data)
            if moment_source_step is None:
                self._add_batch_sources(it, dt)
            else:
                moment_source_step(it)
            
            # Step 5: Compute derivatives
            if it >= 1:
                self.compute_velocity(dt)
            if it >= 2:
                self.compute_acceleration(dt)
            
            # Step 6: Record seismograms
            self._record_batch_seismograms(it)
            
            # Step 7: Save snapshot
            if enable_snapshot and it >= first_snapshot_step and it % snapshot_interval == 0:
                if snapshot_counter < self.num_snapshots:
                    self.save_snapshot(snapshot_counter, it)
                    snapshot_counter += 1
    
    @ti.kernel
    def _add_batch_sources(self, it: ti.i32, dt: float):
        """
        批量添加震源
        """
        for src_idx in range(self.num_sources):
            actual_time = it - self.source_time_delays[src_idx]
            if 0 <= actual_time < self.nt:
                node_idx = self.source_indices[src_idx]
                stf_val = self.source_stf[src_idx, actual_time]
                
                force_x = self.source_sin_angles[src_idx] * stf_val
                force_z = self.source_cos_angles[src_idx] * stf_val
                
                self._apply_nodal_force(node_idx, force_x, force_z)
    
    @ti.kernel
    def _record_batch_seismograms(self, it: ti.i32):
        """
        批量记录地震图
        """
        for rec_id in range(self.num_receivers):
            node_idx = self.receiver_indices[rec_id]
            self.seismograms_disp[rec_id, it] = self.displ_now[node_idx]
            
            # Velocity and acceleration may not be computed yet at early timesteps
            if it >= 1:
                self.seismograms_vel[rec_id, it] = self.velocity[node_idx]
            if it >= 2:
                self.seismograms_acc[rec_id, it] = self.acceleration[node_idx]
