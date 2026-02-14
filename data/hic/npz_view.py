import numpy as np
import matplotlib.pyplot as plt
import os

# ==================== 配置参数（请根据实际情况修改） ====================
# 基础输入目录（包含所有样本子目录）
BASE_INPUT_DIR = "/home/bingxing2/gpuuser1085/hicdata/hic_trans"
# 样本相对路径（可包含子目录，例如 "GM12878/4DNFIXP4QG5B_chr1"）
SAMPLE_PATH = "GM12878/4DNFIXP4QG5B_chr1"
# 输出基础目录（热图将保存在此目录下，自动保持与 SAMPLE_PATH 相同的子目录结构）
BASE_OUTPUT_DIR = "/home/bingxing2/gpuuser1085/hicdata/hic_view"
# 目标区间（bp）
CHROMOSOME = "chr1"          # 染色体名称（仅用于显示和验证）
START_BP = 1_000_000         # 起始坐标
END_BP   = 3_000_000         # 终止坐标
RESOLUTION = 10000            # 分辨率（bp/bin）

# 热图绘制参数
CMAP = "Reds"                 # 颜色映射
VMAX_PERCENTILE = 95          # 用于设置颜色条上限的百分位数（基于非零值）
DPI = 300                      # 输出图片清晰度（每英寸点数），提高此值可增强清晰度

# 增强对比度选项（若关闭则直接绘制原始接触矩阵）
ENHANCE_CONTRAST = True        # 是否开启增强对比度（阈值过滤 + 对数变换）
LOG_TRANSFORM = True           # 是否应用 log(1+x) 变换（仅在 ENHANCE_CONTRAST=True 时生效）
THRESHOLD_QUANTILE = 0.05      # 阈值分位数：将低于此分位数的值置零（基于原始矩阵非零值）
# =====================================================================

def load_hic_diag(npz_path):
    """加载NPZ文件，返回对角线字典（键为字符串，值为一维数组）"""
    print(f"Loading Hi-C data from {npz_path}")
    data = np.load(npz_path)
    # 转换为普通字典以便操作（npz对象只读）
    return dict(data)

def extract_sub_diagonals(diag_data, start_bin, end_bin):
    """
    从全局对角线数据中提取指定bin区间的对角线片段。
    diag_data : 全局对角线字典，键为对角线索引（字符串），值为一维数组
    start_bin, end_bin : 要提取的bin区间 [start_bin, end_bin)
    返回字典，键为整数对角线索引，值为对应的一维片段
    """
    square_len = end_bin - start_bin
    sub_diag = {}
    for diag_i in range(square_len):
        # 上对角线（正）
        key_pos = str(diag_i)
        arr_pos = diag_data[key_pos]
        # 提取长度应为 square_len - diag_i
        needed_len = square_len - diag_i
        if start_bin + needed_len > len(arr_pos):
            raise ValueError(f"Diagonal {diag_i} out of bounds: need up to {start_bin + needed_len}, length {len(arr_pos)}")
        sub_diag[diag_i] = arr_pos[start_bin:start_bin + needed_len].copy()

        # 下对角线（负）
        key_neg = str(-diag_i)
        arr_neg = diag_data[key_neg]
        if start_bin + needed_len > len(arr_neg):
            raise ValueError(f"Diagonal {-diag_i} out of bounds")
        sub_diag[-diag_i] = arr_neg[start_bin:start_bin + needed_len].copy()
    return sub_diag

def diag_to_matrix(sub_diag, square_len):
    """
    将对角线片段字典重建为完整的方阵。
    sub_diag : 键为整数对角线索引，值为一维数组（长度 = square_len - |diag|）
    square_len : 方阵边长
    返回二维numpy数组，形状 (square_len, square_len)
    """
    mat = np.zeros((square_len, square_len), dtype=float)
    for d, arr in sub_diag.items():
        if d >= 0:
            # 填充上对角线 (行 i, 列 i+d)
            for i in range(len(arr)):
                mat[i, i + d] = arr[i]
        else:
            # 填充下对角线 (行 i-d, 列 i)  因为d为负，所以 i-d > i
            for i in range(len(arr)):
                mat[i - d, i] = arr[i]
    return mat

def plot_heatmap(matrix, output_path, cmap='Reds', vmax_percentile=95, dpi=300):
    """
    绘制原始数据的热图并保存。
    matrix : 2D numpy数组（原始接触频率或经增强处理后的矩阵）
    output_path : 保存路径
    cmap : 颜色映射
    vmax_percentile : 用于设定颜色条上限的百分位数（基于非零值）
    dpi : 输出图片清晰度（每英寸点数）
    """
    # 计算vmax：使用非零值的百分位数
    flat = matrix.flatten()
    nonzero = flat[flat > 0]
    if len(nonzero) > 0:
        vmax = np.percentile(nonzero, vmax_percentile)
    else:
        vmax = matrix.max()

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap=cmap, aspect='equal', origin='lower',
               extent=[0, matrix.shape[1], 0, matrix.shape[0]],
               vmax=vmax)
    plt.colorbar(label='Contact frequency' + (' (log1p)' if ENHANCE_CONTRAST and LOG_TRANSFORM else ''))
    plt.xlabel('Bin index (relative)')
    plt.ylabel('Bin index (relative)')
    title = f'Hi-C contact matrix ({matrix.shape[0]}x{matrix.shape[1]})'
    if ENHANCE_CONTRAST:
        title += ' [Enhanced]'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {output_path} (dpi={dpi})")

def main():
    # 构建输入文件完整路径
    input_path = os.path.join(BASE_INPUT_DIR, SAMPLE_PATH + ".npz")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # 构建输出文件路径：保持与 SAMPLE_PATH 相同的子目录结构，文件名添加 .png 后缀
    output_path = os.path.join(BASE_OUTPUT_DIR, SAMPLE_PATH + ".png")
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 加载数据
    diag_data = load_hic_diag(input_path)

    # 获取总bin数（主对角线长度）
    if '0' not in diag_data:
        raise KeyError("Main diagonal '0' not found in NPZ file.")
    total_bins = len(diag_data['0'])
    print(f"Total bins for {CHROMOSOME}: {total_bins}")

    # 坐标转bin索引
    start_bin = START_BP // RESOLUTION
    end_bin   = END_BP   // RESOLUTION
    if end_bin > total_bins:
        print(f"Warning: end_bin {end_bin} exceeds total bins {total_bins}, truncating.")
        end_bin = total_bins
    if start_bin >= end_bin:
        raise ValueError(f"Invalid interval: start_bin {start_bin} >= end_bin {end_bin}")

    square_len = end_bin - start_bin
    print(f"Extracting region: bins [{start_bin}, {end_bin}) (length {square_len})")

    # 提取对角线片段并重建矩阵
    sub_diag = extract_sub_diagonals(diag_data, start_bin, end_bin)
    matrix = diag_to_matrix(sub_diag, square_len)
    print(f"Reconstructed matrix shape: {matrix.shape}")

    # 根据增强对比度选项对矩阵进行预处理
    if ENHANCE_CONTRAST:
        print("Applying contrast enhancement: threshold + log transform")
        # 阈值处理：将低于指定分位数的值置零（基于原始矩阵的非零值）
        flat = matrix.flatten()
        nonzero = flat[flat > 0]
        if len(nonzero) > 0:
            thresh = np.percentile(nonzero, THRESHOLD_QUANTILE * 100)
            matrix = np.where(matrix < thresh, 0, matrix)
            print(f"Threshold set to {thresh:.2f} (quantile {THRESHOLD_QUANTILE})")
        # 对数变换
        if LOG_TRANSFORM:
            matrix = np.log1p(matrix)
            print("Applied log(1+x) transform")
    else:
        print("Using raw contact matrix (no enhancement)")

    # 绘制热图
    plot_heatmap(matrix, output_path,
                 cmap=CMAP,
                 vmax_percentile=VMAX_PERCENTILE,
                 dpi=DPI)

    # 如需同时保存矩阵为.npy文件，取消下面两行注释
    # matrix_output = output_path.replace('.png', '.npy')
    # np.save(matrix_output, matrix)
    # print(f"Matrix saved to {matrix_output}")

if __name__ == "__main__":
    main()
