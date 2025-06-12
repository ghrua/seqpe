from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
def conver_to_numpy(input_tensor):
    if input_tensor.device is not 'cpu':
        return input_tensor.detach().cpu().numpy()
    return input_tensor
def Tsne(input_tensor):
    tsne = TSNE(n_components=3, random_state=42)
    tsne_results = tsne.fit_transform(input_tensor.detach().cpu().numpy())
    
    # 创建 Plotly 图像
    fig = px.scatter_3d(
        x=tsne_results[:, 0], 
        y=tsne_results[:, 1], 
        z=tsne_results[:, 2],
        title='3D TSNE',
        labels={'x': 'TSNE Component 1', 'y': 'TSNE Component 2', 'z': 'TSNE Component 3'}
    )
    annotations = []
    for i in range(0, len(tsne_results), 32):
        annotations.append(
            go.Scatter3d(
                x=[tsne_results[i, 0]],
                y=[tsne_results[i, 1]],
                z=[tsne_results[i, 2]],
                mode='text',
                text=[str(i)],
                textposition='top center'
            )
        )
    for annotation in annotations:
        fig.add_trace(annotation)
    # 添加交互性
    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
    save_path = 'analysis_result/tsne_visualization.html'
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs('analysis_result')    
    fig.write_html(save_path)
    print(f"Interactive 3D TSNE visualization saved to: {save_path}")    

def l2_norm_anl(input_tensor, save_path='l2_norm_plot.png'):
    if input_tensor.dim() != 2:
        raise ValueError("input_tensor must have exactly 2 dimensions (n, d).")    
    # 计算每一行的 L2 范数
    norms = torch.norm(input_tensor, p=2, dim=1).detach().cpu()
    input_tensor = input_tensor.detach().cpu().numpy()
    # 创建保存目录（如果不存在）
    os.makedirs('analysis_result', exist_ok=True)
    save_path = 'analysis_result/' + save_path    
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(norms)), norms.numpy(), marker='o')
    plt.title('L2 Norm of Each Sample')
    plt.xlabel('Sample Index')
    plt.ylabel('L2 Norm')
    plt.grid(True)
    
    # 保存图像到指定路径
    plt.savefig(save_path)
    plt.close()
    
    print(f"Scatter plot saved at: {save_path}")
    
    return norms

def value_count(input_tensor, save_path='value_count.png'):
    if input_tensor.dim() != 2:
        raise ValueError("input_tensor must have exactly 2 dimensions (n, d).")    
    # 将张量展平为一维
    flattened_tensor = input_tensor.flatten()
    
    # 将张量元素转换为 NumPy 数组
    values = flattened_tensor.detach().cpu().numpy()
    
    # 定义区间（细粒度 0.1）
    bins = np.arange(np.floor(values.min()), np.ceil(values.max()) + 0.1, 0.1)
    hist, bin_edges = np.histogram(values, bins=bins)
    
    # 统计结果存储到字典中
    value_distribution = {f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}": hist[i] for i in range(len(hist))}
    
    # 创建保存目录（如果不存在）
    os.makedirs('analysis_result', exist_ok=True)
    save_path = 'analysis_result/' + save_path
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 绘制条形图
    plt.figure(figsize=(12, 6))
    plt.bar(bin_edges[:-1], hist, width=0.1, edgecolor='black', align='edge')
    plt.title('Value Distribution in Tensor')
    plt.xlabel('Value Ranges')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
import pdb
class analysis():
    def __init__(self):
        self.option_dict = {'l2_norm_anl' : l2_norm_anl, 'tsne': Tsne, 'value_count': value_count}

    def __call__(self, options, *args):
        for option in options:
            self.option_dict.get(option)(*args)
