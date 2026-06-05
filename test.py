import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

NPZ_FILE_PATH = "../../data/era5/test_128_128.npz"
ONNX_FILE_PATH = "simvp_convnext_best128.onnx"

VARS = ["msl", "u10", "v10", "t2m"]

def calculate_metrics(true_frames, pred_frames):
    mse = np.mean((true_frames - pred_frames) ** 2)
    mae = np.mean(np.abs(true_frames - pred_frames))
    return mse, mae

def visualize_comparison(input_frames, true_frames, predicted_frames, channel_idx=3, var_name="t2m"):
    print(f"绘制 '{var_name}' 的评估对比图...")
    
    in_viz = input_frames[0, :, channel_idx, :, :]
    true_viz = true_frames[0, :, channel_idx, :, :]
    pred_viz = predicted_frames[0, :, channel_idx, :, :] 
    
    num_frames = in_viz.shape[0]
    fig, axes = plt.subplots(3, num_frames, figsize=(num_frames * 2, 6))
    
    cmap = 'coolwarm' if var_name == 't2m' else 'viridis'
    
    vmin = min(in_viz.min(), true_viz.min(), pred_viz.min())
    vmax = max(in_viz.max(), true_viz.max(), pred_viz.max())

    for i in range(num_frames):
        axes[0, i].imshow(in_viz[i], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Input T={i+1}")
        
        axes[1, i].imshow(true_viz[i], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, i].axis('off')
        axes[1, i].set_title(f"True T={i+13}")
        
        axes[2, i].imshow(pred_viz[i], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[2, i].axis('off')
        axes[2, i].set_title(f"Pred T={i+13}")
        
    plt.tight_layout()
    save_path = f"evaluation_result_{var_name}.png"
    plt.savefig(save_path, dpi=150)
    print(f"对比图保存为: {save_path}")
    plt.show()

def main():
    dataset = np.load(NPZ_FILE_PATH, allow_pickle=True)
    
    raw_data = dataset['data']
    stats = dataset['stats'].item()
    
    normalized_data = np.zeros_like(raw_data, dtype=np.float32)
    for i, var_name in enumerate(VARS):
        d_min = stats[var_name]['min']
        d_max = stats[var_name]['max']
        normalized_data[i] = (raw_data[i] - d_min) / (d_max - d_min)
        
    processed_data = np.transpose(normalized_data, (1, 0, 2, 3))
    
    start_idx = 100
    input_frames = processed_data[start_idx : start_idx+12, :, :, :]
    true_frames = processed_data[start_idx+12 : start_idx+24, :, :, :]
    
    input_tensor = np.expand_dims(input_frames, axis=0)
    true_tensor = np.expand_dims(true_frames, axis=0)

    print(f"进行ONNX推理...")
    session = ort.InferenceSession(ONNX_FILE_PATH)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    predicted_tensor = outputs[0]
    
    mse, mae = calculate_metrics(true_tensor, predicted_tensor)
    print("模型定量评估结果:")
    print(f"MSE: {mse:.6f}") 
    print(f"MAE: {mae:.6f}")
    
    #在var_name配置绘图变量
    visualize_comparison(input_tensor, true_tensor, predicted_tensor, channel_idx=3, var_name="t2m")

if __name__ == "__main__":
    main()
