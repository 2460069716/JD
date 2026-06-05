import torch
import torch.onnx
from models.simvp_model import SimVP_Model 

CHECKPOINT_PATH = "best.ckpt"
ONNX_SAVE_PATH = "simvp_convnext_best128.onnx"

SHAPE_IN = (12, 4, 128, 128) 

CONFIG = {
    'model_type': 'convnext',
    'spatio_kernel_enc': 3,
    'spatio_kernel_dec': 3,
    'hid_S': 32,
    'hid_T': 256,
    'N_T': 8,
    'N_S': 4,
    'drop_path': 0.1
}

def main():
    model = SimVP_Model(in_shape=SHAPE_IN, **CONFIG)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    clean_state_dict = {k.replace('model.', '').replace('net.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=True)

    model.eval()

    print("正在导出ONNX...")
    dummy_input = torch.randn(1, *SHAPE_IN, device="cpu")

    torch.onnx.export(
        model, 
        dummy_input, 
        ONNX_SAVE_PATH,
        export_params=True, 
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_frames'], 
        output_names=['predicted_frames'],
        dynamic_axes={
            'input_frames': {0: 'batch_size'},
            'predicted_frames': {0: 'batch_size'}
        }
    )

    print(f"模型已成功导出为: {ONNX_SAVE_PATH}")

if __name__ == "__main__":
    main()
