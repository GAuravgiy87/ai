import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import onnxruntime
from facenet_pytorch import InceptionResnetV1

def main():
    model = InceptionResnetV1(pretrained='vggface2').eval()
    dummy_input = torch.randn(1, 3, 160, 160)
    onnx_path = 'facenet_opset12.onnx'
    
    print("Exporting to ONNX with opset 12...")
    try:
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path, 
            opset_version=12,
            input_names=['input'], 
            output_names=['output']
        )
    except Exception as e:
        print(f"Export failed: {e}")
        return

    print("Testing ONNX Runtime with DirectML...")
    try:
        providers = ['DmlExecutionProvider']
        sess = onnxruntime.InferenceSession(onnx_path, providers=providers)
        out = sess.run(None, {'input': dummy_input.numpy()})
        print(f"ONNX inference successful! Output shape: {out[0].shape}")
    except Exception as e:
        print(f"ONNX Runtime test failed: {e}")

if __name__ == '__main__':
    main()
