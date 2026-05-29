import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
import onnxruntime
from facenet_pytorch import InceptionResnetV1

def main():
    print("Loading FaceNet model...")
    model = InceptionResnetV1(pretrained='vggface2').eval()
    
    # Create dummy input: FaceNet takes (batch, 3, 160, 160)
    dummy_input = torch.randn(1, 3, 160, 160)
    onnx_path = 'facenet.onnx'
    
    print("Exporting to ONNX... (this may take 1-2 minutes)")
    try:
        # Legacy JIT-based export (more stable for ResNet architectures)
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path, 
            input_names=['input'], 
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
    except Exception as e:
        print(f"Export failed: {e}")
        return

    print("Export successful! Testing ONNX Runtime with DirectML...")
    try:
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        sess = onnxruntime.InferenceSession(onnx_path, providers=providers)
        
        # Test inference
        out = sess.run(None, {'input': dummy_input.numpy()})
        print(f"ONNX inference successful! Output shape: {out[0].shape}")
    except Exception as e:
        print(f"ONNX Runtime test failed: {e}")

if __name__ == '__main__':
    main()
