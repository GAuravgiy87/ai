import openvino as ov
import sys

def check_ov():
    print("--- OpenVINO Hardware Diagnostic v2 ---")
    try:
        core = ov.Core()
        devices = core.available_devices
        print(f"Core detected devices: {devices}\n")
        
        for device in devices:
            print(f"Device: {device}")
            try:
                name = core.get_property(device, "FULL_DEVICE_NAME")
                print(f"  - Name: {name}")
            except: pass
            
            try:
                # Check for device type (Integrated vs Discrete)
                type_val = core.get_property(device, "DEVICE_TYPE")
                print(f"  - Type: {type_val}")
            except: pass
            
            print("-" * 30)
                
        if "GPU" not in [d[:3] for d in devices]:
            print("\n❌ WARNING: No GPU detected.")
        elif len([d for d in devices if "GPU" in d]) == 1:
            print("\n⚠️  NOTICE: Only one GPU detected. If you have both Intel and AMD, ensure both drivers (intel-opencl-icd and mesa-opencl-icd) are installed.")
        else:
            print("\n✅ SUCCESS: Multiple GPUs found! MULTI-GPU mode is ready.")
            
    except Exception as e:
        print(f"Error initializing OpenVINO Core: {e}")

if __name__ == "__main__":
    check_ov()
