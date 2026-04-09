import openvino as ov
import sys

def check_ov():
    print("--- OpenVINO Diagnostic ---")
    try:
        core = ov.Core()
        devices = core.available_devices
        print(f"Available Devices: {devices}")
        
        for device in devices:
            try:
                full_name = core.get_property(device, "FULL_DEVICE_NAME")
                print(f"  - {device}: {full_name}")
            except:
                print(f"  - {device}: (No full name available)")
                
        if "GPU" not in [d[:3] for d in devices]:
            print("\n❌ WARNING: No GPU detected by OpenVINO.")
            print("If you have an Intel iGPU, install: sudo apt-get install -y intel-opencl-icd")
            print("If you have an AMD GPU, install: sudo apt-get install -y mesa-opencl-icd")
            print("If you are in a Virtual Machine (VM), ensure GPU Passthrough is enabled.")
            print("If you are in WSL2, ensure you have the latest Windows GPU drivers installed.")
            
    except Exception as e:
        print(f"Error initializing OpenVINO: {e}")

if __name__ == "__main__":
    check_ov()
