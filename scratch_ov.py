from ultralytics import YOLO
print("Exporting model...")
YOLO('yolov8n.pt').export(format='openvino')
print("Testing predict...")
model = YOLO('yolov8n_openvino_model/', task='detect')
try:
    res = model.predict('https://ultralytics.com/images/bus.jpg', device='CPU')
    print("SUCCESS CPU")
except Exception as e:
    print(f"FAILED CPU: {e}")
try:
    res = model.predict('https://ultralytics.com/images/bus.jpg', device='GPU')
    print("SUCCESS GPU")
except Exception as e:
    print(f"FAILED GPU: {e}")
