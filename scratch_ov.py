from ultralytics import YOLO
model = YOLO('yolov8n_openvino_model/', task='detect')
res = model.predict('dataset/blank.jpg', device='GPU')
print("PASSED")
