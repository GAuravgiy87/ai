import requests

def add_test_camera():
    url = "http://127.0.0.1:8000/api/add_camera"
    data = {
        "camera_id": "TEST_CAM",
        "camera_type": "webcam", # Use webcam type to avoid prober for now
        "source": r"D:\test\AI-VIGILANCE\recordings\2026-04-10\DEI_Gate_5\rec_DEI_Gate_5_113343.mp4"
    }
    try:
        # Need to login first or use a session
        session = requests.Session()
        # Login (if credentials are known, usually deiobject/test@123)
        login_res = session.post("http://127.0.0.1:8000/api/login", data={"username": "deiobject", "password": "test@123"})
        print(f"Login status: {login_res.status_code}")
        
        res = session.post(url, data=data)
        print(f"Add camera status: {res.status_code}")
        print(f"Response: {res.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    add_test_camera()
