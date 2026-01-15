import requests
import json

# Replace with your server's Public IP
SERVER_IP = "209.121.195.118"
PORT = "13024"  # <--- This is the public port mapped to your internal port 8000
URL = f"http://{SERVER_IP}:{PORT}/predict"

# Sample Case Data
case_text = """
Ngày 30 tháng 9 năm 2025, tại trụ sở Tòa án N dân khu vực 2- Thành phố
Hồ Chí Minh xét xử sơ thẩm trực tiếp công khai vụ án hình sự đối với bị cáo:
Đồng Quang H, sinh năm 1999.
Nội dung: Bị cáo lấy trộm 01 điện thoại iPhone 15 Pro Max và 01 iPhone 14 Pro Max.
Tổng trị giá tài sản là 35.900.000 đồng.
Bị cáo đã khai nhận toàn bộ hành vi.
"""

payload = {
    "case_content": case_text,
    "role": "neutral"  # Options: "neutral", "defense", "victim"
}

print(f"🚀 Sending request to {URL}...")

try:
    response = requests.post(URL, json=payload, timeout=60) # 60s timeout for AI processing
    
    if response.status_code == 200:
        print("\n✅ SUCCESS! Server Response:\n")
        data = response.json()
        print(data["result"])
    else:
        print(f"\n❌ Server Error (Status {response.status_code}):")
        print(response.text)

except Exception as e:
    print(f"\n❌ Connection Failed: {e}")
    print("Check if Port 8000 is open in your GPU provider settings.")