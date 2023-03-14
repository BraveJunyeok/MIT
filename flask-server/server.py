from flask import Flask, request
from flask_cors import CORS
import openai
import requests

import cv2
import numpy as np
from src.config import *
from src.dataset import CLASSES
import torch
import base64
from PIL import Image

app = Flask(__name__)


CORS(app)
class_dict = {'apple': '사과', 'book': '책', 'bowtie': '보타이', 'candle': '촛대', 'cloud': '구름', 'cup': '컵',
'door': '문', 'envelope': '봉투', 'eyeglasses': '안경', 'guitar': '기타', 'hammer': '망치', 'hat': '모자',
'ice cream': '아이스크림', 'leaf': '나뭇잎', 'scissors': '가위', 'star': '별', 't-shirt': '티셔츠',
'pants': '바지', 'lightning': '번개', 'tree': '나무'}


@app.route('/post_data', methods=['POST',"GET"])
def post_data():
    
    # HTTP POST 요청 데이터를 추출합니다.
    data = request.get_json()
    image_data = data.get('image', '')
    # base64 문자열로부터 이미지 데이터를 복원합니다.
    image_64 = base64.b64decode(image_data.split(',')[1])
    with open('react_project/react_project/flask-server/image/canvas_image.png', 'wb') as f:
        f.write(image_64)

    image = cv2.imread('react_project/react_project/flask-server/image/canvas_image.png', cv2.IMREAD_UNCHANGED)
    _, _, _, alpha = cv2.split(image)
    image_gray = alpha
    
    # PIL Image로 변환합니다.
    pil_image = Image.fromarray(image)
    # 이미지를 저장할 파일 경로와 파일 이름을 지정합니다.
    save_path = 'react_project/react_project/flask-server/image/new_gray_image12345.png'
    # 이미지를 저장합니다.
    pil_image.save(save_path)


    # 여기서 부터 모델 코드 ------------------------------------------------------
    # 이미지를 28*28 크기로 조정합니다.
    img_resized = cv2.resize(image_gray, (28, 28))
    
    # 이미지를 numpy 배열로 변환합니다.
    img_array = np.array(img_resized, dtype=np.float32)

    # 이미지를 4차원 입력으로 만듭니다.
    img_tensor = np.expand_dims(img_array, axis=0)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    # 이미지를 pytorch tensor로 변환합니다.
    img_tensor = torch.from_numpy(img_tensor)


    if torch.cuda.is_available():
        model = torch.load("react_project/react_project/flask-server/src/whole_model_quickdraw.txt")
    else:
        model = torch.load("react_project/react_project/flask-server/src/whole_model_quickdraw.txt", map_location=lambda storage, loc: storage)
    model.eval()

    with torch.no_grad():
        logits = model(img_tensor)
        pred = torch.argmax(logits, dim=1).item()
        pred_class = CLASSES[pred]
        pred_class_kr = class_dict.get(pred_class, '알 수 없는 객체')  # 클래스 이름을 한글로 변환합니다.

        # dall-e api 가져오는 코드
        openai.api_key = "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"
        openai.Model.list()
        response = openai.Image.create(
            prompt=f"Cute fairy tale with {pred_class}",
            n=4,
            size = "256x256"
        )

        
        if response and response.data and response.data[0].url:
            url = response.data[0].url
            image_data = requests.get(url).content
            
            # 이미지를 저장할 파일 경로와 파일 이름을 지정합니다.
            save_path = 'react_project/react_project/flask-server/image/new_image.png'
            
            # 이미지를 파일로 저장합니다.
            with open(save_path, 'wb') as f:
                f.write(image_data)

    print(pred_class,pred_class_kr)
    return {"prediction": pred_class_kr}
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4450, debug=True)
