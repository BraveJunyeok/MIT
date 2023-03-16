from flask import Flask, request , send_file, jsonify
from flask_cors import CORS
import openai
import requests
import io


import cv2
import numpy as np
from src.config import *
from src.dataset import CLASSES
import torch
import base64
# from PIL import Image // 이미지 저장 
import googletrans
import urllib.request

app = Flask(__name__)
CORS(app)

class_dict = {'apple': '사과', 'book': '책', 'bowtie': '보타이', 'candle': '촛대', 'cloud': '구름', 'cup': '컵',
'door': '문', 'envelope': '봉투', 'eyeglasses': '안경', 'guitar': '기타', 'hammer': '망치', 'hat': '모자',
'ice cream': '아이스크림', 'leaf': '나뭇잎', 'scissors': '가위', 'star': '별', 't-shirt': '티셔츠',
'pants': '바지', 'lightning': '번개', 'tree': '나무'}

keyword_key = []
story_key = []
translator = googletrans.Translator()

@app.route('/post_data', methods=['POST',"GET"])
def post_data():
    
    # HTTP POST 요청 데이터를 추출합니다.
    data = request.get_json()
    image_data = data.get('image', '')
    # base64 문자열로부터 이미지 데이터를 복원합니다.
    image_64 = base64.b64decode(image_data.split(',')[1])
    image_array = np.frombuffer(image_64, np.uint8)
    with open('/Users/kijun/Desktop/react_project/flask-server/image/canvas_image.png', 'wb') as f:
        f.write(image_64)

    image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
    _, _, _, alpha = cv2.split(image)
    image_gray = alpha
    
    # # PIL Image로 변환합니다.
    # pil_image = Image.fromarray(image)
    # # 이미지를 저장할 파일 경로와 파일 이름을 지정합니다.
    # save_path = '/Users/kijun/Desktop/react_project/flask-server/image/new_gray_image12345.png'
    # # 이미지를 저장합니다.
    # pil_image.save(save_path)


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
        model = torch.load("/Users/kijun/Desktop/react_project/flask-server/src/whole_model_quickdraw.txt")
    else:
        model = torch.load("/Users/kijun/Desktop/react_project/flask-server/src/whole_model_quickdraw.txt", map_location=lambda storage, loc: storage)
    model.eval()

    with torch.no_grad():
        logits = model(img_tensor)
        pred = torch.argmax(logits, dim=1).item()
        pred_class = CLASSES[pred]
        pred_class_kr = class_dict.get(pred_class, '알 수 없는 객체')  # 클래스 이름을 한글로 변환합니다.
        keyword_key.clear()
        keyword_key.append(pred_class)

    # print(keyword_key[0])
    print(pred_class,pred_class_kr)
    return {"prediction": pred_class_kr}


@app.route('/get_story', methods=['GET','POST'])
def get_story():
    # GPT-3 API 가져오는 코드
    openai.api_key = "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"  # OpenAI API Key를 입력해주세요
    model_engine = "text-davinci-002"  # GPT-3 엔진 모델을 선택해주세요
    prompt = f"Make a fairy tale with the keyword {keyword_key[0]} as cute and content that children will like."
    response = openai.Completion.create(engine=model_engine, prompt=prompt, max_tokens=100)
    story = response.choices[0].text
    story = story.replace("-","")
    story = story.replace(".",". ")
    story_key.clear()
    story_key.append(story)
    print(story)
    # 스토리 한글 변환
    story_kr = translator.translate(story, dest='ko')
    # print(story_kr)
    return jsonify({"story": story_kr.text})


@app.route('/get_data', methods=['GET','POST'])
def get_data():
        # dall-e api 가져오는 코드
        openai.api_key = "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"
        openai.Model.list()
        response = openai.Image.create(
            prompt=f"very cute fairy-tail with{keyword_key[0]}",
            n=1,
            size = "512x512"
        )
        # Please make a cute image that kids will like with the keyword {story_key[0]}.
        if response and response.data and response.data[0].url:
            url = response.data[0].url
            image_data = requests.get(url).content

            # 이미지 데이터를 BytesIO 객체로 변환합니다.
            image_io = io.BytesIO(image_data)
            
            # 이미지를 저장할 파일 경로와 파일 이름을 지정합니다.
            save_path = '/Users/kijun/Desktop/react_project/flask-server/image/new_image.png'
            
            # 이미지를 파일로 저장합니다.
            with open(save_path, 'wb') as f:
                f.write(image_data)

        return send_file(image_io, mimetype='image/png', as_attachment=False, attachment_filename='new_image.png')

@app.route('/get_voice', methods=['GET','POST'])
def get_voice():
    client_id = "ginth2haq7"
    client_secret = "HMkOFIL5djQDjYuxpJ8nNSi6nV65s39cQc3PHHob"
    story_kr = translator.translate(story_key[0], dest='ko')
    print(story_kr)
    encText = urllib.parse.quote(story_kr.text,encoding="UTF-8")
    data = f"speaker=ngoeun&volume=0&speed=0&pitch=0&format=mp3&text=" + encText
    url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
    request = urllib.request.Request(url)
    request.add_header("X-NCP-APIGW-API-KEY-ID",client_id)
    request.add_header("X-NCP-APIGW-API-KEY",client_secret)
    response = urllib.request.urlopen(request, data=data.encode('utf-8'))
    rescode = response.getcode()
    if(rescode==200):
        print("TTS mp3 저장")
        response_body = response.read()
        with open('/Users/kijun/Desktop/react_project/flask-server/image/1111.mp3', 'wb') as f:
            f.write(response_body)
    else:
        print("Error Code:" + rescode)

    return send_file("/Users/kijun/Desktop/react_project/flask-server/image/1111.mp3", mimetype='audio/mpeg') 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
