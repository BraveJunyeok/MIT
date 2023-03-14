import cv2
import numpy as np
from src.config import *
from src.dataset import CLASSES
import torch



def main():
    # 모델 불러오기(받아온 폴더로 경로 설정해주세요!! 절대경로로 하면 확실함)
    if torch.cuda.is_available():
        model = torch.load("C:/Users/June/Downloads/converted_keras/PROJECK/3project/react_project/react_project/flask-server/src/whole_model_quickdraw.txt")
    else:
        model = torch.load("C:/Users/June/Downloads/converted_keras/PROJECK/3project/react_project/react_project/flask-server/src/whole_model_quickdraw.txt", map_location=lambda storage, loc: storage)
    model.eval()
    #이미지를 저장할 numpy배열 만들기
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.namedWindow("Canvas")
    global ix, iy, is_drawing
    is_drawing = False

    #마우스 이벤트 콜백 함수!
    def paint_draw(event, x, y, flags, param):
        global ix, iy, is_drawing
        if event == cv2.EVENT_LBUTTONDOWN: #마우스 왼쪽 버튼을 누를 때
            is_drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE: #마우스가 움직일 때
            if is_drawing == True: #is_drawing 변수가 True일 때만(마우스 왼쪽 버튼을 누르고 있을 때만)
                cv2.line(image, (ix, iy), (x, y), WHITE_RGB, 5)
                #현재 위치와 이전 위치를 연결하는 선을 그리고
                ix = x
                iy = y
                #현재 위치의 좌표를 ix, iy 변수에 저장
        elif event == cv2.EVENT_LBUTTONUP: #마우스 왼쪽 버튼을 놓을 때
            is_drawing = False
            cv2.line(image, (ix, iy), (x, y), WHITE_RGB, 5)
            ix = x
            iy = y
        return x, y

    cv2.setMouseCallback('Canvas', paint_draw) 
    #마우스 이벤트를 처리할 콜백 함수 paint_draw를 Canvas 창에 등록

    while (1): 
        cv2.imshow('Canvas', 255 - image)#위의 콜백 함수로 사용자가 캔버스 창에서 그린 이미지를 표시
        key = cv2.waitKey(10) #10밀리초마다 키 입력 대기
        if key == ord(" "): #스페이스바(" ") 눌렀을 때 이미지 처리하고 분류 모델에 입력 (다른 키로 변경해도됨)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 이미지를 그레이스케일로 변환해줌
            ys, xs = np.nonzero(image) # 이미지에서 0이 아닌 픽셀의 위치 찾기
            min_y = np.min(ys)
            max_y = np.max(ys)
            min_x = np.min(xs)
            max_x = np.max(xs) # 이미지에서 0이 아닌 최소/최대 y/x 좌표 찾기
            image = image[min_y:max_y, min_x: max_x] # 이미지에서 0이 아닌 부분만 잘라내기
            
            image = cv2.resize(image, (28, 28)) # 이미지를 28*28크기로 조정 (모델 입력값에 맞게)
            image = np.array(image, dtype=np.float32)[None, None, :, :] # 이미지를 numpy 배열로 변환
            image = torch.from_numpy(image) # 이미지를 pytorch tensor로 변환
            logits = model(image) # 분류 모델에 입력 이미지를 전달해서 결과 얻기
            print(CLASSES[torch.argmax(logits[0])]) # 분류 결과 출력(배열 형태로 나오므로 그 중 제일 확률이 높은 인덱스의 class 값을 출력하는듯)
            image = np.zeros((480, 640, 3), dtype=np.uint8) # 새 이미지 만들게 초기화해주는건듯
            ix = -1
            iy = -1 #마우스포인터 초기화







if __name__ == '__main__':
    main()
