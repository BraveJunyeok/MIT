
        # dall-e api 가져오는 코드
        # openai.api_key = "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"
        # openai.Model.list()
        # response = openai.Image.create(
        #     prompt=f"Cute fairy tale with {pred_class}",
        #     n=4,
        #     size = "256x256"
        # )

        
        # if response and response.data and response.data[0].url:
        #     url = response.data[0].url
        #     image_data = requests.get(url).content
            
        #     # 이미지를 저장할 파일 경로와 파일 이름을 지정합니다.
        #     save_path = '/Users/kijun/Desktop/react_project/flask-server/image/new_image.png'
            
        #     # 이미지를 파일로 저장합니다.
        #     with open(save_path, 'wb') as f:
        #         f.write(image_data)
