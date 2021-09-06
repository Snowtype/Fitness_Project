'''
OpenPose ToolKit, 원본인용코드, 라이브러리 내용 참조 링크들:
<OpenPose 관련문서>
https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/

<OpenPose GitHub>
https://github.com/CMU-Perceptual-Computing-Lab/openpose

<Kaggle OpenPose - 딥러닝 진행할 caffemodel>
https://www.kaggle.com/changethetuneman/openpose-model/version/1?select=pose_iter_584000.caffemodel

<구글링 블로그 OpenPose 화면 출력코드>
https://m.blog.naver.com/rhrkdfus/221531159811
https://hanryang1125.tistory.com/2

###발표 시연용###
동영상을 찍고 캡쳐된 이미지를 사용

-8월 14일 ~ 15일 호환 완료
이미지 파일로 저장 확인 가능하게 완성
동영상 촬영해서 프레임 얻고 저장한 후 불러오기
'''
import cv2  #OpenPose 사용과
import csv  #파일 입출력을 위한
import pandas as pd    #glob의 glob()을 사용해서 모든 파일에 작업하기
import glob

#directory 에서 모든 이미지 가져오기 위한 os 라이브러리 호출
import os
import numpy as np

# 구글링, 인용 원본 코드 출처: https://deftkang.tistory.com/182
# 동영상에서 이미지 프레임 추출
# IMG_6305.mov
# sample_video.mp4
vidcap = cv2.VideoCapture('IMG_6305.mov')
count = 0

while (vidcap.isOpened()):
    ret, image = vidcap.read()
    # image = cv2.resize(image, (960, 540))   #이미지 사이즈 변경 필요할때

    if (int(vidcap.get(1)) % 20 == 0):  # 30 프레임당 한개의 이미지
        print('Frame #: ' + str(int(vidcap.get(1))))

        # 이미지프레임 저장 경로 생성
        if not os.path.exists("captured_image_list"):
            os.mkdir("captured_image_list")

        # 이미지프레임 저장
        cv2.imwrite('captured_image_list/capture%d.jpg' % count, image)
        print("'capture%d.jpg' saved" % count)
        count += 1

    if count == 16:  # 열장 찍으면 나옴
        vidcap.release()
vidcap.release()
