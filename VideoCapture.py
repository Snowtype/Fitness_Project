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

img_path_list = []  # 이미지 경로들 저장용

##################################################################

global V  # 파일명이 들어갈 장소
V = []
global V_s  # 임시저장소
V_s = []
# 동영상에서 캡쳐된 프레임들
root_dir = 'C:/피트니스 자세 이미지/Training/Day04_200924_F/1/C/033-1-1-21-Z5_C'
# 디버깅용 %debug (get_images)

total_file_list = os.listdir(root_dir)  # 파일 읽어오기


def get_images():
    global img_path_list  # 이미지들만 추출해서 리스트에 추가
    possible_img_extension = ['.jpg']

    for (root, dir, files) in os.walk(root_dir):
        print(len(files))
        for i in range(0, len(files)):  # 조사결과 32번 사진까지 있는 것 확인
            if i % 2 == 1:  # 여기선 모든 사진 가져올 예정 고로 이 코드는 사용하지 않음
                if len(files) > 0:
                    for file_name in files:
                        if file_name.endswith(str(i) + ".jpg"):  # [1]: 홀수사진들만 추출

                            img_path = root + '/' + file_name
                            # img_path = img_path.replace('\\', '/') # \는 /로 나타내야함
                            V_s.append(file_name)
                            img_path_list.append(img_path)


get_images()  # 함수 호출: 디렉토리 안에 있는 모든 이미지파일들만 추출

# 중복값 제거 하고 다시 원래 리스트에 삽입
result = []
for value in img_path_list:
    if value not in result:
        result.append(value)
img_path_list = result  # 다시 전역변수로 선언 (바깥쪽)
img_num = len(img_path_list)  # 이미지 트래킹될 모든 파일들의 숫자, 전역변수

result = []
for value in V_s:
    if value not in result:
        result.append(value)
V_s = result

print("***********************Check*************************")
print("img_path_list: " + str(img_num))
print("사용될 예정의 이미지 파일 개수: " + str(len(V_s)))

img_path_list.sort()  # 이미지 파일 경로 리스트 오름차순 정리

for i in range(len(img_path_list)):
    print(img_path_list[i])

print("\n\n***********************Check*************************")

# -----------------------------------------------


# 파일명 그대로 가져오기. 인용 원본코드출처:https://appia.tistory.com/502
for i in V_s:

    if os.path.isdir(root_dir + r"/" + i):
        pass

    else:
        print("file : " + i)
        if i.count(".") == 1:  # . 이 한개일떄

            V_split = i.split(".")
            # print("file Name : " + V_split[0])
            V.append(V_split[0])
        else:
            # print(len(i))
            for k in range(len(i) - 1, 0, -1):  # . 이 여러개 일때
                if i[k] == ".":
                    # print("file Name : "+i[:k])
                    break

print("현재 디렉토리의 전체 사용될 홀수번호 파일개수는: " + str(len(V)) + " 입니다.")

# ---------------------------------
print("가져온 이미지 개수:" + str(len(V)))
V.sort()
for i in range(len(V)):
    print(V[i])

img_path_list = []  # 이미지 경로들 저장용

# 라잉레그레이즈
# captured_image_list
# C:/Users/revol/downloads/피트니스 자세 이미지/Training/Day04_200924_F/1/C/033-1-1-21-Z5_C

# C:/Users/revol/downloads/피트니스 자세 이미지/Training/Day04_200924_F/1/C/033-1-1-21-Z5_C
global V  # 파일명이 들어갈 장소
V = []
global V_s  # 임시저장소
V_s = []
# 동영상에서 캡쳐된 프레임들
root_dir = 'C:/Users/revol/downloads/피트니스 자세 이미지/Training/Day04_200924_F/1/C/033-1-1-21-Z5_C'
# 디버깅용 %debug (get_images)

total_file_list = os.listdir(root_dir)  # 파일 읽어오기


def get_images():
    global img_path_list  # 이미지들만 추출해서 리스트에 추가
    possible_img_extension = ['.jpg']

    for (root, dir, files) in os.walk(root_dir):
        print(len(files))
        for i in range(0, len(files)):  # 조사결과 32번 사진까지 있는 것 확인
            if i % 2 == 1:  # 여기선 모든 사진 가져올 예정 고로 이 코드는 사용하지 않음
                if len(files) > 0:
                    for file_name in files:
                        if file_name.endswith(str(i) + ".jpg"):  # [1]: 홀수사진들만 추출

                            img_path = root + '/' + file_name
                            # img_path = img_path.replace('\\', '/') # \는 /로 나타내야함
                            V_s.append(file_name)
                            img_path_list.append(img_path)


get_images()  # 함수 호출: 디렉토리 안에 있는 모든 이미지파일들만 추출

# 중복값 제거 하고 다시 원래 리스트에 삽입
result = []
for value in img_path_list:
    if value not in result:
        result.append(value)
img_path_list = result  # 다시 전역변수로 선언 (바깥쪽)
img_num = len(img_path_list)  # 이미지 트래킹될 모든 파일들의 숫자, 전역변수

result = []
for value in V_s:
    if value not in result:
        result.append(value)
V_s = result

print("***********************Check*************************")
print("img_path_list: " + str(img_num))
print("사용될 예정의 이미지 파일 개수: " + str(len(V_s)))

img_path_list.sort()  # 이미지 파일 경로 리스트 오름차순 정리

for i in range(len(img_path_list)):
    print(img_path_list[i])

print("\n\n***********************Check*************************")

# -----------------------------------------------


# 파일명 그대로 가져오기. 인용 원본코드출처:https://appia.tistory.com/502
for i in V_s:

    if os.path.isdir(root_dir + r"/" + i):
        pass

    else:
        print("file : " + i)
        if i.count(".") == 1:  # . 이 한개일떄

            V_split = i.split(".")
            # print("file Name : " + V_split[0])
            V.append(V_split[0])
        else:
            # print(len(i))
            for k in range(len(i) - 1, 0, -1):  # . 이 여러개 일때
                if i[k] == ".":
                    # print("file Name : "+i[:k])
                    break

print("현재 디렉토리의 전체 사용될 홀수번호 파일개수는: " + str(len(V)) + " 입니다.")

# ---------------------------------
print("가져온 이미지 개수:" + str(len(V)))
V.sort()
for i in range(len(V)):
    print(V[i])