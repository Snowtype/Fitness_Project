import cv2  #OpenPose 사용과
import csv  #파일 입출력을 위한
import pandas as pd    #glob의 glob()을 사용해서 모든 파일에 작업하기
import glob

# prototxt파일: 합성곱 신경망(CNN)에 대한 정의, 파라미터와 설정값들을 포함하는 .prototxt 파일을 사용합니다. (출처: https://kyubot.tistory.com/97)
# COCO에서 각 파트번호, 선으로 연결된 POSE_PAIRS
BODY_PARTS_COCO = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                   5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                   10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
                   15: "LEye", 16: "REar", 17: "LEar", 18: "Background"}

# 17개의 라인을 만드는 pairs
POSE_PAIRS_COCO = [[0, 1], [0, 14], [0, 15], [1, 2], [1, 5], [1, 8], [1, 11], [2, 3], [3, 4],
                   [5, 6], [6, 7], [8, 9], [9, 10], [12, 13], [11, 12], [14, 16], [15, 17]]
# 신경 네트워크의 구조를 지정하는 prototxt 파일 (다양한 계층이 배열되는 방법 등) for COCO
protoFile_coco = "coco_s\\pose_deploy_linevec.prototxt"

# 각 파일 path
protoFile = "pose_deploy_linevec.prototxt"
# 조금더 정확한 pose_deploy_linevec.prototxt
# 빠른 출력 (처리속도 향상 정확도 소폭하락): pose_deploy_linevec_faster_4_stages.prototxt

weightsFile = "pose_iter_160000.caffemodel"  # MPI 용도의 코드
weightsFile_coco = "pose_iter_440000.caffemodel"  # COCO Model

# 위의 path에 있는 network 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 이미지 읽어오기
# image = cv2.imread("fitness_jpg_test/033-1-1-21-Z1_E-0000001.jpg")

row_list = []
err_printed_img = []
temp_list = []
temp_list_str = []

# ************************************관절좌표 추출 시작 ***************************
for k in range(0, len(img_path_list)):
    ff = np.fromfile(img_path_list[k], np.uint8)
    image = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)

    # image = image[220:920,720:1220]

    points = []  # 관절 좌표값을 담을 리스트


    #     #신뢰도가 0.5 이상이면 물체가 거의 정확하게 감지되었음
    #     #보통 임계값이 0에 가까울 수록 탐지되는 물체의 수가 많고 1에 가까울수록 정확도가 높아짐
    # ----------COCO MODEL-------------------------------
    def output_keypoints(frame, proto_file, weights_file, threshold, model_name, BODY_PARTS):
        global points
        global temp_list

        # 네트워크 불러오기
        net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

        # 입력 이미지의 사이즈 정의
        image_height = 368
        image_width = 368

        # 네트워크에 넣기 위한 전처리
        input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False,
                                           crop=False)

        # 전처리된 blob 네트워크에 입력
        net.setInput(input_blob)

        # 결과 받아오기 - out 변수에 전달
        out = net.forward()

        out_height = out.shape[2]
        # The fourth dimension is the width of the output map.
        out_width = out.shape[3]

        # 원본 이미지의 높이, 너비를 받아오기
        frame_height, frame_width = frame.shape[:2]

        # 포인트 리스트 초기화
        points = []

        print(f"\n============================== {model_name} Model ==============================")
        print(f"\n현재 이미지 파일은 " + img_path_list[k] + " 입니다." + str(k) + "번째 이미지 입니다.")
        for i in range(len(BODY_PARTS)):

            # ***********prob의 개념 매우 중요 ***************************
            # 신뢰도가 0.5 이상이면 물체가 거의 정확하게 감지되었음
            # 보통 임계값이 0에 가까울 수록 탐지되는 물체의 수가 많고 1에 가까울수록 정확도가 높아짐

            # 신체 부위의 신뢰도 측정
            prob_map = out[0, i, :, :]

            # 최소값, 최대값, 최소값 위치, 최대값 위치
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

            # 원본 이미지에 맞게 포인트 위치 조정
            x = (frame_width * point[0]) / out_width
            x = int(x)
            y = (frame_height * point[1]) / out_height
            y = int(y)

            if prob > threshold:  # 임계값 이상일 경우 올바른 값, 관절좌표 출력 빨간색으로 출력
                cv2.circle(frame, (x, y), 7, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)  # thickness=-1 내부 채워진 원
                cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)

                points.append((x, y))
                print(f"[pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

            else:  # [not pointed]   #텍스트를 파랑색으로 출력 - 의미없는 값 사실상 좌표 출력 안된것(B,G,R)
                cv2.circle(frame, (x, y), 7, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)

                points.append((0, 0))
                print(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

        # ***********************디렉토리 생성 후 이미지 저장***************************
        if not os.path.exists("all_image_results/userCapture_image/userCapture_printed_image_list"):
            os.makedirs("all_image_results/userCapture_image/userCapture_printed_image_list")

        cv2.imwrite("all_image_results/userCapture_image/userCapture_printed_image_list/userCapture_printed_image_"
                    + V[k] + '_' + str(k) + '.jpg', frame)
        print(points)
        # 확장자 지정안할때 오류 날 수 있음. " could not find a writer for the specified extension in function 'cv::imwrite_' "

        # ***********************디렉토리 생성 후 이미지 저장***************************

        #         cv2.imshow("Output_Keypoints", frame)
        #         cv2.waitKey(0)
        #         cv2.destroyAllWindows()

        # ***********************각 관절좌표들을 모아놓을 리스트들*********숫자형태로 저장***************
        nose_x[k] = float(points[0][0])
        nose_y[k] = float(points[0][1])
        left_shoulder_x[k] = float(points[5][0])
        left_shoulder_y[k] = float(points[5][1])
        right_shoulder_x[k] = float(points[2][0])
        right_shoulder_y[k] = float(points[2][1])
        left_elbow_x[k] = float(points[6][0])
        left_elbow_y[k] = float(points[6][1])
        right_elbow_x[k] = float(points[3][0])
        right_elbow_y[k] = float(points[3][1])
        left_wrist_x[k] = float(points[7][0])
        left_wrist_y[k] = float(points[7][1])
        right_wrist_x[k] = float(points[4][0])
        right_wrist_y[k] = float(points[4][1])
        left_hip_x[k] = float(points[11][0])
        left_hip_y[k] = float(points[11][1])
        right_hip_x[k] = float(points[8][0])
        right_hip_y[k] = float(points[8][1])
        left_knee_x[k] = float(points[12][0])
        left_knee_y[k] = float(points[12][1])
        right_knee_x[k] = float(points[9][0])
        right_knee_y[k] = float(points[9][1])
        left_ankle_x[k] = float(points[13][0])
        left_ankle_y[k] = float(points[13][1])
        right_ankle_x[k] = float(points[10][0])
        right_ankle_y[k] = float(points[10][1])

        temp_list = [nose_x[k], nose_y[k], left_shoulder_x[k],
                     left_shoulder_y[k], right_shoulder_x[k],
                     right_shoulder_y[k], left_elbow_x[k],
                     left_elbow_y[k], right_elbow_x[k],
                     right_elbow_y[k], left_wrist_x[k],
                     left_wrist_y[k], right_wrist_x[k],
                     right_wrist_y[k], left_hip_x[k],
                     left_hip_y[k], right_hip_x[k],
                     right_hip_y[k], left_knee_x[k],
                     left_knee_y[k], right_knee_x[k],
                     right_knee_y[k], left_ankle_x[k], left_ankle_y[k], right_ankle_x[k], right_ankle_y[k]]

        row_list.append(temp_list)

        # ****************************************************************************

        # ***********************에러이미지 따로 저장*******************************
        for i in range(0, len(temp_list)):  # 에러가 생성된 사진 파일들의 모음 생성 csv파일로 저장
            if temp_list[i] == 0:

                err_printed_img.append(V[k] + '_' + str(k) + '.jpg')
                if not os.path.exists("all_image_results/userCapture_image/userCapture_printed_image_list_ERR"):
                    os.makedirs("all_image_results/userCapture_image/userCapture_printed_image_list_ERR")

                cv2.imwrite(
                    "all_image_results/userCapture_image/userCapture_printed_image_list_ERR/userCapture_printed_image_"
                    + V[k] + '_' + str(k) + '_ERR.jpg', frame)

            # global root_dir   #파일 가져오는 디렉토리에 이미지의 고유 이름이 포함됨
            # cv2.imwrite("c:/printed_image_list/" + root_dir[-25:] + '_' + str(k) +'_printed.jpg',image) #root_dir 뒷부분만 가져옴

        print(points)
        # ***********************에러이미지 따로 저장*******************************
        #         cv2.imshow("Output_Keypoints", frame)
        #         cv2.waitKey(0)
        #         cv2.destroyAllWindows()

        return frame


    def output_keypoints_with_lines(frame, POSE_PAIRS):
        global temp_list
        print()
        for pair in POSE_PAIRS:
            part_a = pair[0]  # 0 (Head)
            part_b = pair[1]  # 1 (Neck)
            if points[part_a] and points[part_b]:
                print(f"[linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")
                cv2.line(frame, points[part_a], points[part_b], (100, 255, 255), 4)  # ****중요한 부분
            else:
                print(f"[not linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")

            # ***********************디렉토리 생성 후 이미지 저장***************************
            if not os.path.exists("all_image_results/userCapture_image/userCapture_line_printed_image_list"):
                os.makedirs("all_image_results/userCapture_image/userCapture_line_printed_image_list")

            cv2.imwrite(
                'all_image_results/userCapture_image//userCapture_line_printed_image_list/line_' + V[k] + '_' + str(
                    k) + '.jpg', frame)

            # 확장자 지정안할때 오류 날 수 있음. " could not find a writer for the specified extension in function 'cv::imwrite_' "

            # ********************************************************************

            # ***********************에러이미지 따로 저장*******************************
        for i in range(0, len(temp_list)):  # 에러가 생성된 사진 파일들의 모음 생성 csv파일로 저장
            if temp_list[i] == 0:

                err_printed_img.append(V[k] + '_' + str(k) + '.jpg')
                if not os.path.exists("all_image_results/userCapture_image/userCapture_line_printed_image_list_ERR"):
                    os.makedirs("all_image_results/userCapture_image/userCapture_line_printed_image_list_ERR")

                cv2.imwrite(
                    "all_image_results/userCapture_image/userCapture_line_printed_image_list_ERR/userCapture_line_printed_image_"
                    + V[k] + '_' + str(k) + '_ERR.jpg', frame)

        print(points)
        # ***********************에러이미지 따로 저장*******************************


    #         cv2.imshow("output_keypoints_with_lines", frame)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    # 이미지 읽어오기
    frame_coco = image.copy()
    # COCO Model--------
    frame_COCO = output_keypoints(frame=frame_coco, proto_file=protoFile_coco, weights_file=weightsFile_coco,
                                  threshold=0.01, model_name="COCO", BODY_PARTS=BODY_PARTS_COCO)

    print('*******************###############**')
    print(temp_list)
    print('*******#################**************')
    output_keypoints_with_lines(frame=frame_COCO, POSE_PAIRS=POSE_PAIRS_COCO)
    # ----------
    # Frame 출력
    print("*********frame***********")

    print(points)
    # str(points))
    print(points[0][0], points[0][1], points[18][0], points[1][0], points[6][0])  # 최종 포인트를 시범 테스트

    # pose_name = data['type_info']['exercise']    #테스트 파일의 동작은 학습모델에서 추론해야함

    row_int = [nose_x, nose_y, left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y, left_elbow_x,
               left_elbow_y,
               right_elbow_x, right_elbow_y, left_wrist_x, left_wrist_y, right_wrist_x, right_wrist_y, left_hip_x,
               left_hip_y,
               right_hip_x, right_hip_y, left_knee_x, left_knee_y, right_knee_x, right_knee_y, left_ankle_x,
               left_ankle_y,
               right_ankle_x, right_ankle_y]

    # 포인트 초기화
    points = []

print("*****************-----###########전체 관절 좌표 출력 끝##################------***************")

col_name_OpenPose = ['nose_x', 'nose_y', 'left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x', 'right_shoulder_y',
                     'left_elbow_x',
                     'left_elbow_y', 'right_elbow_x', 'right_elbow_y', 'left_wrist_x', 'left_wrist_y', 'right_wrist_x',
                     'right_wrist_y', 'left_hip_x', 'left_hip_y', 'right_hip_x', 'right_hip_y', 'left_knee_x',
                     'left_knee_y', 'right_knee_x',
                     'right_knee_y', 'left_ankle_x', 'left_ankle_y', 'right_ankle_x', 'right_ankle_y']

final_row_list = []
# temp_str = []
# for names, lists in zip(col_name_OpenPose, row_int):
#     temp_str = names + '_list'
#     temp_list = lists.tolist()
#     final_row_list.append(temp_list)

final_row_list = row_int

######################################################################################

last_idx = len(final_row_list[0]) - 1
for i in range(0, len(final_row_list)):  # nose_x, nose_y, left_shoulder_x, ... 관절부위들
    for j in range(last_idx):  # 해당 관절부위의 시계열 데이터 탐색
        if final_row_list[i][j] == 0:  # 만약 단 한개의 좌표라도 0이면 그 좌표값을 근사치로 바꿔준다.
            # 1
            if final_row_list[i][0] == 0:  # 만약 첫번째 좌표가 0이다.

                # 하지만, 만약 현재좌표 다음 좌표값이 0일경우 그 다음으로 넘어가기위함. j부터 시작
                for k in range(j, last_idx):  # 추가 인덱스.
                    if final_row_list[i][k] != 0:
                        final_row_list[i][j] = final_row_list[i][k]
                        break
            # 2
            elif final_row_list[i][last_idx] == 0:  # 만약 마지막 좌표가 0이다.
                # 결측값이 아니며 마지막 좌표에서 가까운 이전 좌표값을 가져온다.
                for k in range(last_idx + 1):
                    if final_row_list[i][last_idx - k] != 0:
                        final_row_list[i][last_idx] = final_row_list[i][last_idx - k]
                        break
            # 3
            else:  # 나머지들은 직전과 직후 값의 평균
                # 하지만, 만약 현재좌표 다음 좌표값이 0일경우 그 다음으로 넘어가기위함. j부터 시작
                for k in range(j, last_idx):  # 추가 인덱스. 마지막까지 감
                    if final_row_list[i][k] != 0:
                        final_row_list[i][j] = (final_row_list[i][j - 1] + final_row_list[i][k]) / 2
                        break

######################################################################################


nose_x = final_row_list[0]
nose_y = final_row_list[1]
left_shoulder_x = final_row_list[2]
left_shoulder_y = final_row_list[3]
right_shoulder_x = final_row_list[4]
right_shoulder_y = final_row_list[5]
left_elbow_x = final_row_list[6]
left_elbow_y = final_row_list[7]
right_elbow_x = final_row_list[8]
right_elbow_y = final_row_list[9]
left_wrist_x = final_row_list[10]
left_wrist_y = final_row_list[11]
right_wrist_x = final_row_list[12]
right_wrist_y = final_row_list[13]
left_hip_x = final_row_list[14]
left_hip_y = final_row_list[15]
right_hip_x = final_row_list[16]
right_hip_y = final_row_list[17]
left_knee_x = final_row_list[18]
left_knee_y = final_row_list[19]
right_knee_x = final_row_list[20]
right_knee_y = final_row_list[21]
left_ankle_x = final_row_list[22]
left_ankle_y = final_row_list[23]
right_ankle_x = final_row_list[24]
right_ankle_y = final_row_list[25]

print("**********파이널 리스트 완료************")

row_list_str = []  # 시계열 데이터가 각 컬럼마다 스트링으로 저장된 리스트
row_list_str_temp = []

len(final_row_list[1])
for i in range(len(final_row_list)):

    for j in range(len(final_row_list[i])):
        row_list_str_temp.append((final_row_list[i][j]))

    row_list_str.append(row_list_str_temp)
    row_list_str_temp = []

# 데이터 프레임 문자열 리스트 형태로 저장 - CSV 출력 용도
nose_x_list = row_list_str[0]
nose_y_list = row_list_str[1]
left_shoulder_x_list = row_list_str[2]
left_shoulder_y_list = row_list_str[3]

right_shoulder_x_list = row_list_str[4]
right_shoulder_y_list = row_list_str[5]
left_elbow_x_list = row_list_str[6]
left_elbow_y_list = row_list_str[7]

right_elbow_x_list = row_list_str[8]
right_elbow_y_list = row_list_str[9]

left_wrist_x_list = row_list_str[10]
left_wrist_y_list = row_list_str[11]
right_wrist_x_list = row_list_str[12]
right_wrist_y_list = row_list_str[13]

left_hip_x_list = row_list_str[14]
left_hip_y_list = row_list_str[15]
right_hip_x_list = row_list_str[16]
right_hip_y_list = row_list_str[17]

left_knee_x_list = row_list_str[18]
left_knee_y_list = row_list_str[19]
right_knee_x_list = row_list_str[20]
right_knee_y_list = row_list_str[21]

left_ankle_x_list = row_list_str[22]
left_ankle_y_list = row_list_str[23]
right_ankle_x_list = row_list_str[24]
right_ankle_y_list = row_list_str[25]

list_OpenPose = [nose_x_list, nose_y_list, left_shoulder_x_list,
                 left_shoulder_y_list, right_shoulder_x_list,
                 right_shoulder_y, left_elbow_x_list,
                 left_elbow_y_list, right_elbow_x_list, right_elbow_y_list,
                 left_wrist_x_list, left_wrist_y_list, right_wrist_x_list,
                 right_wrist_y_list, left_hip_x_list, left_hip_y_list,
                 right_hip_x_list, right_hip_y_list, left_knee_x_list,
                 left_knee_y_list, right_knee_x_list,
                 right_knee_y_list, left_ankle_x_list,
                 left_ankle_y_list, right_ankle_x_list, right_ankle_y_list]

# 딕셔너리로 저장 - 데이터 프레임 용도
dict_OpenPose = {'nose_x': nose_x_list, 'nose_y': nose_y_list, 'left_shoulder_x': left_shoulder_x_list,
                 'left_shoulder_y': left_shoulder_y_list, 'right_shoulder_x': right_shoulder_x_list,
                 'right_shoulder_y': right_shoulder_y, 'left_elbow_x': left_elbow_x_list,
                 'left_elbow_y': left_elbow_y_list, 'right_elbow_x': right_elbow_x_list,
                 'right_elbow_y': right_elbow_y_list,
                 'left_wrist_x': left_wrist_x_list, 'left_wrist_y': left_wrist_y_list,
                 'right_wrist_x': right_wrist_x_list,
                 'right_wrist_y': right_wrist_y_list, 'left_hip_x': left_hip_x_list, 'left_hip_y': left_hip_y_list,
                 'right_hip_x': right_hip_x_list, 'right_hip_y': right_hip_y_list, 'left_knee_x': left_knee_x_list,
                 'left_knee_y': left_knee_y_list, 'right_knee_x': right_knee_x_list,
                 'right_knee_y': right_knee_y_list, 'left_ankle_x': left_ankle_x_list,
                 'left_ankle_y': left_ankle_y_list, 'right_ankle_x': right_ankle_x_list,
                 'right_ankle_y': right_ankle_y_list}

# OpenPose_df = pd.DataFrame(final_list, columns=col_name_OpenPose)
OpenPose_df = pd.DataFrame.from_dict([dict_OpenPose])

row = [nose_x, nose_y, left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y, left_elbow_x, left_elbow_y,
       right_elbow_x, right_elbow_y, left_wrist_x, left_wrist_y, right_wrist_x, right_wrist_y, left_hip_x, left_hip_y,
       right_hip_x, right_hip_y, left_knee_x, left_knee_y, right_knee_x, right_knee_y, left_ankle_x, left_ankle_y,
       right_ankle_x, right_ankle_y]

# xy좌표 csv파일로 저장하기
# newline 설정 안하면 한 줄마다 공백있는 줄 생김
OpenPose_df.to_csv("test_csv.csv")

with open('test_points_userCapture.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['nose_x', 'nose_y', 'left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x', 'right_shoulder_y',
                     'left_elbow_x',
                     'left_elbow_y', 'right_elbow_x', 'right_elbow_y', 'left_wrist_x', 'left_wrist_y', 'right_wrist_x',
                     'right_wrist_y', 'left_hip_x', 'left_hip_y', 'right_hip_x', 'right_hip_y', 'left_knee_x',
                     'left_knee_y', 'right_knee_x',
                     'right_knee_y', 'left_ankle_x', 'left_ankle_y', 'right_ankle_x', 'right_ankle_y'])
    writer.writerow(row_list_str)

    f.close()