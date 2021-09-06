import cv2  #OpenPose 사용과
import csv  #파일 입출력을 위한
import pandas as pd    #glob의 glob()을 사용해서 모든 파일에 작업하기
import glob

for i in range(0, len(final_row_list)):  # nose_x, nose_y, left_shoulder_x, ... 관절부위들

    for j in range(len(final_row_list[i])):  # 해당 관절부위의 시계열 데이터 탐색

        if final_row_list[i][j] == 0:  # 만약 단 한개의 좌표라도 0이면 그 좌표값을 의미있게 바꿔준다.

            if final_row_list[i][0] == 0:  # 만약 첫번째 좌표가 0이다.

                # 하지만, 만약 현재좌표 다음 좌표값이 0일경우 그 다음으로 넘어가기위함. j부터 시작
                for k in range(j, len(final_row_list[i])):  # 추가 인덱스.

                    if final_row_list[i][k] != 0:
                        final_row_list[i][j] = final_row_list[i][k]
                        break

                # row_list[i][0] = row_list[i][1]    #두번째 좌표값을 첫번째 좌표값으로 돌려준다.

            elif final_row_list[i][len(final_row_list[i]) - 1] == 0:  # 만약 마지막 좌표가 0이다.
                final_row_list[i][len(final_row_list[i]) - 1] = final_row_list[i][
                    len(final_row_list[i]) - 2]  # 마지막 좌표바로 직전 좌표값을 가져온다.

                for k in range(len(final_row_list[i])):
                    if final_row_list[i][k] != 0:
                        final_row_list[i][j] = final_row_list[i][k]
                        break

            else:  # 나머지들은 직전과 직후 값의 평균
                # 하지만, 만약 현재좌표 다음 좌표값이 0일경우 그 다음으로 넘어가기위함. j부터 시작
                for k in range(j, len(final_row_list[i])):  # 추가 인덱스. 마지막까지 감
                    if final_row_list[i][k] != 0:
                        final_row_list[i][j] = (final_row_list[i][j - 1] + final_row_list[i][k]) / 2

                        break

# print(final_row_list)
for i in range(len(final_row_list)):
    # for j in range(len(final_row_list[i])):
    print(final_row_list[i])

print(len(final_row_list[1]))
print(final_row_list[1])
print("Hello")

for k in range(2, 5):
    print(str(k))