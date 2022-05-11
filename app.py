import streamlit as st
from imutils.perspective import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import os 

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)


    # 創建可視化文件夾
    file_dir = "vis/"
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

    # 讀取圖片
    image = cv2_img
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    # 對輸入進行裁剪操作
    image = imutils.resize(image, height = 500)

    # 圖像灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 進行高斯濾波處理
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # 進行邊緣檢測處理
    edged = cv2.Canny(gray, 75, 200)

    # 顯示並保存結果
    print("STEP 1: Edge Detection")
    cv2.imwrite("vis/edged.png", edged)

    # 在邊緣圖像中尋找輪廓，並過濾點較小的輪廓
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # 按照區域的大小進行排序並獲取前5個結果
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    # 遍歷整個輪廓集合
    for c in cnts:
        # 使用多邊形近似輪廓
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    # 顯示並保存結果
    print("STEP 2: Find contours of paper")
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imwrite("vis/contours.png", image)

    # 使用座標點進行座標變換
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # 將變換後的結果轉換爲灰度值
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # 獲取局部區域的閾值
    T = threshold_local(warped, 11, offset = 10, method = "gaussian")
    # 進行二值化處理
    warped = (warped > T).astype("uint8") * 255

    # 顯示並保存結果
    print("STEP 3: Apply perspective transform")
    cv2.imwrite("vis/orig.png", orig)
    cv2.imwrite("vis/warped.png", warped)


    result_layout = st.columns(4)
    result_layout[0].image(edged)
    result_layout[1].image(image)
    result_layout[2].image(orig)
    result_layout[3].image(warped)