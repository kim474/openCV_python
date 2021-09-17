import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
img1 = cv2.imread("../test_binary_wooden_plate.jpg")
img2 = cv2.imread("../test_blur_binary_wooden_plate.jpg")
img3 = cv2.imread("../test_binary_glass_plate1.jpg")
img4 = cv2.imread("../test_blur_binary_glass_plate1.jpg")
img5 = cv2.imread("../test_binary_glass_plate2.jpg")
img6 = cv2.imread("../test_blur_binary_glass_plate2.jpg")
img7 = cv2.imread("../test_binary_glass_plate3.jpg")
img8 = cv2.imread("../test_blur_binary_glass_plate3.jpg")
img9 = cv2.imread("../test_binary_plastic_plate1.jpg")
#img10 = cv2.imread("../test_blur_binary_plastic_plate1.jpg")
img11 = cv2.imread("../test_binary_plastic_plate2.jpg")
#img12 = cv2.imread("../test_blur_binary_plastic_plate2.jpg")
img13 = cv2.imread("../test_binary_stainless_plate1.jpg")
img14 = cv2.imread("../test_blur_binary_stainless_plate1.jpg")
img15 = cv2.imread("../test_binary_stainless_plate2.jpg")
img16 = cv2.imread("../test_blur_binary_stainless_plate2.jpg")
img17 = cv2.imread("../test1_wooden_plate1.jpg")
img18 = cv2.imread("../test1_wooden_plate2.jpg")
img19 = cv2.imread("../test1_wooden_plate3.jpg")
img20 = cv2.imread("../test1_.jpg")

img1 = cv2.imread("../binary_wooden_plate1.jpg")
img2 = cv2.imread("../blur_wooden_plate1.jpg")
img3= cv2.imread("../binary_wooden_plate2.jpg")
img4 = cv2.imread("../blur_wooden_plate2.jpg")
img5 = cv2.imread("../binary_wooden_plate3.jpg")
img6 = cv2.imread("../blur_wooden_plate3.jpg")
img7 = cv2.imread("../binary_stainless_plate1.jpg")
img8 = cv2.imread("../blur_stainless_plate1.jpg")
img9 = cv2.imread("../binary_stainless_plate2.jpg")
img10 = cv2.imread("../blur_stainless_plate2.jpg")
img11 = cv2.imread("../blur_glass_plate1.jpg")
img12 = cv2.imread("../blur_glass_plate2.jpg")
img13 = cv2.imread("../blur_paper_plate1.jpg")
img14 = cv2.imread("../blur_paper_plate2.jpg")
img15 = cv2.imread("../blur_paper_plate3.jpg")
img16 = cv2.imread("../blur_plastic_plate1.jpg")
img17 = cv2.imread("../blur_plastic_plate2.jpg")
img18 = cv2.imread("../test1_.jpg")
"""

img1 = cv2.imread("../m1_ll.jpg")
img2 = cv2.imread("../m2_ll.jpg")
img3 = cv2.imread("../p1_ll.jpg")
img4 = cv2.imread("../p2_ll.jpg")
img5 = cv2.imread("../w1_ll.jpg")

imgs = [img3, img4, img1, img2, img5]
hists = []

for i, img in enumerate(imgs):
    plt.subplot(1, len(imgs), i+1)
    plt.title('img%d' % (i+1))
    plt.axis('off')
    plt.imshow(img[:, :, ::-1])

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    #images: input, channel: input image의 특정 channel 지정 가능, mask: 이미지 분석 영역(none이면 전체),
    #histSize: Bins value, ranges: range value

    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    hists.append(hist)

img1 = hists[0]
methods = {'CORREL': cv2.HISTCMP_CORREL, 'CHISQR': cv2.HISTCMP_CHISQR,
           'INTERSECT': cv2.HISTCMP_INTERSECT,
           'BHATTACHARYYA': cv2.HISTCMP_BHATTACHARYYA}
#Correlation, Chi-Square, Intersection, BHattacharyya Distance
for j, (name, flag) in enumerate(methods.items()):
    print('%-10s'%name, end='\t')
    for i, (hist, img) in enumerate(zip(hists, imgs)):
        ret: int = 0
        result1: float = 0.0
        result2: float = 0.0
        result3: float = 0.0
        result4: float = 0.0
        count: int = 0
        retVal: int = 0

        ret = cv2.compareHist(img1, hist, flag)
        if flag == cv2.HISTCMP_CORREL:
            result1 = ret
        if flag == cv2.HISTCMP_CHISQR:
            result2 = ret
        if flag == cv2.HISTCMP_INTERSECT:
            ret = ret/np.sum(img1)
            result3 = ret
        if flag == cv2.HISTCMP_BHATTACHARYYA:
            result4 = ret
        print("img%d:%7.2f"% (i+1, ret), end='\t')

    #하얀색 glass plate 기준 / plastic과 비교 불가능
    if result1 > 0.8:
        count = count + 1
    if result2 < 200:
        count = count + 1
    if result3 < 0.74:
        count = count + 1
    if result4 < 0.54:
        count = count + 1
    print()

"""
    # wooden plate 기준
    if result1 > 0.4:
        count = count + 1
    if result2 < 300:
        count = count + 1
    if result3 > 0.5:
        count = count + 1
    if result4 < 0.4:
        count = count + 1
    print()
    
    # 이진화 wooden plate 기준
    if result1 > 0.99:
        count = count + 1
    if result2 <= 0.03:
        count = count + 1
    if result3 >= 0.96:
        count = count + 1
    if result4 < 0.10:
        count = count + 1
    print()

    # 이진화 stainless plate 기준
    if result1 > 0.99:
        count = count + 1
    if result2 < 0.1:
        count = count + 1
    if result3 >= 0.98:
        count = count + 1
    if result4 <= 0.08:
        count = count + 1
    print()
    """

#print(result1, result2, result3, result4)
plt.show()
