import cv2 as cv
import caer

''' image processing
## Drawing tools
cv.circle(img, center, redius, color, thikness)
cv.rectangle(img, pt1, pt2, color, thikness)
cv.line(img, pt1, pt2, color, thikness)

### the standard color is BGR in openCV ###
cv.cvtColor(img, color: cv.COLOR_BRG2GRAY) 
cv.resize(img, (widht, height))
cv.GaussianBlur(img, kernel)
cv.bilateralFilter(img)
cv.calcHist(imgs: list, channels: list, nBins: list, ranges: list)
_, thresh_img=cv.threshold(img, thresh_val, max_val, type: cv.THRESH_BINARY)
cv.equalizeHist(src, dest)
edges=cv.Canny(img, thresh1, thresh2)

## 2D transformations
transMat=cv.getRotationMatrix2D(rotCenter, angle, scale)
cv.wrapAffine(img, transMat, dims=[width, hight])
black=np.zeros(img.shape[:2], dtype='uint8')
mask=cv.circle(mask, (black.shape[0]//2, black.shape[1]//2), 100, 255, -1)
masked_img=cv.bitwise_and(img, img, mask=mask)

## morphological ops
delated=cv.dilate(edges, kernel, iters)
cv.erode(delated, kernel, iters)
cv.flip(img, code) --> code={-1:vh, 0:h, 1:v}
contours,hierarchs=cv.findContours(edges, ret_data_type:cv.RETR_LIST, approx_type)
cv.drawContours(img, contours, idx:-1, colors:(,,), thikness:1)

'''

cap = cv.VideoCapture('./video1.mp4')

# works only for live streams
# width, hight = 800, 600
# cap.set(3, width)
# cap.set(4, hight)

def resizeFrame(frame, scale):
    # works for all
    width = int(frame.shape[1] * scale)
    hight = int(frame.shape[0] * scale)
    return cv.resize(frame, (width, hight), interpolation=cv.INTER_AREA)

while True:
    ret, frame = cap.read()
    # frame = resizeFrame(frame, .5)
    cv.imshow('video', frame)
    if cv.waitKey(0) & 0xff == ord('q'):
        break
cap.release()
cv.destroyAllWindows()