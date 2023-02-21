import queue
from threading import Thread
import cv2

def concat_vh(list_2d):
    return cv2.vconcat([cv2.hconcat(list_h) for list_h in list_2d])

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src, q):
        self.stream = cv2.VideoCapture(src)
        print(src)
        if (self.stream.isOpened()== False): 
            print("Error opening video: ", src)

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.q = q
        self.current_frame = 0

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, frame) = self.stream.read()
                self.current_frame = self.current_frame + 1
                frame_small = cv2.resize(frame, (802, 550), interpolation = cv2.INTER_AREA)
                self.q.put((self.current_frame, frame_small))

    def stop(self):
        self.stopped = True


view_queues = []
num_cams = 8

for i in range(num_cams):
    view_queues.append(queue.Queue())

threadpool = []
for i in range(num_cams):
    source = "/home/jinyao/Calibration/newrig8/Cam{}.mp4".format(i)
    threadpool.append(VideoGet(source, view_queues[i]))

for thread in threadpool:
    thread.start()

imgs = [None] * num_cams


while True:
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break
    
    if key == ord('p'):
        cv2.waitKey(-1) #wait until any key is pressed

    for i in range(num_cams):
        frame_id, img = view_queues[i].get()
        cv2.putText(img, "{:.0f}".format(frame_id),
            (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
        imgs[i] = img

    if (img is None):
            break
    
    img_tile = concat_vh([[imgs[0], imgs[1], imgs[2], imgs[3]], 
                        [imgs[4], imgs[5], imgs[6], imgs[7]]])
    cv2.imshow('Frame', img_tile)


for i, thread in enumerate(threadpool):
    thread.stop()

cv2.destroyAllWindows()