import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
from PIL import Image, ImageDraw, ImageFont
import serial as sl

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/paris.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

# Get angles and distances from center to aim the turret.
#THIS FUNCTION IS ANDREW AND JASONS CODE
def get_angles_and_distances(img, boxes, i):
    wh = np.flip(img.shape[0:2])
    img = Image.fromarray(img)
    w = img.width
    h = img.height
    center = tuple([(0+w)/2, (0+h)/2])
    x1y1 = ((np.array(boxes[i][0:2]) * wh).astype(np.int32))
    x2y2 = ((np.array(boxes[i][2:4]) * wh).astype(np.int32))
    aimpoint = tuple([(x1y1[0]+x2y2[0])/2, (x1y1[1]+x2y2[1])/2])
    distance_from_x = aimpoint[0] - center[0]
    distance_from_y = aimpoint[1] - center[1]
    degrees_from_x = np.degrees(np.arctan((aimpoint[1]-center[1])/((aimpoint[0]-center[0]))))
    degrees_from_y = np.degrees(np.arctan((aimpoint[0]-center[0])/((aimpoint[1]-center[1]))))
    return tuple([degrees_from_x, degrees_from_y]), tuple([distance_from_x, distance_from_y])

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    fps = 0.0
    count = 0

    ser = sl.Serial("COM3", 57600)

    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        fps  = ( fps + (1./(time.time()-t1)) ) / 2

        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

        for i in range(nums[0]):
            #TEST THAT ONLY SENDS DETECTIONS THAT ARE PEOPLE (ANDREW AND JASONS CODE) 
            if(classes[0][i] == 0 and scores[0][i] >= 0.90):
                angles, distances = get_angles_and_distances(img, boxes[0], i)
                #IF THEY ARE WITH A 10px SQUARE GO AHEAD AND SHOOT
                #IF WE WERE MOVING THE CAMERA/TURRET WE WOULD JUST SEND THE DISTANCES TO THE ARDUINO
                #AND LET IT FIGURE OUT WHAT TO DO
                if((distances[0] <= 10 and distances[0] >= -10) and (distances[1] <= 10 and distances[1] >= -10)):
                    ser.write("SHOOT".encode())
                    print(ser.readline())
                print(angles)
                print(distances)
                print("\n")

        img = cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        
        if FLAGS.output:
            out.write(img)
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
