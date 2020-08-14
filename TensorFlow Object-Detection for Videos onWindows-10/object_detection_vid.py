import os
import cv2
import numpy as np
import tensorflow as tf
import time
import pandas as pd
from utils import label_map_util
from utils import visualization_utils as vis_util


MODEL_NAME = 'ssd_mobilenet_v2_coco'
VIDEO_NAME = 'time_sq_vid.mov'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_complete_label_map.pbtxt')
NUM_CLASSES = 90
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph, config = config)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

video = cv2.VideoCapture(VIDEO_NAME)


while True:
    stime = time.time()
    ret, frame = video.read()
    if ret == True:
        frame = cv2.resize(frame,(300,300))
        frame_expanded = np.expand_dims(frame, axis = 0)
        
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict = {image_tensor: frame_expanded})
        
        vis_util.visualize_boxes_and_labels_on_image_array(frame,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=1,min_score_thresh=0.75)
        
        cv2.imshow('modi', frame)
        
        print('FPS {:.1f}'.format(1/ (time.time() - stime)))        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if ret == False:
        print('vid not Present')
        break
video.release()
cv2.destroyAllWindows()






