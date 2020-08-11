import os
import cv2
import numpy as np
import tensorflow as tf

from utils import label_map_util
from utils import visualization_utils as vis_util

# The 3 files (frozen_inference_graph.pb, images.jpg, label_map.pbtxt) needed for object detection
MODEL_NAME = 'ssd_mobilenet_v2_coco'
IMG_NAME = 'Times_sq1.jpg'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_complete_label_map.pbtxt')
PATH_TO_IMG = os.path.join(CWD_PATH,IMG_NAME)

# Now we need to declare total number of classes that our graph will search for
# in the image, this numerical data will help to translate the 'integer number'
# result obtained from graph to actual category which we can make sense of. 
# This number is fixed and user doesn't need to change it 
NUM_CLASSES = 90
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# calls configuration part of TensorFlow graph where we set a limit to how 
# much % of GPU memory we want to allocate for our graph to load and perform in.
# Here we have set the limit to 0.2 i.e 20 % of GPU mem will be now be reserved for this purpose only.
 
# load the frozen_inference_graph.pb into a variable called 'sess'.
# TensorFlow has it own set of code structure which we need to follow to load 
# file in specific manner. 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph, config = config)
    
# After loading the graph we need to declare variables
# which will be responsible for feeding and fetching data from our 'sess'.
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


frame = cv2.imread(PATH_TO_IMG)
frame_expanded = np.expand_dims(frame, axis = 0)
(boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict = {image_tensor: frame_expanded})
vis_util.visualize_boxes_and_labels_on_image_array(frame,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=2,min_score_thresh=0.25)
# line 29-31
cv2.imshow('FINAL IMG', frame)
cv2.imwrite('result1.jpg', frame)



