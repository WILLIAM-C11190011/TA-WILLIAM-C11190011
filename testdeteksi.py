import cv2
import numpy as np
import pyrealsense2 as rs
from paho.mqtt import client as mqtt_client
import time
import random 
from datetime import datetime

var = '0'
var2 = '0'
count1 = 0
count2 = 0
count3 = 0
count4 = 0
mostdata = 0
distance = 0
var4 = []
var3 = []
start = 0
stack = 0
average1 = 0
average2 = 0
average3 = 0
# Constants.
INPUT_WIDTH = 416
INPUT_HEIGHT = 416
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.7
CONFIDENCE_THRESHOLD = 0.5

broker = '146.190.106.65'
port = 1883
topic = "status_product"
client_id = f'python-mqtt-{random.randint(0, 1000)}'
username = 'sorting'
password = 'fiHyn{odOmlap3@sorting'

def connect_mqtt():
	def on_connect(client, userdata, flags, rc):
		if rc == 0:
			print("Connected to MQTT Broker!")
		else:
			print("Failed to connect, return code %d\n", rc)
	# Set Connecting Client ID
	client = mqtt_client.Client(client_id)
	client.username_pw_set(username, password)
	client.on_connect = on_connect
	client.connect(broker, port)
	return client

def publish(client, msg):
	#client = connect_mqtt()
	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	date_time = now.strftime("%m/%d/%Y, %H:%M:%S: ")
	send = str(date_time)+str(msg)
	send = str(msg)
	result = client.publish(topic, send)
	# result: [0, 1]
	status = result[0]
	if status == 0:
		print(f"Send `{send}` to topic `{topic}`")
	else:
		print(f"Failed to send message to topic {topic}")

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

def draw_label(input_image, label, left, top):
	"""Draw text onto image at location."""
	
	# Get text size.
	text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
	dim, baseline = text_size[0], text_size[1]
	# Use text size to create a BLACK rectangle. 
	cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED);
	# Display text inside the rectangle.
	cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


def pre_process(input_image, net):
	# Create a 4D blob from a frame.
	blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)

	# Sets the input to the network.
	net.setInput(blob)

	# Runs the forward pass to get output of the output layers.
	output_layers = net.getUnconnectedOutLayersNames()
	outputs = net.forward(output_layers)
	# print(outputs[0].shape)

	return outputs


def post_process(input_image, outputs):
	global var, var2, var3, var4, count1, count2, count3, count4, distance, mostdata, average1, average2, average3
	# Lists to hold respective values while unwrapping.
	class_ids = []
	confidences = []
	boxes = []

	image_height, image_width = input_image.shape[:2]

	# Resizing factor.
	x_factor = image_width
	y_factor =  image_height

	# Iterate through 25200 detections.
	for out in outputs:
		for detection in out:
			classes_scores = detection[5:]
			class_id = np.argmax(classes_scores)
			confidence = classes_scores[class_id]
			#  Continue if the class score is above threshold.
			if (classes_scores[class_id] > SCORE_THRESHOLD):
				confidences.append(confidence)
				class_ids.append(class_id)

				cx, cy, w, h = detection[0], detection[1], detection[2], detection[3]

				left = int((cx - w/2) * x_factor)
				top = int((cy - h/2) * y_factor)
				width = int(w * x_factor)
				height = int(h * y_factor)
			  
				box = np.array([left, top, width, height])
				boxes.append(box)

	# Perform non maximum suppression to eliminate redundant overlapping boxes with
	# lower confidences.
	indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
	for i in indices:
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, 3)

		label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
		print("deteksi:"+label)
		draw_label(input_image, label, left, top)
	
		var = classes[class_ids[i]]
		var2 = confidences[i]

		var3.append(var2)
		#print(var3)
		var4.append(var)
		#print(var4)

	return input_image

def hitungarray(array):
	
	
	count1 = array.count('Normal')
	count2 = array.count('Sobek')
	count3 = array.count('Berkerut')
	count4 = array.count('LabelTidakRapi')

	#average1 = calculate_average(var3, var4, "Sobek")
	#average2 = calculate_average(var3, var4, "Berkerut")
	#average3 = calculate_average(var3, var4, "LabelTidakRapi")
	
	max_count = max(count1, count2, count3, count4)
	print(max_count)
	if max_count > 0:
		if max_count == count1:
			if (count2 > 0 or count3 > 0 or count4 > 0):

				if count2 == count3 == count4:
					# max_conf = max(average1, average2, average3)
					# if max_conf == average1:
					# 	return 'type1b'
					
					# elif max_conf == average2:
					# 	return 'type1c'
					
					# elif max_conf == average3:
					# 	return 'type1d'
					
					return 'type1a'

				
				else:
					max_cacat = max(count2, count3, count4)
					if max_cacat == count2:
						return 'type1b'
					elif max_cacat == count3:
						return 'type1c'
					elif max_cacat == count4:
						return 'type1d'
					
			else:
				return 'type1'
		elif max_count == count2:
			return 'type2'
		elif max_count == count3:
			return 'type3'
		elif max_count == count4:
			return 'type4'
	else:
		print("ga onok opo2")
		return 'type0'


def calculate_average(var3, var4, item):
    total = 0
    count = 0
    for i in range(len(var4)):
        if var4[i] == item:
            total += var3[i]
            count += 1
    if count > 0:
        average = total / count
        return average
    else:
        return 0  # Handle the case when item is not found in var4

if __name__ == '__main__':
	
	global client
	client = connect_mqtt()
	client.loop_start()

	# Load class names.
	#classesFile = "C:/darknet/darknet-master/build/darknet/x64/data/coco.names"
	classesFile = "D:/yolov4-tiny/darknet/build/darknet/x64/data/obj.names"
	#classesFile = "data2/obj.names"
	#classesFile = "data3/obj.names"
	classes = None
	with open(classesFile, 'rt') as f:
		classes = f.read().rstrip('\n').split('\n')

	# Give the weight files to the model and load the network using them.
	#modelWeights = "yolov4_1_3_416_416_static.onnx"
 	#net = cv2.dnn.readNet(modelWeights)
	#net = cv2.dnn.readNet(model='yolov4-tiny-custom_best.weights', config='cfg/yolov4-tiny-custom.cfg')
	#net = cv2.dnn.readNet(model='/home/recomputerj1010/darknet/TrainingBaru/training13Apr21.00/yolov4-tiny-custom_best.weights', config='cfg/yolov4-tiny-custom.cfg')
	#net = cv2.dnn.readNet(model='yolov4-tiny.weights', config='cfg/yolov4-tiny.cfg')
	#net = cv2.dnn.readNet(model='C:/darknet/darknet-master/build/darknet/x64/yolov4-tiny.weights', config='C:/darknet/darknet-master/build/darknet/x64/cfg/yolov4-tiny.cfg')
	net = cv2.dnn.readNet(model='D:/yolov4-tiny/training/yolov4-tiny-custom_best.weights', config='D:/yolov4-tiny/darknet/build/darknet/x64/cfg/yolov4-tiny-custom.cfg')
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
 
	# Process image.
	fps_awal = 0
	fps_baru = 0

	while True:

		fps_baru = time.time()
		fps = 1/(fps_baru-fps_awal)
		fps_awal = fps_baru
		fps = int(fps)
		label2 = ("FPS: {:.2f}".format(fps))
		 		
		color_frame = frame.get_color_frame()
		depth_frame = frame.get_depth_frame()

		if not color_frame:
			continue
		
		color_image = np.asanyarray(color_frame.get_data()) 
		depth_image = np.asanyarray(depth_frame.get_data())
		print(depth_image)
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
		

		jarakkanan = depth_image[240, 600]/10

		jaraktengahkanan = depth_image[240, 500]/10
		jarakbaru = depth_image[240, 320]/10

		
		cv2.circle(depth_colormap,(600,240), 3, RED, -1)
		cv2.circle(depth_colormap,(400,240), 3, RED, -1)
		cv2.circle(depth_colormap,(200,240), 3, RED, -1)
		cv2.circle(depth_colormap,(320,240), 5, BLACK, -1)
		cv2.circle(depth_colormap,(500,240), 5, RED, -1)
		cv2.putText(depth_colormap, str(jarakkanan), (550, 220), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
		cv2.putText(depth_colormap, str(jaraktengahkanan), (400, 220), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
		cv2.putText(depth_colormap, str(jarakbaru), (320, 220), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
	

		detections = pre_process(color_image, net)
		img = post_process(color_image.copy(), detections)

		t, _ = net.getPerfProfile()
		label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())

		cv2.putText(img, str(label2), (500, 30), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)

		cv2.putText(img, label, (10, 30), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
		img3=cv2.resize(img, (640,720))
		img4=cv2.resize(img2, (640,720))
		#Output = np.hstack((img, img2))
		Output = cv2.resize(img.copy(), (854, 480))
		cv2.putText(Output, "Depth: {}cm".format(jarakbaru), (10, 55), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)


		if 1 < jarakbaru < 23 and  stack == 0:
			stack = 1
			limitkanan = jaraktengahkanan
			print("stack")

		else:
			limitkanan = jarakkanan
		
		if limitkanan < 28 and start == 0:
			start = 1
			print("start")
	
		elif limitkanan > 28 and start==1:
			print("kirim")
			# average1 = calculate_average(var3, var4, "Sobek")
			# average2 = calculate_average(var3, var4, "Berkerut")
			# average3 = calculate_average(var3, var4, "LabelTidakRapi")
			mostdata = hitungarray(var4)
			print (mostdata)

			if stack == 1:
				print("TisuTertumpuk")
				publish(client, "false,5")
				stack = 0
			
			elif mostdata == 'type1':
				print("Normal")
				publish(client, "true,1")

			elif mostdata == 'type2':
				print("CacatSobek")
				publish(client, "false,1")

			elif mostdata == 'type3':
				print("CacatKerut")
				publish(client, "false,2")

			elif mostdata == 'type4':
				print("CacatLabelTidakRapi")
				publish(client, "false,3")

			elif mostdata == 'type1a':
				print("NormalDenganCacat")
				publish(client, "false,1")

			elif mostdata == 'type1b':
				print("NormalDenganSobek")
				publish(client, "false,1")

			elif mostdata == 'type1c':
				print("NormalDenganKerutan")
				publish(client, "false,2")

			elif mostdata == 'type1d':
				print("NormalDenganLabelTidakRapi")
				publish(client, "false,3")

			elif mostdata == 'type0':
				print("Nothing Detected")

			#postmqtt
			start = 0
			var4=[]
			var3 =[]
		
		cv2.imshow('Output', Output)
		cv2.imshow('depth', depth_colormap)	
		if cv2.waitKey(10) & 0xFF == ord('q'):	
			break
	pipeline.stop() 
	cv2.destroyAllWindows()
