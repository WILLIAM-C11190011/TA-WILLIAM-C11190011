import cv2
import numpy as np
import pyrealsense2 as rs
from paho.mqtt import client as mqtt_client
import time
import random 
from datetime import datetime


distance = 0
start = 0
ukuran = 0

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
	# send = str(date_time)+str(msg)
	send = str(msg)
	result = client.publish(topic, send)
	# result: [0, 1]
	status = result[0]
	if status == 0:
		print(f"Send `{send}` to topic `{topic}` @ `{current_time}` ")
	else:
		print(f"Failed to send message to topic {topic}")

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

if __name__ == '__main__':

	
	global client
	client = connect_mqtt()
	client.loop_start()

	fps_awal = 0
	fps_baru = 0

	while True:

		fps_baru = time.time()
		fps = 1/(fps_baru-fps_awal)
		fps_awal = fps_baru
		fps = int(fps)
		label2 = ("FPS: {:.2f}".format(fps))

		frame = pipeline.wait_for_frames()
		
		color_frame = frame.get_color_frame()
		depth_frame = frame.get_depth_frame()

		

		if not color_frame:
			continue
		
	
		color_image = np.asanyarray(color_frame.get_data()) 
		depth_image = np.asanyarray(depth_frame.get_data())
	
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

	
		jarakkanan = depth_image[240, 560]/10
		jaraktengahkanan = depth_image[240, 500]/10
		jarakbaru = depth_image[240, 320]/10

		
		cv2.circle(depth_colormap,(560,240), 3, RED, -1)
		cv2.circle(depth_colormap,(400,240), 3, RED, -1)
		cv2.circle(depth_colormap,(200,240), 3, RED, -1)
		cv2.circle(depth_colormap,(320,240), 5, BLACK, -1)
		cv2.circle(depth_colormap,(500,240), 5, RED, -1)
		cv2.putText(depth_colormap, str(jarakkanan), (550, 220), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
		cv2.putText(depth_colormap, str(jaraktengahkanan), (400, 220), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
		cv2.putText(depth_colormap, str(jarakbaru), (320, 220), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)


		img = color_image.copy()
		img2 = depth_colormap.copy()


		img3=cv2.resize(img, (640,720))
		img4=cv2.resize(img2, (640,720))
	
		Output = cv2.resize(img.copy(), (854, 480))
		cv2.putText(Output, str(label2), (500, 30), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)

		cv2.putText(Output, "Depth: {}cm".format(jarakbaru), (10, 55), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)

		if 1 < jarakbaru < 23 and  ukuran == 0:
			ukuran = 1
			limitkanan = jaraktengahkanan
			print("KemasanBesarCheck")

		else:
			limitkanan = jarakkanan
		
		if limitkanan < 28 and start == 0:
			start = 1
			print("start")
	
		elif limitkanan > 28 and start == 1:
			print("Fasekirim")

			if ukuran == 1:
				print("KemasanBesar")
				publish(client, "false,5")
				ukuran = 0

			else:
				print("KemasanKecil")
				publish(client, "true,1")

			start = 0

		cv2.imshow('Output', Output)
		cv2.imshow('depth', depth_colormap)
		if cv2.waitKey(10) & 0xFF == ord('q'):	
			break
	pipeline.stop() 
	cv2.destroyAllWindows()
