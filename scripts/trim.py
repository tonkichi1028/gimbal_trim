#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PWM : 35.743 ~ 59.179 -- 61.133 ~ 86.523

import rospy
import cv2
import message_filters
import numpy as np
import csv
import os
import datetime
# msg
from sensor_msgs.msg import Image, CameraInfo
from apriltag_ros.msg import AprilTagDetectionArray
from apriltag_ros.msg import AprilTagDetectionPositionArray

from cv_bridge import CvBridge, CvBridgeError



class tracking_apriltag(object):
	def __init__(self):
		# ROS
		rospy.init_node("tracking_apriltag")
		rospy.on_shutdown(self.cleanup)

		self.bridge = CvBridge()
		self.image_pub = rospy.Publisher("/masking_image", Image, queue_size=1)
		self.info_pub = rospy.Publisher("/masking_info", CameraInfo, queue_size=1)
		self.tag_det = rospy.Subscriber('/tag_detections',AprilTagDetectionArray,self.tag_camera_callback)
		self.tag_pos = rospy.Subscriber('/tag_position',AprilTagDetectionPositionArray,self.tag_image_callback)
		sub1 = message_filters.Subscriber('/usb_cam/image_raw', Image)
		sub2 = message_filters.Subscriber('/usb_cam/camera_info', CameraInfo)
		ts = message_filters.ApproximateTimeSynchronizer([sub1,sub2], 1, 0.5)
		ts.registerCallback(self.image_callback)

		# Image Size
		self.image_size = [1280,720]

		# Tag_camera
		self.Position_now_camera = [0, 0, 0]

		# Tag_image
		self.Position_now_image = [0, 0]
		
		# flag Tag
		self.flag_camera = 0
		self.flag_image = 0
		self.flag_detection = 0

		# Flaf Image Prosess
		self.flag_mask = 0
		self.flag_trim = 1

		# Time
		self.previous_detection_time = None
		self.time_start = 0
		self.time = 0

		# Data
		self.data = []
		self.TagPosImg_data = [["time"],["Image_u"],["Image_v"]]
		self.TagPosCam_data = [["time"],["camera_x"],["camera_y"],["camera_z"]]
		self.Rate_recognition = [["time"],["recognition"]]
		


		# Data File Name
		self.filename = "trim1_right.csv"



	def image_callback(self, ros_image,camera_info):

		if self.flag_detection == 1:
			input_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
			output_image = self.image_process(input_image)
		else:
			output_image = ros_image

		now = rospy.Time.now()
		output_image.header.stamp = now
		camera_info.header.stamp = now
		self.image_pub.publish(output_image)
		self.info_pub.publish(camera_info)



	def image_process(self, input_image):

		if self.flag_mask == 1:
			# Masking process
			mask_image = self.Mask(input_image)
			output_image = self.bridge.cv2_to_imgmsg(np.array(mask_image), "bgr8")
		elif self.flag_trim == 1:
			# Triming process
			trim_image = self.Trim(input_image)
			output_image = self.bridge.cv2_to_imgmsg(np.array(trim_image), "bgr8")
		else:
			output_image = self.bridge.cv2_to_imgmsg(np.array(input_image), "bgr8")

		return output_image



	# Triming Process
	def Trim(self,input_image):
		trim0_u0 = 900
		trim0_v0 = 400
		trim0_u1 = 300
		trim0_v1 = 300

		trim_image = input_image[trim0_v0:trim0_v0 + trim0_v1, trim0_u0:trim0_u0 + trim0_u1]

		return trim_image
		

	# Masking Process
	def Mask(self,input_image):
		#mask0_u0,mask0_u1,mask0_v0,mask0_v1 = self.Wide_Mask()
		
		mask0_u0 = 320#int(mask0_u0)
		mask0_v0 = 180#int(mask0_v0)
		mask0_u1 = 960#int(mask0_u1)
		mask0_v1 = 540#int(mask0_v1)

		mask_image = cv2.rectangle(input_image,(0,0),(1280,mask0_v0),color=0, thickness=-1)
		mask_image = cv2.rectangle(input_image,(0,mask0_v1),(1280,720),color=0, thickness=-1)
		mask_image = cv2.rectangle(input_image,(0,0),(mask0_u0,720),color=0, thickness=-1)
		mask_image = cv2.rectangle(input_image,(mask0_u1,0),(1280,720),color=0, thickness=-1)

		return mask_image
		

	
	def Wide_Mask(self):
		center_u = self.Position_now_image.x
		center_v = self.Position_now_image.y
			
		f = 1581
		z = self.Position_now_camera.z
		Length_Tag_world = 0.043

		Length_Tag_image = f * (Length_Tag_world / z)
		alpha = 0.8

		mask0_u0 = center_u - Length_Tag_image * alpha 
		mask0_u1 = center_u + Length_Tag_image * alpha
		mask0_v0 = center_v - Length_Tag_image * alpha
		mask0_v1 = center_v + Length_Tag_image * alpha

		return mask0_u0,mask0_u1,mask0_v0,mask0_v1



	def tag_camera_callback(self,data_camera):
		if len(data_camera.detections) >= 1:
			self.Position_now_camera = data_camera.detections[0].pose.pose.pose.position
			# GetData
			self.TagPosCam_data[0].append(self.time)
			self.TagPosCam_data[1].append(self.Position_now_camera.x)
			self.TagPosCam_data[2].append(self.Position_now_camera.y)
			self.TagPosCam_data[3].append(self.Position_now_camera.z)
			# FPS calcurate
			self.update_recognition_rate()

			self.flag_detection = 1
		else:
			self.flag_detection = 0



	def update_recognition_rate(self):
		current_time = rospy.get_time()
		# FPS calcurate
		if self.previous_detection_time is not None:
			recognition_rate = 1 / (current_time - self.previous_detection_time)
			# Get Data
			self.Rate_recognition[0].append(self.time)
			self.Rate_recognition[1].append(recognition_rate)


		self.previous_detection_time = current_time




	def tag_image_callback(self, data_image):
		if len(data_image.detect_positions) >= 1:
			self.Position_now_image = data_image.detect_positions[0]
			# Get Data
			self.TagPosImg_data[0].append(self.time)
			self.TagPosImg_data[1].append(self.Position_now_image.x)
			self.TagPosImg_data[2].append(self.Position_now_image.y)

		else:
			# init
			self.flag_image = 0


	
	def save_data_to_csv(self):
		# Create folder with today's date
		today = datetime.datetime.now().strftime('%Y.%m.%d')
		data_dir = '/home/wanglab/catkin_wsTrim/src/trim_apriltag/data/{}'.format(today)
		if not os.path.exists(data_dir):
			os.makedirs(data_dir)
		file_path = os.path.join(data_dir, self.filename)

		with open(file_path, 'w') as f:
			writer = csv.writer(f)

			data_to_save = [
				self.TagPosImg_data,
				self.TagPosCam_data,
				self.Rate_recognition
			]

			# Efficiently extend data and write to file
			for data in data_to_save:
				self.data.extend(data)

			self.data = list(zip(*self.data))

			for row in self.data:
				writer.writerow(row)


	# Timer
	def timer(self,event=None):	
		# TIME
		if self.time_start == 0:
			self.time_start = rospy.get_time()
		else:
			self.time = rospy.get_time()-self.time_start


	# Finish start Process
	def cleanup(self):
		cv2.destroyAllWindows()
		self.save_data_to_csv()


if __name__ == "__main__":
	ts = tracking_apriltag()
	rospy.Timer(rospy.Duration(1.0/100), ts.timer)
	rospy.spin()
