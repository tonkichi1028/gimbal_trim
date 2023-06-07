#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PWM : 35.743 ~ 59.179 -- 61.133 ~ 86.523

import rospy
import cv2
import message_filters
import numpy as np
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

		# image size
		self.image_size = [1280,720]

		# Tag_camera
		self.Position_now_camera = [0, 0, 0]

		# Tag_image
		self.Position_now_image = [0, 0]
		
		# flag Tag
		self.flag_camera = 0
		self.flag_image = 0
		self.flag_detection = 0

		# flaf image prosess
		self.flag_mask = 0
		self.flag_trim = 0

		# Time
		self.time_start = 0
		self.time = 0



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
		trim0_u0 = 0
		trim0_v0 = 0
		trim0_u1 = 620
		trim0_v1 = 360

		trim_image = input_image[trim0_v0:trim0_v0 + trim0_v1, trim0_u0:trim0_u0 + trim0_u1]

		return trim_image
		

	# Masking Process
	def Mask(self,input_image):
		mask0_u0,mask0_u1,mask0_v0,mask0_v1 = self.Wide_Mask()
		
		mask0_u0 = int(mask0_u0)
		mask0_v0 = int(mask0_v0)
		mask0_u1 = int(mask0_u1)
		mask0_v1 = int(mask0_v1)

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
			self.flag_detection = 1
		else:
			self.flag_detection = 0



	def tag_image_callback(self, data_image):
		if len(data_image.detect_positions) >= 1:
			self.Position_now_image = data_image.detect_positions[0]
		else:
			# init
			self.flag_image = 0



	def cleanup(self):
		cv2.destroyAllWindows()



if __name__ == "__main__":
	ts = tracking_apriltag()
	rospy.spin()
