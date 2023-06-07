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
		self.flag_image = 0
		self.flag_detection = 0

		# flaf image prosess
		self.flag_trim = 1

		# Time
		self.time_start = 0
		self.time = 0

		# trim_position
		self.trim0_u0 = 0
		self.trim0_u1 = 1280
		self.trim0_v0 = 0
		self.trim0_v1 = 720


	def image_callback(self, ros_image, camera_info):

		if self.flag_detection == 1:
			input_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
			output_image, camera_info = self.image_process(input_image, camera_info)
		else:
			output_image = ros_image

		now = rospy.Time.now()
		output_image.header.stamp = now
		camera_info.header.stamp = now
		self.image_pub.publish(output_image)
		self.info_pub.publish(camera_info)



	def image_process(self, input_image, camera_info):

		if self.flag_trim == 1:
			trim_image = self.Trim(input_image)
			if self.flag_trim == 1:
				camera_info = self.ROI_process(camera_info)
				output_image = self.bridge.cv2_to_imgmsg(np.array(trim_image), "bgr8")
			else:
				output_image = self.bridge.cv2_to_imgmsg(np.array(input_image), "bgr8")
		else:
			output_image = self.bridge.cv2_to_imgmsg(np.array(input_image), "bgr8")

		return output_image, camera_info



	# Triming Process
	def Trim(self,input_image):
		self.Wide_Trim()
		trim_image = input_image[self.trim0_v0:self.trim0_v1, self.trim0_u0:self.trim0_u1]

		return trim_image
		

	
	def Wide_Trim(self):
		center_u = self.Position_now_image.x
		center_v = self.Position_now_image.y
			
		f = 1581
		z = self.Position_now_camera.z
		Length_Tag_world = 0.043

		Length_Tag_image = f * (Length_Tag_world / z)
		alpha = 1.5

		self.trim0_u0 = int(center_u - Length_Tag_image * alpha) 
		self.trim0_u1 = int(center_u + Length_Tag_image * alpha)
		self.trim0_v0 = int(center_v - Length_Tag_image * alpha)
		self.trim0_v1 = int(center_v + Length_Tag_image * alpha)

		if self.trim0_u0 < 0:
			self.trim0_u0 = 0
		if self.trim0_v0 < 0:
			self.trim0_v0 = 0



	def ROI_process(self, camera_info):
		width = self.trim0_u1 - self.trim0_u0
		height = self.trim0_v1 - self.trim0_v0
		# Modify the RegionOfInterest in the CameraInfo message
		camera_info.roi.x_offset = self.trim0_u0
		camera_info.roi.y_offset = self.trim0_v0
		camera_info.roi.width = width
		camera_info.roi.height = height
		camera_info.roi.do_rectify = True

		return camera_info



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
			self.trim0_u0 = 0
			self.trim0_u1 = 1280
			self.trim0_v0 = 0
			self.trim0_v1 = 720



	def cleanup(self):
		cv2.destroyAllWindows()



if __name__ == "__main__":
	ts = tracking_apriltag()
	rospy.spin()
