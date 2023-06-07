#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PWM : 35.743 ~ 59.179 -- 61.133 ~ 86.523

import rospy
import cv2
import message_filters
import numpy as np
import Jetson.GPIO as GPIO
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
		self.Position_old_camera = [0, 0, 0]
		self.Position_now_camera = [0, 0, 0]
		self.Position_predicted_camera = [0, 0, 0]
		self.delta_Position_camera = [0, 0, 0]

		# Tag_image
		self.Position_old_image = [0, 0]
		self.Position_now_image = [0, 0]
		self.Position_predicted_image = [0, 0]
		self.delta_Position_image = [0, 0]
		self.delta_delta_Position_image = [0, 0]

		# gimbal_init
		pitch_pin = 32
		yaw_pin = 33
		GPIO.setmode(GPIO.BOARD)

		# pitch init
		GPIO.setup(pitch_pin, GPIO.OUT, initial=GPIO.HIGH)
		self.pitch = GPIO.PWM(pitch_pin, 400)
		self.pitch.start(60.156)
		# yaw init
		GPIO.setup(yaw_pin, GPIO.OUT, initial=GPIO.HIGH)
		self.yaw = GPIO.PWM(yaw_pin, 400)
		self.yaw.start(60.156)

		# pwm_input_value, [0]=t, [1]=t-1, center_value=60.156
		self.pitch_input_pwm = 60.156
		self.yaw_input_pwm = 60.156

		# error_value_deg, [0]=t, [1]=t-1, [2]=t-2
		self.pitch_error = [0.00, 0.00, 0.00]
		self.yaw_error = [0.00, 0.00, 0.00]
		
		# flag
		self.flag_camera = 0
		self.flag_image = 0
		self.flag_detection = 0

		self.flag_trim = 1

		# Pitch PID
		self.pitch_P = 0.053
		self.pitch_I = 0.005
		self.pitch_D = 0.002

		# yaw PID
		self.yaw_P = 0.055
		self.yaw_I = 0.002
		self.yaw_D = 0.003

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
		if self.flag_camera == 1:
			center_u = self.Position_predicted_image[0]
			center_v = self.Position_predicted_image[1]
		else:
			center_u = self.Position_now_image.x
			center_v = self.Position_now_image.y
			
		f = 1581
		z = self.Position_predicted_camera[2]
		Length_Tag_world = 0.043

		Length_Tag_image = f * (Length_Tag_world / z)
		alpha = 1.0

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

			if self.flag_camera == 0:
				self.flag_camera = 1

				self.Position_old_camera = data_camera.detections[0].pose.pose.pose.position

			else:

				self.Position_now_camera = data_camera.detections[0].pose.pose.pose.position

				self.Position_predicter_camera()
				self.Position_old_camera = self.Position_now_camera
				self.flag_detection = 1
		else:
			self.Position_old_camera = [0, 0, 0]
			self.flag_camera = 0

			self.flag_detection = 0



	def tag_image_callback(self, data_image):
		if len(data_image.detect_positions) >= 1:
			
			if self.flag_image == 0:
				self.Position_now_image = data_image.detect_positions[0]

				self.pixel_error()
				# controller
				self.pitch_pid_controller()
				self.yaw_pid_controller()

				self.Position_old_image = self.Position_now_image

				self.flag_image = 1
			else:
				self.Position_now_image = data_image.detect_positions[0]

				self.Position_predicter_image()

				self.pixel_error()
				# controller
				self.pitch_pid_controller()
				self.yaw_pid_controller()

				self.Position_old_image = self.Position_now_image
		else:
			# init
			self.pitch_input_pwm = 60.156
			self.pitch.ChangeDutyCycle(self.pitch_input_pwm)
			self.yaw_input_pwm = 60.156
			self.yaw.ChangeDutyCycle(self.yaw_input_pwm)

			self.pitch_error = [0.00, 0.00, 0.00]
			self.yaw_error = [0.00, 0.00, 0.00]

			self.Position_old_image = [0, 0, 0]
			self.Position_predicted_image = [640, 360]
			self.flag_image = 0

			self.trim0_u0 = 0
			self.trim0_u1 = 1280
			self.trim0_v0 = 0
			self.trim0_v1 = 720


	def pitch_pid_controller(self,event=None):
		P = self.pitch_P
		I = self.pitch_I
		D = self.pitch_D

		P = P*(self.pitch_error[0]-self.pitch_error[1])
		I = I*self.pitch_error[0]
		D = D*((self.pitch_error[0]-self.pitch_error[1])-(self.pitch_error[1]-self.pitch_error[2]))

		self.pitch_input_pwm =  self.pitch_input_pwm + P + I + D

		# commandable area of PWM
		if self.pitch_input_pwm >= 86.523:
			self.pitch_input_pwm = 86.523
			self.pitch.ChangeDutyCycle(self.pitch_input_pwm)

		elif self.pitch_input_pwm <= 35.743:
			self.pitch_input_pwm = 35.743
			self.pitch.ChangeDutyCycle(self.pitch_input_pwm)

		else:
			self.pitch.ChangeDutyCycle(self.pitch_input_pwm)

		# storage of error values
		self.pitch_error[2] = self.pitch_error[1]
		self.pitch_error[1] = self.pitch_error[0]



	def yaw_pid_controller(self,event=None):
		P = self.yaw_P
		I = self.yaw_I
		D = self.yaw_D
		
		P = P*(self.yaw_error[0]-self.yaw_error[1])
		I = I*self.yaw_error[0]
		D = D*((self.yaw_error[0]-self.yaw_error[1])-(self.yaw_error[1]-self.yaw_error[2]))

		self.yaw_input_pwm = self.yaw_input_pwm + P + I + D

		# commandable area of PWM
		if self.yaw_input_pwm >= 86.523:
			self.yaw_input_pwm = 86.523
			self.yaw.ChangeDutyCycle(self.yaw_input_pwm)
			
		elif self.yaw_input_pwm <= 35.743:
			self.yaw_input_pwm = 35.743
			self.yaw.ChangeDutyCycle(self.yaw_input_pwm)

		else:
			self.yaw.ChangeDutyCycle(self.yaw_input_pwm)
		# storage of error values
		self.yaw_error[2] = self.yaw_error[1]
		self.yaw_error[1] = self.yaw_error[0]



	def Position_predicter_camera(self):
		self.delta_Position_camera[0] = self.Position_now_camera.x - self.Position_old_camera.x
		self.delta_Position_camera[1] = self.Position_now_camera.y - self.Position_old_camera.y
		self.delta_Position_camera[2] = self.Position_now_camera.z - self.Position_old_camera.z

		self.Position_predicted_camera[0] = self.Position_now_camera.x + self.delta_Position_camera[0]
		self.Position_predicted_camera[1] = self.Position_now_camera.y + self.delta_Position_camera[1]
		self.Position_predicted_camera[2] = self.Position_now_camera.z + self.delta_Position_camera[2]



	def Position_predicter_image(self):
		self.delta_Position_image[0] = self.Position_now_image.x - self.Position_old_image.x
		self.delta_Position_image[1] = self.Position_now_image.y - self.Position_old_image.y

		self.Position_predicted_image[0] = self.Position_now_image.x + self.delta_Position_image[0]
		self.Position_predicted_image[1] = self.Position_now_image.y + self.delta_Position_image[1]

		self.delta_delta_Position_image = self.delta_Position_image



	def pixel_error(self):
		if self.flag_image == 1:
			error_pitch = -(360 - self.Position_predicted_image[1])
			error_yaw = (640 - self.Position_predicted_image[0])
		else:
			error_pitch = -(360 - self.Position_now_image.y)
			error_yaw = (640 - self.Position_now_image.x)

		safe_pix = 0
		# tolerance of pixel
		if -safe_pix <= error_pitch <= safe_pix:
			self.pitch_error[0] = 0
		else:
			self.pitch_error[0] = error_pitch

		if -safe_pix <= error_yaw <= safe_pix:
			self.yaw_error[0] = 0
		else:
			self.yaw_error[0] = error_yaw



	def cleanup(self):
		cv2.destroyAllWindows()
		self.pitch.stop()
		self.yaw.stop()
		GPIO.cleanup()



if __name__ == "__main__":
	ts = tracking_apriltag()
	rospy.spin()
