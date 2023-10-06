import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped


class SensorTransformsPublisher(Node):
    def __init__(self):
        super().__init__('sensor_transforms_publisher')
        # transient_local qos
        qos = rclpy.qos.QoSProfile(
            depth=10, durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.transform_pub = self.create_publisher(
            TFMessage, '/tf_static', qos)
        self.timer = self.create_timer(0.1, self.publish_transforms)
        self.once = False

    def publish_transforms(self):
        if self.once:
            return
        msg = TFMessage()

        # Publish transform for sensor 1
        sensor1_transform = TransformStamped()
        sensor1_transform.header.stamp = self.get_clock().now().to_msg()
        sensor1_transform.header.frame_id = 'base_link'
        sensor1_transform.child_frame_id = 'camera_link'
        sensor1_transform.transform.translation.x = 0.115
        sensor1_transform.transform.translation.y = 0.0
        sensor1_transform.transform.translation.z = -0.071
        # >>> import transforms3d as tf
        # >>> tf.euler.euler2quat(0,math.radians(45),0,'sxyz')
        # array([0.92387953, 0.        , 0.38268343, 0.        ]) w x y z
        sensor1_transform.transform.rotation.x = 0.0
        sensor1_transform.transform.rotation.y = 0.38268343
        sensor1_transform.transform.rotation.z = 0.0
        sensor1_transform.transform.rotation.w = 0.92387953
        msg.transforms.append(sensor1_transform)

        # Publish transform for gps sensor
        sensor2_transform = TransformStamped()
        sensor2_transform.header.stamp = self.get_clock().now().to_msg()
        sensor2_transform.header.frame_id = 'base_link'
        sensor2_transform.child_frame_id = 'gps_link'
        sensor2_transform.transform.translation.x = -0.115
        sensor2_transform.transform.translation.y = 0.0
        sensor2_transform.transform.translation.z = 0.160  # 0.058
        sensor2_transform.transform.rotation.x = 0.0
        sensor2_transform.transform.rotation.y = 0.0
        sensor2_transform.transform.rotation.z = 0.0
        sensor2_transform.transform.rotation.w = 1.0
        msg.transforms.append(sensor2_transform)

        # Publish transform for altimeter
        altimeter_tf = TransformStamped()
        altimeter_tf.header.stamp = self.get_clock().now().to_msg()
        altimeter_tf.header.frame_id = 'base_link'
        altimeter_tf.child_frame_id = 'teraranger_evo_40m'
        altimeter_tf.transform.translation.x = -0.133
        altimeter_tf.transform.translation.y = 0.029
        altimeter_tf.transform.translation.z = -0.070
        altimeter_tf.transform.rotation.x = 0.0
        altimeter_tf.transform.rotation.y = 0.0
        altimeter_tf.transform.rotation.z = 0.0
        altimeter_tf.transform.rotation.w = 1.0
        msg.transforms.append(altimeter_tf)

        self.transform_pub.publish(msg)
        self.once = True


def main(args=None):
    rclpy.init(args=args)
    sensor_transforms_publisher = SensorTransformsPublisher()
    rclpy.spin(sensor_transforms_publisher)
    sensor_transforms_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
