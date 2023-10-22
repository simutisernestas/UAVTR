import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2D


class BboxPublisher(Node):

    def __init__(self):
        super().__init__('bbox_publisher')
        self.publisher_ = self.create_publisher(
            Detection2D, 'bounding_box', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = Detection2D()
        # - 960.0
        # - 0.0
        # - 0.0
        # - 1397.2235298156738
        # - 540.0
        msg.bbox.center.position.x = 960/2
        msg.bbox.center.position.y = 540/2
        msg.bbox.size_x = 3.0
        msg.bbox.size_y = 4.0
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg)


def main(args=None):
    print("Hello!")
    rclpy.init(args=args)
    bbox_publisher = BboxPublisher()
    rclpy.spin(bbox_publisher)
    bbox_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
