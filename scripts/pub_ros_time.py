import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time

class WallTimePublisher(Node):
    def __init__(self):
        super().__init__('wall_time_publisher')
        self.publisher_ = self.create_publisher(Time, 'wall_time', 10)
        self.timer_ = self.create_timer(1.0, self.publish_wall_time)

    def publish_wall_time(self):
        msg = Time()
        msg = self.get_clock().now().to_msg()
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = WallTimePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()