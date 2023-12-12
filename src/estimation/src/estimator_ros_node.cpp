#include "estimator_ros.hpp"

#define RCLCPP__NODE_IMPL_HPP_

#include "rclcpp/node.hpp"

int main(int argc, char **argv) {
    cv::setNumThreads(4);
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StateEstimationNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
