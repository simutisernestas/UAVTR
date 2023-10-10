#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include "tracker.cpp"
#include <cv_bridge/cv_bridge.h>
#include <vision_msgs/msg/bounding_box2_d.hpp>

class ImageSubscriber : public rclcpp::Node
{
public:
    ImageSubscriber() : Node("image_subscriber")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/color/image_raw", 10, std::bind(&ImageSubscriber::image_callback, this, std::placeholders::_1));
        publisher_ = this->create_publisher<vision_msgs::msg::BoundingBox2D>("/bounding_box", 10);
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // cv brdige msg to cv
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
            cv::cvtColor(cv_ptr->image, cv_ptr->image, cv::COLOR_RGB2BGR);
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        cv::Rect bbox{};
        tracker.process(cv_ptr->image, bbox);
        cv::rectangle(cv_ptr->image, bbox, cv::Scalar(0, 255, 0), 2, 1);
        cv::imshow("Image window", cv_ptr->image);
        cv::waitKey(1);

        // publish bbox TODO: double check if correct
        auto bbox_msg = vision_msgs::msg::BoundingBox2D();
        bbox_msg.center.position.x = bbox.x + bbox.width / 2;
        bbox_msg.center.position.y = bbox.y + bbox.height / 2;
        bbox_msg.center.theta = 0;
        bbox_msg.size_x = bbox.width;
        bbox_msg.size_y = bbox.height;
        publisher_->publish(bbox_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<vision_msgs::msg::BoundingBox2D>::SharedPtr publisher_;
    Tracker tracker{};
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ImageSubscriber>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}