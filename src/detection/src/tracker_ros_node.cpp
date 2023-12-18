#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include "tracker.cpp"
#include <cv_bridge/cv_bridge.h>
#include <vision_msgs/msg/detection2_d.hpp>

class TrackerROSNode : public rclcpp::Node {
public:
    TrackerROSNode() : Node("image_subscriber") {
        img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                "/camera/color/image_raw", 1, std::bind(&TrackerROSNode::image_callback, this, std::placeholders::_1));
        detection_pub_ = this->create_publisher<vision_msgs::msg::Detection2D>("/bounding_box", 1);
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
            cv::cvtColor(cv_ptr->image, cv_ptr->image, cv::COLOR_RGB2BGR);
        }
        catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // resize by .5 for simulation i.e. only if image width is more than 1000
        if (cv_ptr->image.cols > 1000)
            cv::resize(cv_ptr->image, cv_ptr->image, cv::Size(), 0.5, 0.5);

        cv::Rect bbox{};
        bool success = tracker.process(cv_ptr->image, bbox);

        cv::rectangle(cv_ptr->image, bbox, cv::Scalar(0, 255, 0), 2, 1);
        std::string text = "bbox size: " + std::to_string(bbox_size_);
        cv::putText(cv_ptr->image, text, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        cv::imshow("Image window", cv_ptr->image);
        cv::setMouseCallback("Image window", TrackerROSNode::onMouse, this);
        cv::waitKey(1);

        if (!success) {
            return;
        }

        auto bbox_msg = vision_msgs::msg::Detection2D();
        bbox_msg.header = msg->header;
        bbox_msg.bbox.center.position.x = bbox.x + bbox.width / 2;
        bbox_msg.bbox.center.position.y = bbox.y + bbox.height / 2;
        bbox_msg.bbox.center.theta = 0;
        bbox_msg.bbox.size_x = bbox.width;
        bbox_msg.bbox.size_y = bbox.height;
        detection_pub_->publish(bbox_msg);
    }

    static void onMouse(int event, int x, int y, int flags, void *userdata) {
        auto obj_ptr = ((TrackerROSNode *) userdata);
        (void) flags;
        if (event == cv::EVENT_LBUTTONDOWN) {
            cv::Rect bbox;
            bbox.x = x - obj_ptr->bbox_size_ / 2;
            bbox.y = y - obj_ptr->bbox_size_ / 2;
            bbox.width = obj_ptr->bbox_size_;
            bbox.height = obj_ptr->bbox_size_;
            obj_ptr->tracker.hard_reset_bbox(bbox);
        }

        if (event == cv::EVENT_RBUTTONDOWN) {
            if (x > 480 / 2) {
                obj_ptr->bbox_size_ += 10;
            } else {
                obj_ptr->bbox_size_ -= 10;
            }
        }
    }

    double bbox_size_ = 100;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2D>::SharedPtr detection_pub_;
    Tracker tracker{};
};

int main(int argc, char *argv[]) {
    cv::setNumThreads(4);
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TrackerROSNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}