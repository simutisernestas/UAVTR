#include "estimator_ros.hpp"
#include <cv_bridge/cv_bridge.h>

StateEstimationNode::StateEstimationNode() : Node("state_estimation_node") {
    {
        vel_meas_callback_group_ = this->create_callback_group(
                rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions options;
        options.callback_group = vel_meas_callback_group_;
        img_sub_ = create_subscription<sensor_msgs::msg::Image>(
                "/camera/color/image_raw", 1,
                std::bind(&StateEstimationNode::img_callback, this, std::placeholders::_1), options);
    }
    {
        target_bbox_callback_group_ = this->create_callback_group(
                rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions options;
        options.callback_group = target_bbox_callback_group_;
        bbox_sub_ = create_subscription<vision_msgs::msg::Detection2D>(
                "/bounding_box", 1,
                std::bind(&StateEstimationNode::bbox_callback, this, std::placeholders::_1), options);
    }
    {
        imu_callback_group_ = this->create_callback_group(
                rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions options;
        options.callback_group = imu_callback_group_;
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
                "/imu/data_raw", 1,
                std::bind(&StateEstimationNode::imu_callback, this, std::placeholders::_1), options);
    }

    cam_imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            "/camera/imu", 20,
            std::bind(&StateEstimationNode::cam_imu_callback, this, std::placeholders::_1));
    cam_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/color/camera_info", 1,
            std::bind(&StateEstimationNode::cam_info_callback, this, std::placeholders::_1));
    range_sub_ = create_subscription<sensor_msgs::msg::Range>(
            "/teraranger_evo_40m", 1, std::bind(&StateEstimationNode::range_callback, this, std::placeholders::_1));
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1));
    qos.best_effort();
    air_data_sub_ = create_subscription<px4_msgs::msg::VehicleAirData>(
            "/fmu/out/vehicle_air_data", qos,
            std::bind(&StateEstimationNode::air_data_callback, this, std::placeholders::_1));
    timesync_sub_ = create_subscription<px4_msgs::msg::TimesyncStatus>(
            "/fmu/out/timesync_status", qos,
            std::bind(&StateEstimationNode::timesync_callback, this, std::placeholders::_1));

    state_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
            "/state", 1);

    tf_buffer_ =
            std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ =
            std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_timer_ = create_wall_timer(std::chrono::milliseconds(1000),
                                  std::bind(&StateEstimationNode::tf_callback, this));

    estimator_ = std::make_unique<Estimator>();

    declare_parameter<bool>("simulation", false);
    get_parameter("simulation", simulation_);

    cam_ang_vel_accumulator_ = std::make_unique<AngVelAccumulator>();
    drone_ang_vel_accumulator_ = std::make_unique<AngVelAccumulator>();
}

inline float d2f(double d) {
    return static_cast<float>(d);
}

rclcpp::Time StateEstimationNode::get_correct_fusion_time(
        const std_msgs::msg::Header &header, const bool use_offset) {
    double time_point = (double) header.stamp.sec + (double) header.stamp.nanosec * 1e-9;
    if (use_offset)
        time_point += offset_.load();
    return rclcpp::Time(static_cast<int64_t>(time_point * 1e9));
}

void StateEstimationNode::timesync_callback(const px4_msgs::msg::TimesyncStatus::SharedPtr msg) {
    double offset;
    if (simulation_ && msg->estimated_offset != 0) {
        offset = -(double) msg->estimated_offset / 1e6;
    } else {
        offset = (double) msg->observed_offset / 1e6;
    }
    offset_.store(offset);
}

void StateEstimationNode::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    rclcpp::Time time;
    if (!simulation_) {
        time = get_correct_fusion_time(msg->header, false);
    } else {
        // should negate the offset in get_correct_fusion_time
        assert(false);
    }

    Eigen::Vector3f accel;
    accel << d2f(msg->linear_acceleration.x),
            d2f(msg->linear_acceleration.y),
            d2f(msg->linear_acceleration.z);

    estimator_->update_imu_accel(accel, time.seconds());

    // int64_t pub_time = static_cast<int64_t>(
    //     time.seconds() * 1e9 - offset_.load() * 1e9 + time.nanoseconds());
    auto state = estimator_->state();
    geometry_msgs::msg::PoseArray state_msg;
    state_msg.header.stamp = msg->header.stamp;
    state_msg.header.frame_id = "odom";
    state_msg.poses.resize(4);
    for (long i = 0; i < 4; i++) {
        state_msg.poses[i].position.x = state(i * 3);
        state_msg.poses[i].position.y = state(i * 3 + 1);
        state_msg.poses[i].position.z = state(i * 3 + 2);
    }
    if (target_in_sight_.load()) {
        state_msg.poses[0].orientation.x = 1.0;
        target_in_sight_.store(false);
    }
    state_pub_->publish(state_msg);

    drone_ang_vel_accumulator_->add(
            d2f(msg->angular_velocity.x),
            d2f(msg->angular_velocity.y),
            d2f(msg->angular_velocity.z));
}

void StateEstimationNode::air_data_callback(const px4_msgs::msg::VehicleAirData::SharedPtr msg) {
    float altitude;

    if (!simulation_)
        altitude = std::abs(d2f(-25.94229507446289 - msg->baro_alt_meter)) - 2.0;
    else
        altitude = msg->baro_alt_meter;

    estimator_->update_height(altitude);
}

void StateEstimationNode::range_callback(const sensor_msgs::msg::Range::SharedPtr msg) {
    return;
    if (std::isnan(msg->range) || std::isinf(msg->range))
        return;
    Eigen::Vector3f altitude{0, 0, d2f(msg->range)};

    auto time = get_correct_fusion_time(msg->header, true);
    geometry_msgs::msg::TransformStamped base_link_enu;
    bool succ = tf_lookup_helper(base_link_enu, "odom", "base_link", time);
    if (!succ) return;
    auto base_T_odom = tf_msg_to_affine(base_link_enu);

    altitude = base_T_odom.rotation() * altitude;
    estimator_->update_height(altitude[2]);
}

void StateEstimationNode::bbox_callback(const vision_msgs::msg::Detection2D::SharedPtr bbox) {
    if (!is_K_received())
        return;
    if (!image_tf_)
        return;

    auto time = get_correct_fusion_time(bbox->header, true);
    geometry_msgs::msg::TransformStamped base_link_enu;
    bool succ = tf_lookup_helper(base_link_enu, "odom", "base_link", time);
    if (!succ) return;
    auto base_T_odom = tf_msg_to_affine(base_link_enu);
    auto img_T_base = tf_msg_to_affine(*image_tf_);

    Eigen::Vector2f uv_point;
    uv_point << d2f(bbox->bbox.center.position.x),
            d2f(bbox->bbox.center.position.y);
    auto rect_point = cam_model_.rectifyPoint(cv::Point2d(uv_point[0], uv_point[1]));
    uv_point << d2f(rect_point.x), d2f(rect_point.y);

    auto cam_R_enu = base_T_odom.rotation() * img_T_base.rotation();
    estimator_->compute_pixel_rel_position(uv_point, cam_R_enu, K_);
    target_in_sight_.store(true);
}

void StateEstimationNode::cam_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    // only do once
    if (is_K_received())
        return;
    std::scoped_lock lock(mtx_);
    if (msg->header.frame_id == "x500_0/OakD-Lite/base_link/StereoOV7251")
        return;

    cam_model_.fromCameraInfo(*msg);
    cv::Matx<float, 3, 3> cvK = cam_model_.intrinsicMatrix();
    // convert to eigen
    for (long i = 0; i < 9; i++) {
        long row = std::floor(i / 3);
        long col = i % 3;
        K_(row, col) = cvK.operator()(row, col);
    }
}

bool StateEstimationNode::tf_lookup_helper(geometry_msgs::msg::TransformStamped &tf,
                                           const std::string &target_frame, const std::string &source_frame,
                                           const rclcpp::Time &time) {
    try {
        tf = tf_buffer_->lookupTransform(
                target_frame, source_frame,
                time, rclcpp::Duration::from_seconds(.008));
    }
    catch (const tf2::TransformException &ex) {
        RCLCPP_INFO(
                this->get_logger(), "Could not transform %s to %s: %s",
                source_frame.c_str(), target_frame.c_str(), ex.what());
        return false;
    }

    return true;
}

void StateEstimationNode::tf_callback() {
    if (!image_tf_) {
        if (simulation_) {
            geometry_msgs::msg::TransformStamped image_tf;
            std::string optical_frame = "camera_link_optical";
            bool succes = tf_lookup_helper(image_tf, "base_link", optical_frame);
            if (succes)
                image_tf_ = std::make_unique<geometry_msgs::msg::TransformStamped>(image_tf);
        } else {
            // camera_color_optical_frame -> base_link
            image_tf_ = std::make_unique<geometry_msgs::msg::TransformStamped>();
            image_tf_->transform.translation.x = 0.115;
            image_tf_->transform.translation.y = -0.059;
            image_tf_->transform.translation.z = -0.071;
            image_tf_->transform.rotation.x = 0.654;
            image_tf_->transform.rotation.y = -0.652;
            image_tf_->transform.rotation.z = 0.271;
            image_tf_->transform.rotation.w = -0.272;

            tera_tf_ = std::make_unique<geometry_msgs::msg::TransformStamped>();
            tera_tf_->transform.translation.x = -0.133;
            tera_tf_->transform.translation.y = 0.029;
            tera_tf_->transform.translation.z = -0.07;
            tera_tf_->transform.rotation.x = 0.0;
            tera_tf_->transform.rotation.y = 0.0;
            tera_tf_->transform.rotation.z = 0.0;
            tera_tf_->transform.rotation.w = 1.0;
        }
    }
}

Eigen::Transform<float, 3, Eigen::Affine>
StateEstimationNode::tf_msg_to_affine(geometry_msgs::msg::TransformStamped &tf_stamp) {
    Eigen::Quaternionf rotation(d2f(tf_stamp.transform.rotation.w),
                                d2f(tf_stamp.transform.rotation.x),
                                d2f(tf_stamp.transform.rotation.y),
                                d2f(tf_stamp.transform.rotation.z));
    Eigen::Translation3f translation(d2f(tf_stamp.transform.translation.x),
                                     d2f(tf_stamp.transform.translation.y),
                                     d2f(tf_stamp.transform.translation.z));
    Eigen::Transform<float, 3, Eigen::Affine> transform = translation * rotation;
    return transform;
}

void StateEstimationNode::img_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    if (!is_K_received())
        return;
    if (!image_tf_)
        return;

    // create time object from header stamp
    auto time = get_correct_fusion_time(msg->header, true);
    geometry_msgs::msg::TransformStamped base_link_enu;
    bool succ = tf_lookup_helper(base_link_enu, "odom", "base_link", time);
    if (!succ)
        return;
    auto base_T_odom = tf_msg_to_affine(base_link_enu);
    auto img_T_base = tf_msg_to_affine(*image_tf_);
    auto cam_T_enu = base_T_odom * img_T_base;

    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    }
    catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
    if (!cv_ptr) return;
    cv::Mat rectified;
    cam_model_.rectifyImage(cv_ptr->image, rectified);

    if (simulation_)
        cv::resize(cv_ptr->image, cv_ptr->image, cv::Size(), 0.5, 0.5);

    auto cam_omega = cam_ang_vel_accumulator_->get_ang_vel();
    auto drone_omega = drone_ang_vel_accumulator_->get_ang_vel();

    estimator_->update_flow_velocity(rectified, time.seconds(), cam_T_enu.rotation(),
                                     cam_T_enu.translation(), K_, cam_omega, drone_omega);
}

void StateEstimationNode::cam_imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    if (!image_tf_)
        return;

    // accumulate angular velocity into a vector
    cam_ang_vel_accumulator_->add(
            d2f(msg->angular_velocity.x),
            d2f(msg->angular_velocity.y),
            d2f(msg->angular_velocity.z));

    rclcpp::Time time = get_correct_fusion_time(msg->header, true);
    geometry_msgs::msg::TransformStamped base_link_enu;
    bool succ = tf_lookup_helper(base_link_enu, "odom", "base_link", time);
    if (!succ) return;
    const auto base_T_odom = tf_msg_to_affine(base_link_enu);
    const auto img_T_base = tf_msg_to_affine(*image_tf_);
    const auto cam_R_enu = base_T_odom.rotation() * img_T_base.rotation();

    const Eigen::Vector3f accel{d2f(msg->linear_acceleration.x),
                                d2f(msg->linear_acceleration.y),
                                d2f(msg->linear_acceleration.z)};
    const Eigen::Vector3f omega{d2f(msg->angular_velocity.x),
                                d2f(msg->angular_velocity.y),
                                d2f(msg->angular_velocity.z)};
    static Eigen::Vector3f arm{0.109f, -0.030f, 0.017f};
    estimator_->update_cam_imu_accel(accel, omega, cam_R_enu, arm);
}

// TODO:
//Eigen::Affine3f get_base2enu_T(const sensor_msgs::msg::Imu::SharedPtr msg) {
//    geometry_msgs::msg::TransformStamped base_link_enu;
//    bool succ = tf_lookup_helper(base_link_enu, "odom", "base_link", time);
//    if (!succ) return;
//    auto base_T_odom = tf_msg_to_affine(base_link_enu);
//    auto img_T_base = tf_msg_to_affine(*image_tf_);
//}