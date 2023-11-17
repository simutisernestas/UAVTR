#include "estimator_ros_node.hpp"

StateEstimationNode::StateEstimationNode() : Node("state_estimation_node") {
#if VEL_MEAS
    {
        vel_meas_callback_group_ = this->create_callback_group(
                rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions options;
        options.callback_group = vel_meas_callback_group_;
        img_sub_ = create_subscription<sensor_msgs::msg::Image>(
                "/camera/color/image_raw", 1,
                std::bind(&StateEstimationNode::img_callback, this, std::placeholders::_1), options);
    }
#endif
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

    cam_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/color/camera_info", 1,
            std::bind(&StateEstimationNode::cam_info_callback, this, std::placeholders::_1));
    cam_target_pos_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(
            "/cam_target_pos", 1);
    gt_pose_array_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
            "/gz/gt_pose_array", 1,
            std::bind(&StateEstimationNode::gt_pose_array_callback, this, std::placeholders::_1));
    gt_target_pos_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(
            "/gt_target_pos", 1);
    range_sub_ = create_subscription<sensor_msgs::msg::Range>(
            "/teraranger_evo_40m", 1, std::bind(&StateEstimationNode::range_callback, this, std::placeholders::_1));

    state_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
            "/state", 1);

    auto qos = rclcpp::QoS(rclcpp::KeepLast(1));
    qos.best_effort();
    air_data_sub_ = create_subscription<px4_msgs::msg::VehicleAirData>(
            "/fmu/out/vehicle_air_data", qos,
            std::bind(&StateEstimationNode::air_data_callback, this, std::placeholders::_1));

    // imu_cam_sub_ = create_subscription<sensor_msgs::msg::Imu>(
    //     "/camera/imu", 10,
    //     std::bind(&StateEstimationNode::imu_cam_callback, this, std::placeholders::_1));

    imu_world_pub_ = create_publisher<geometry_msgs::msg::Vector3Stamped>(
            "/imu/data_world", 10);

    gps_sub_ = create_subscription<px4_msgs::msg::SensorGps>(
            "/fmu/out/vehicle_gps_position", qos,
            std::bind(&StateEstimationNode::gps_callback, this, std::placeholders::_1));
    gps_pub_ = create_publisher<sensor_msgs::msg::NavSatFix>(
            "/gps_postproc", qos);

    vec_pub_ = create_publisher<visualization_msgs::msg::Marker>(
            "/vec_target", 10);

    tf_buffer_ =
            std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ =
            std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_timer_ = create_wall_timer(std::chrono::milliseconds(1000),
                                  std::bind(&StateEstimationNode::tf_callback, this));

    timesync_sub_ = create_subscription<px4_msgs::msg::TimesyncStatus>(
            "/fmu/out/timesync_status", qos,
            std::bind(&StateEstimationNode::timesync_callback, this, std::placeholders::_1));

    estimator_ = std::make_unique<Estimator>();
}

void StateEstimationNode::gps_callback(const px4_msgs::msg::SensorGps::SharedPtr msg) {
    auto timestamp = (double) msg->timestamp / 1e6; // seconds
    auto time = rclcpp::Time(timestamp * 1e9);

    sensor_msgs::msg::NavSatFix gps_msg;
    gps_msg.header.stamp = time;
    gps_msg.header.frame_id = "odom";
    gps_msg.latitude = msg->lat * 1e-7;
    gps_msg.longitude = msg->lon * 1e-7;
    gps_msg.altitude = msg->alt * 1e-3;
    gps_pub_->publish(gps_msg);
}

void StateEstimationNode::timesync_callback(const px4_msgs::msg::TimesyncStatus::SharedPtr msg) {
    double offset = 0;
    if (simulation_ && msg->estimated_offset != 0) {
        offset = -(double) msg->estimated_offset / 1e6;
        return;
    }
    // assert(msg->estimated_offset == 0);
    // TODO: handle with estiamted offset not 0
    offset = (double) msg->observed_offset / 1e6;
    offset_.store(offset);
}

void StateEstimationNode::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    auto time_point = (msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9);
    if (simulation_) {
        double offset = offset_.load();
        time_point -= offset;
    }
    auto time = rclcpp::Time(time_point * 1e9);

    // geometry_msgs::msg::TransformStamped base_link_imu_cam;
    // bool succ = tf_lookup_helper(base_link_imu_cam, "odom", "camera_imu_frame", time);
    // if (succ)
    // {
    //     RCLCPP_INFO(this->get_logger(), "got tf!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    // }
    // auto base_T_odom = tf_msg_to_affine(base_link_enu);

    Eigen::Vector3d accel;
    accel << msg->linear_acceleration.x,
            msg->linear_acceleration.y,
            msg->linear_acceleration.z;
    // accel = base_T_odom.rotation() * accel;
    // accel[2] += 9.81;

    if (simulation_) {
        // publish imu in world frame
        geometry_msgs::msg::Vector3Stamped acc_world;
        acc_world.header.stamp = time; // this will have to change to absolute
        acc_world.header.frame_id = "odom";
        acc_world.vector.x = accel[0];
        acc_world.vector.y = accel[1];
        acc_world.vector.z = accel[2];
        imu_world_pub_->publish(acc_world);
    }

    if (prev_imu_time_s < 0) {
        prev_imu_time_s = time_point;
        return;
    }
    double dt = time_point - prev_imu_time_s;
    assert(dt > 0);
    prev_imu_time_s = time_point;
    estimator_->update_imu_accel(accel, dt);

    auto state = estimator_->state();
    geometry_msgs::msg::PoseArray state_msg;
    state_msg.header.stamp = time;
    state_msg.header.frame_id = "odom";
    state_msg.poses.resize(4);
    for (size_t i = 0; i < 4; i++)
    {
        state_msg.poses[i].position.x = state(i * 3);
        state_msg.poses[i].position.y = state(i * 3 + 1);
        state_msg.poses[i].position.z = state(i * 3 + 2);
    }
    state_pub_->publish(state_msg);
//    geometry_msgs::msg::PointStamped pt_msg;
//    pt_msg.header.frame_id = "odom";
//    pt_msg.header.stamp = time;
//    pt_msg.point.x = state(0);
//    pt_msg.point.y = state(1);
//    pt_msg.point.z = state(2);
//    cam_target_pos_pub_->publish(pt_msg);
}

void StateEstimationNode::air_data_callback(const px4_msgs::msg::VehicleAirData::SharedPtr msg) {
    if (!simulation_) // above beach ground
        height_.store(std::abs(-25.94229507446289 - msg->baro_alt_meter));
    else
        height_.store(msg->baro_alt_meter);

    estimator_->update_height(height_);
}

void StateEstimationNode::range_callback(const sensor_msgs::msg::Range::SharedPtr msg) {
    if (std::isnan(msg->range) || std::isinf(msg->range))
        return;
    // TODO: deal with measurement
}

void StateEstimationNode::bbox_callback(const vision_msgs::msg::Detection2D::SharedPtr bbox) {
    auto h = height_.load();
    if (h < 0 || std::isnan(h) || std::isinf(h))
        return;
    if (!is_K_received())
        return;
    if (!image_tf_)
        return;

    // create time object from header stamp
    auto time_point = offset_.load() + (bbox->header.stamp.sec + bbox->header.stamp.nanosec * 1e-9);
    auto time = rclcpp::Time(time_point * 1e9);
    geometry_msgs::msg::TransformStamped base_link_enu;
    bool succ = tf_lookup_helper(base_link_enu, "odom", "base_link", time);
    if (!succ)
        return;
    auto base_T_odom = tf_msg_to_affine(base_link_enu);
    auto img_T_base = tf_msg_to_affine(*image_tf_);
    img_T_base.translation() = Eigen::Vector3d::Zero();
    // auto tera_T_base = tf_msg_to_affine(*tera_tf_);

    Eigen::Vector2d uv_point;
    uv_point << bbox->bbox.center.position.x,
            bbox->bbox.center.position.y;

    // TODO:
    // Eigen::Vector3d H_vec;
    // H_vec << 0, 0, h;
    // H_vec = base_T_odom * tera_T_base * H_vec;

    auto cam_R_enu = base_T_odom.rotation() * img_T_base.rotation();
    // std::cout << cam_R_enu << std::endl;
    Eigen::Vector3d Pt = estimator_->compute_pixel_rel_position(uv_point, cam_R_enu, K_, h);

    // geometry_msgs::msg::PointStamped pt_msgs;
    // pt_msgs.header.stamp = time; // this will have to change to absolute
    // pt_msgs.header.frame_id = "odom";
    // pt_msgs.point.x = Pt[0];
    // pt_msgs.point.y = Pt[1];
    // pt_msgs.point.z = Pt[2];
    // cam_target_pos_pub_->publish(pt_msgs);

    RCLCPP_INFO(this->get_logger(), "xyz: %f %f %f; norm: %f",
                Pt[0], Pt[1], Pt[2], Pt.norm());

    // publish normalized vector to target
    visualization_msgs::msg::Marker vec_msgs{};
    if (simulation_) {
        time = rclcpp::Time(bbox->header.stamp.sec * 1e9 + bbox->header.stamp.nanosec);
    }
    vec_msgs.header.stamp = time; // this will have to change to absolute
    vec_msgs.header.frame_id = "odom";
    // Pt /= Pt.norm();
    vec_msgs.id = 0;
    vec_msgs.type = visualization_msgs::msg::Marker::ARROW;
    vec_msgs.action = visualization_msgs::msg::Marker::MODIFY;
    vec_msgs.pose.position.x = 0.0;
    vec_msgs.pose.position.y = 0.0;
    vec_msgs.pose.position.z = 0.0;
    vec_msgs.pose.orientation.w = 1.0;
    vec_msgs.scale.x = .1;
    vec_msgs.scale.y = .1;
    vec_msgs.scale.z = .1;
    vec_msgs.color.a = 1.0;
    vec_msgs.color.r = 1.0;
    vec_msgs.color.g = 0.0;
    vec_msgs.color.b = 0.0;
    vec_msgs.points.resize(2);
    vec_msgs.points[0].x = 0.0;
    vec_msgs.points[0].y = 0.0;
    vec_msgs.points[0].z = 0.0;
    vec_msgs.points[1].x = Pt[0];
    vec_msgs.points[1].y = Pt[1];
    vec_msgs.points[1].z = Pt[2];
    vec_pub_->publish(vec_msgs);
}

void StateEstimationNode::cam_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    // only do once
    if (is_K_received())
        return;
    std::scoped_lock lock(mtx_);
    // TODO; simulation specific : ( but will still work just fine )
    if (msg->header.frame_id == "x500_0/OakD-Lite/base_link/StereoOV7251")
        return;

    // TODO: scale does not appy for real data :<)
    static double scale{1.0};
    if (simulation_)
        scale = .5;
    for (size_t i = 0; i < 9; i++) {
        size_t row = std::floor(i / 3);
        size_t col = i % 3;
        if (row == 0 || row == 1)
            K_(row, col) = msg->k[i] * scale;
        else
            K_(row, col) = msg->k[i];
    }
}

void StateEstimationNode::gt_pose_array_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg) {
    // take norm between points 0 and 1
    if (msg->poses.size() < 2)
        return;
    auto p0 = msg->poses[0];
    auto p1 = msg->poses[1];
    Eigen::Vector3d p0_vec(p0.position.x, p0.position.y, p0.position.z);
    Eigen::Vector3d p1_vec(p1.position.x, p1.position.y, p1.position.z);
    auto diff = (p0_vec - p1_vec);
    auto gt_point = std::make_unique<geometry_msgs::msg::PointStamped>();
    gt_point->header.stamp = msg->header.stamp;
    gt_point->header.frame_id = "odom";
    gt_point->point.x = diff[0];
    gt_point->point.y = diff[1];
    gt_point->point.z = diff[2];
    gt_target_pos_pub_->publish(*gt_point);
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
        // TOOD: handle better
        RCLCPP_INFO(
                this->get_logger(), "Could not transform %s to %s: %s",
                source_frame.c_str(), target_frame.c_str(), ex.what());
        return false;
    }

    return true;
}

void StateEstimationNode::tf_callback() {
    // TODO: tf from the real data
    if (!image_tf_) {
        if (simulation_) {
            geometry_msgs::msg::TransformStamped image_tf;
            std::string optical_frame = "camera_link_optical";
            bool succes = tf_lookup_helper(image_tf, "base_link", optical_frame);
            if (succes)
                image_tf_ = std::make_unique<geometry_msgs::msg::TransformStamped>(image_tf);
        } else {
            // TODO: might be inverse
            //  : "camera_color_optical_frame"
            // - Translation: [0.115, -0.059, -0.071]
            // - Rotation: in Quaternion [0.654, -0.652, 0.271, -0.272]
            image_tf_ = std::make_unique<geometry_msgs::msg::TransformStamped>();
            image_tf_->transform.translation.x = 0.115;
            image_tf_->transform.translation.y = -0.059;
            image_tf_->transform.translation.z = -0.071;
            image_tf_->transform.rotation.x = 0.654;
            image_tf_->transform.rotation.y = -0.652;
            image_tf_->transform.rotation.z = 0.271;
            image_tf_->transform.rotation.w = -0.272;
        }
    }

    // tf_lookup_helper(tera_tf, "base_link", "teraranger_evo_40m");
    // geometry_msgs::msg::TransformStamped tera_tf;
    // tera_tf.transform.translation.x = -0.133;
    // tera_tf.transform.translation.y = 0.029;
    // tera_tf.transform.translation.z = -0.07;
    // tera_tf.transform.rotation.x = 0.0;
    // tera_tf.transform.rotation.y = 0.0;
    // tera_tf.transform.rotation.z = 0.0;
    // tera_tf.transform.rotation.w = 1.0;
    // tera_tf_ = std::make_unique<geometry_msgs::msg::TransformStamped>(tera_tf);
}

Eigen::Transform<double, 3, Eigen::Affine>
StateEstimationNode::tf_msg_to_affine(geometry_msgs::msg::TransformStamped &tf_stamp) {
    Eigen::Quaterniond rotation(tf_stamp.transform.rotation.w,
                                tf_stamp.transform.rotation.x,
                                tf_stamp.transform.rotation.y,
                                tf_stamp.transform.rotation.z);
    Eigen::Translation3d translation(tf_stamp.transform.translation.x,
                                     tf_stamp.transform.translation.y,
                                     tf_stamp.transform.translation.z);
    Eigen::Transform<double, 3, Eigen::Affine> transform = translation * rotation;
    return transform;
}

#if VEL_MEAS

void StateEstimationNode::img_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    if (!is_K_received())
        return;
    if (!image_tf_) //  || !tera_tf_
        return;
    double h = height_.load();
    if (h < 0 || std::isnan(h) || std::isinf(h))
        return;

    // create time object from header stamp
    double time_point = offset_.load() + (msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9);
    auto time = rclcpp::Time(time_point * 1e9);
    geometry_msgs::msg::TransformStamped base_link_enu;
    bool succ = tf_lookup_helper(base_link_enu, "odom", "base_link", time);
    if (!succ)
        return;
    auto base_T_odom = tf_msg_to_affine(base_link_enu);
    auto img_T_base = tf_msg_to_affine(*image_tf_);
    Eigen::Vector3d r = base_T_odom.rotation() * img_T_base.translation(); // = Eigen::Vector3d::Zero();
    auto cam_R_enu = base_T_odom.rotation() * img_T_base.rotation();

    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    }
    catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
    if (!cv_ptr) return;
    // cv::cvtColor(cv_ptr->image, cv_ptr->image, cv::COLOR_RGB2BGR);
    if (cv_ptr->image.cols > 1000)
        cv::resize(cv_ptr->image, cv_ptr->image, cv::Size(), 0.5, 0.5);

    Eigen::Vector3d vel_enu = estimator_->update_flow_velocity(cv_ptr->image, time_point, cam_R_enu, r, K_, h);
    // TODO: already published in sim!!!
    // construct Vector3Stamped message
    geometry_msgs::msg::Vector3Stamped vel_msg;
    vel_msg.header.stamp = time; // this will have to change to absolute
    vel_msg.header.frame_id = "odom";
    vel_msg.vector.x = vel_enu[0];
    vel_msg.vector.y = vel_enu[1];
    vel_msg.vector.z = vel_enu[2];
    imu_world_pub_->publish(vel_msg);
}

#endif

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StateEstimationNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}


// void imu_cam_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
// {
// auto time_point = offset_ + (msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9);
// if (simulation_)
//     time_point -= offset_;
// auto time = rclcpp::Time(time_point * 1e9);

// geometry_msgs::msg::TransformStamped base_link_imu_cam;
// bool succ = tf_lookup_helper(base_link_imu_cam, "odom", "camera_accel_optical_frame", time);
// if (!succ)
//     return;
// auto imu_T_odom = tf_msg_to_affine(base_link_imu_cam);

// Eigen::Vector3d accel;
// accel << msg->linear_acceleration.x,
//     msg->linear_acceleration.y,
//     msg->linear_acceleration.z;
// accel = imu_T_odom.rotation() * accel;
// accel[2] -= 9.81;

// RCLCPP_INFO(this->get_logger(), "imu_cam: %f %f %f", accel[0], accel[1], accel[2]);
// }
