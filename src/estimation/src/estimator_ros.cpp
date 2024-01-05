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
    imu_callback_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions options;
    options.callback_group = imu_callback_group_;
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data_raw", 2,
        std::bind(&StateEstimationNode::imu_callback, this, std::placeholders::_1), options);
  }
  {
    rest_sensors_callback_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::Reentrant);
    rclcpp::SubscriptionOptions options;
    options.callback_group = rest_sensors_callback_group_;

    bbox_sub_ = create_subscription<vision_msgs::msg::Detection2D>(
        "/bounding_box", 1,
        std::bind(&StateEstimationNode::bbox_callback, this, std::placeholders::_1), options);
    range_sub_ = create_subscription<sensor_msgs::msg::Range>(
        "/teraranger_evo_40m", 1,
        std::bind(&StateEstimationNode::range_callback, this, std::placeholders::_1), options);
    auto air_data_qos = rclcpp::QoS(1);
    air_data_qos.keep_all();
    air_data_qos.best_effort();
    air_data_sub_ = create_subscription<px4_msgs::msg::VehicleAirData>(
        "/fmu/out/vehicle_air_data", air_data_qos,
        std::bind(&StateEstimationNode::air_data_callback, this, std::placeholders::_1), options);
  }

  state_pub_timer_ = create_wall_timer(std::chrono::milliseconds(10),
                                       std::bind(
                                           &StateEstimationNode::state_pub_callback, this));
  cam_imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      "/camera/imu", 20,
      std::bind(&StateEstimationNode::cam_imu_callback, this, std::placeholders::_1));
  cam_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      "/camera/color/camera_info", 1,
      std::bind(&StateEstimationNode::cam_info_callback, this, std::placeholders::_1));

  auto timesync_qos = rclcpp::QoS(1);
  timesync_qos.best_effort();
  timesync_qos.keep_all();
  timesync_qos.durability_volatile();
  timesync_sub_ = create_subscription<px4_msgs::msg::TimesyncStatus>(
      "/fmu/out/timesync_status", timesync_qos,
      std::bind(&StateEstimationNode::timesync_callback, this, std::placeholders::_1));

  state_pub_ = create_publisher<geometry_msgs::msg::PoseArray>("/state", 1);

  tf_buffer_ =
      std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ =
      std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  tf_timer_ = create_wall_timer(std::chrono::milliseconds(1000),
                                std::bind(&StateEstimationNode::tf_callback, this));

  declare_parameter<float>("spatial_vel_flow_error", 10);
  declare_parameter<float>("flow_vel_rejection_perc", 5);
  EstimatorConfig config{
      .spatial_vel_flow_error = static_cast<float>(get_parameter("spatial_vel_flow_error").as_double()),
      .flow_vel_rejection_perc = static_cast<float>(get_parameter("flow_vel_rejection_perc").as_double()) / 100.0f};
  estimator_ = std::make_unique<Estimator>(config);

  declare_parameter<bool>("simulation", false);
  get_parameter("simulation", simulation_);
  declare_parameter<float>("baro_ground_ref", 10.0);
  get_parameter("baro_ground_ref", baro_ground_ref_);

  cam_ang_vel_accumulator_ = std::make_unique<AngVelAccumulator>();
  drone_ang_vel_accumulator_ = std::make_unique<AngVelAccumulator>();

  range_pub_ = create_publisher<sensor_msgs::msg::Range>("/range", 1);
  target_pt_pub_ = create_publisher<geometry_msgs::msg::PointStamped>("/target_point", 1);

  ins_angl_vel_pub_ = create_publisher<sensor_msgs::msg::Imu>("/ins_angl_vel", 1);
  ins_angl_vel_pub_2_ = create_publisher<sensor_msgs::msg::Imu>("/ins_angl_vel_2", 1);
  twist_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>("/vel", 1);
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
  } else if (msg->estimated_offset != 0) {
    auto delta = msg->timestamp - msg->remote_timestamp;
    offset = (double) delta / 1e6;
  } else {
    offset = (double) msg->observed_offset / 1e6;
  }
  offset_.store(offset);
}

typedef std::chrono::high_resolution_clock Clock;
#define LATENCY_MEASUREMENT 0

void StateEstimationNode::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
#if LATENCY_MEASUREMENT
  auto t0 = Clock::now();
#endif

  rclcpp::Time time;
  if (!simulation_) {
    time = get_correct_fusion_time(msg->header, false);
  } else {
    // should negate the offset in get_correct_fusion_time
    assert(false);
  }
  time_.store(time.nanoseconds());

  Eigen::Vector3f accel;
  accel << d2f(msg->linear_acceleration.x),
      d2f(msg->linear_acceleration.y),
      d2f(msg->linear_acceleration.z);

  estimator_->update_imu_accel(accel, time.seconds());
  
#if LATENCY_MEASUREMENT
  auto t1 = Clock::now();
  RCLCPP_INFO(this->get_logger(), "imu callback took %f ms", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f);
#endif

  // drone_ang_vel_accumulator_->add(
  //     d2f(msg->angular_velocity.x),
  //     d2f(msg->angular_velocity.y),
  //     d2f(msg->angular_velocity.z));
}

void StateEstimationNode::air_data_callback(const px4_msgs::msg::VehicleAirData::SharedPtr msg) {
  float altitude;

  if (!simulation_)
    altitude = msg->baro_alt_meter + baro_ground_ref_;
  else
    altitude = msg->baro_alt_meter;

  estimator_->update_height(altitude);
}

void StateEstimationNode::range_callback(const sensor_msgs::msg::Range::SharedPtr msg) {
  // max range of sensor is 40m
  if (std::isnan(msg->range) || std::isinf(msg->range) || msg->range > 40)
    return;
  static float prev_range = 0;
  if (std::abs(msg->range - prev_range) < 1e-2)
    return;
  prev_range = msg->range;

  auto time = get_correct_fusion_time(msg->header, true);
  geometry_msgs::msg::TransformStamped base_link_enu;
  bool succ = tf_lookup_helper(base_link_enu, "odom", "base_link", time);
  if (!succ)
    return;
  auto base_T_odom = tf_msg_to_affine(base_link_enu);

  Eigen::Vector3f altitude{0, 0, d2f(msg->range)};
  altitude = base_T_odom.rotation() * altitude;

  sensor_msgs::msg::Range pub_range_msg;
  pub_range_msg.header.stamp = msg->header.stamp;
  pub_range_msg.header.frame_id = "odom";
  pub_range_msg.range = altitude[2];
  range_pub_->publish(pub_range_msg);

  float est_h = std::abs(estimator_->state()[2]);
  bool too_high_deviation = std::abs(est_h - altitude[2]) > 3.0;
  if (too_high_deviation) {
    RCLCPP_WARN(this->get_logger(), "Height deviation too high: %f", est_h - altitude[2]);
    return;
  }

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
  if (!succ)
    return;
  auto base_T_odom = tf_msg_to_affine(base_link_enu);
  auto img_T_base = tf_msg_to_affine(*image_tf_);

  Eigen::Vector2f avg_target_pixel{d2f(bbox->bbox.center.position.x),
                                   d2f(bbox->bbox.center.position.y)};
  Eigen::Vector2f c1, c2, c3, c4; // corners
  c1 << d2f(bbox->bbox.center.position.x - bbox->bbox.size_x / 2),
      d2f(bbox->bbox.center.position.y - bbox->bbox.size_y / 2);
  c2 << d2f(bbox->bbox.center.position.x + bbox->bbox.size_x / 2),
      d2f(bbox->bbox.center.position.y - bbox->bbox.size_y / 2);
  c3 << d2f(bbox->bbox.center.position.x - bbox->bbox.size_x / 2),
      d2f(bbox->bbox.center.position.y + bbox->bbox.size_y / 2);
  c4 << d2f(bbox->bbox.center.position.x + bbox->bbox.size_x / 2),
      d2f(bbox->bbox.center.position.y + bbox->bbox.size_y / 2);
  avg_target_pixel += (c1 + c2 + c3 + c4);
  avg_target_pixel /= 5;
  auto rect_point = cam_model_.rectifyPoint(cv::Point2d(avg_target_pixel[0],
                                                        avg_target_pixel[1]));
  avg_target_pixel[0] = d2f(rect_point.x);
  avg_target_pixel[1] = d2f(rect_point.y);

  auto cam_T_enu = base_T_odom * img_T_base;
  Eigen::Vector3f Pt = estimator_->update_target_position(
      avg_target_pixel, cam_T_enu, K_, img_T_base.translation());
  target_in_sight_.store(true);

  geometry_msgs::msg::PointStamped target_pt_msg;
  target_pt_msg.header.stamp = time;
  target_pt_msg.header.frame_id = "odom";
  target_pt_msg.point.x = Pt[0];
  target_pt_msg.point.y = Pt[1];
  target_pt_msg.point.z = Pt[2];
  target_pt_pub_->publish(target_pt_msg);
}

void StateEstimationNode::cam_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  // only do once
  if (is_K_received())
    return;
  if (msg->header.frame_id == "x500_0/OakD-Lite/base_link/StereoOV7251")
    return;

  std::scoped_lock lock(mtx_);
  cam_model_.fromCameraInfo(*msg);
  cv::Matx<float, 3, 3> cvK = cam_model_.intrinsicMatrix();
  // convert to eigen
  for (long i = 0; i < 9; i++) {
    long row = std::floor(i / 3);
    long col = i % 3;
    K_(row, col) = cvK.operator()(row, col);
  }

  // unregister callback
  cam_info_sub_ = nullptr;
}

bool StateEstimationNode::tf_lookup_helper(geometry_msgs::msg::TransformStamped &tf,
                                           const std::string &target_frame, const std::string &source_frame,
                                           const rclcpp::Time &time) {
  try {
    tf = tf_buffer_->lookupTransform(
        target_frame, source_frame,
        time, rclcpp::Duration::from_seconds(.008));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_ERROR_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000 /*ms*/,
        "Could not transform %s to %s: %s",
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
      image_tf_->transform.translation.x = -0.115;
      image_tf_->transform.translation.y = 0.059;
      image_tf_->transform.translation.z = 0.071;
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

  // unregister callback
  tf_timer_ = nullptr;
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

  const auto time = get_correct_fusion_time(msg->header, true);
  geometry_msgs::msg::TransformStamped base_link_enu;
  if (!tf_lookup_helper(base_link_enu, "odom", "base_link", time))
    return;
  const auto base_T_odom = tf_msg_to_affine(base_link_enu);
  const auto img_T_base = tf_msg_to_affine(*image_tf_);

  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  if (!cv_ptr)
    return;
  cv::Mat rectified;
  cam_model_.rectifyImage(cv_ptr->image, rectified);

  if (simulation_)
    cv::resize(cv_ptr->image, cv_ptr->image, cv::Size(), 0.5, 0.5);

  // actually these are in body and camera frame
  const auto cam_omega = cam_ang_vel_accumulator_->get_ang_vel();
  const auto drone_omega = drone_ang_vel_accumulator_->get_ang_vel();

  Eigen::Vector3f vel = estimator_->update_flow_velocity(
      rectified, time.seconds(), base_T_odom,
      img_T_base, K_, cam_omega, drone_omega);

  {
    sensor_msgs::msg::Imu ins_angl_vel_msg;
    ins_angl_vel_msg.header.stamp = time;
    ins_angl_vel_msg.header.frame_id = "camera";
    ins_angl_vel_msg.angular_velocity.x = cam_omega[0];
    ins_angl_vel_msg.angular_velocity.y = cam_omega[1];
    ins_angl_vel_msg.angular_velocity.z = cam_omega[2];
    ins_angl_vel_pub_->publish(ins_angl_vel_msg);
  }
  {
    sensor_msgs::msg::Imu ins_angl_vel_msg;
    ins_angl_vel_msg.header.stamp = time;
    ins_angl_vel_msg.header.frame_id = "base_link";
    ins_angl_vel_msg.angular_velocity.x = drone_omega[0];
    ins_angl_vel_msg.angular_velocity.y = drone_omega[1];
    ins_angl_vel_msg.angular_velocity.z = drone_omega[2];
    ins_angl_vel_pub_2_->publish(ins_angl_vel_msg);
  }

  // publish velocity
  geometry_msgs::msg::TwistStamped vel_msg;
  vel_msg.header.stamp = time;
  vel_msg.header.frame_id = "odom";
  vel_msg.twist.linear.x = vel[0];
  vel_msg.twist.linear.y = vel[1];
  vel_msg.twist.linear.z = vel[2];
  twist_pub_->publish(vel_msg);
}

void StateEstimationNode::state_pub_callback() {
  // publish state
  Eigen::VectorXf state = estimator_->state();
  Eigen::MatrixXf cov = estimator_->covariance();
  geometry_msgs::msg::PoseArray state_msg;
  state_msg.header.stamp = rclcpp::Time(time_.load());
  state_msg.header.frame_id = "odom";
  state_msg.poses.resize(4);
  for (long i = 0; i < 4; i++) {
    state_msg.poses[i].position.x = state(i * 3);
    state_msg.poses[i].position.y = state(i * 3 + 1);
    state_msg.poses[i].position.z = state(i * 3 + 2);

    if (i > 0) {
      // compute eigen values of 3x3 covariance matrix of position
      Eigen::Matrix3f cov_pos = cov.block<3, 3>(0, 0, 3, 3);
      Eigen::EigenSolver<Eigen::Matrix3f> es(cov_pos);
      Eigen::Vector3f eigenvalues = es.eigenvalues().real();
      // store position covariance
      long cov_idx = (i - 1);
      state_msg.poses[i].orientation.x = eigenvalues(cov_idx);
    }
  }
  if (target_in_sight_.load()) {
    state_msg.poses[0].orientation.x = 1.0;
    target_in_sight_.store(false);
  }
  state_pub_->publish(state_msg);
}

void StateEstimationNode::cam_imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
  if (!image_tf_)
    return;

  // accumulate angular velocity into a vector
  cam_ang_vel_accumulator_->add(
      d2f(msg->angular_velocity.x),
      d2f(msg->angular_velocity.y),
      d2f(msg->angular_velocity.z));

  //  rclcpp::Time time = get_correct_fusion_time(msg->header, true);
  //  geometry_msgs::msg::TransformStamped base_link_enu;
  //  bool succ = tf_lookup_helper(base_link_enu, "odom", "base_link", time);
  //  if (!succ) return;
  //  const auto base_T_odom = tf_msg_to_affine(base_link_enu);
  //  const auto img_T_base = tf_msg_to_affine(*image_tf_);
  //  const auto cam_R_enu = base_T_odom.rotation() * img_T_base.rotation();

  //  const Eigen::Vector3f accel{d2f(msg->linear_acceleration.x),
  //                              d2f(msg->linear_acceleration.y),
  //                              d2f(msg->linear_acceleration.z)};
  //  const Eigen::Vector3f omega{d2f(msg->angular_velocity.x),
  //                              d2f(msg->angular_velocity.y),
  //                              d2f(msg->angular_velocity.z)};
  //  static Eigen::Vector3f arm{0.109f, -0.030f, 0.017f};
  //  estimator_->update_cam_imu_accel(accel, omega, cam_R_enu, arm);
}
