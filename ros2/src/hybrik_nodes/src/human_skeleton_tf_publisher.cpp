#include <chrono>
#include <memory>
#include <string>
#include <cmath>
#include <thread>
#include <atomic>
#include <mutex>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <array>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/magnetic_field.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Vector3.h"
#include "tf2/LinearMath/Matrix3x3.h"

using namespace std::chrono_literals;

class HumanSkeletonTFPublisher : public rclcpp::Node
{
public:
  HumanSkeletonTFPublisher()
  : Node("human_skeleton_tf_publisher")
  {
    sensor_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
      "/sensor/capacitance", 10,
      std::bind(&HumanSkeletonTFPublisher::sensor_callback, this, std::placeholders::_1));

    normalized_sensor_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
      "/sensor/normalized", 10);

    imu_relative_transform_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
      "/imu/relative_transform", 10);

    for (int i = 0; i < kImuCount; ++i) {
      std::string topic = "/imu/channel_" + std::to_string(i);
      imu_subs_[i] = this->create_subscription<sensor_msgs::msg::Imu>(
        topic, 10,
        [this, i](const sensor_msgs::msg::Imu::SharedPtr msg) {
          this->imu_callback(i, msg);
        });
    }

    for (int i = 0; i < kImuCount; ++i) {
      std::string topic = "/imu/channel_" + std::to_string(i) + "/magnetic_field";
      mag_subs_[i] = this->create_subscription<sensor_msgs::msg::MagneticField>(
        topic, 10,
        [this, i](const sensor_msgs::msg::MagneticField::SharedPtr msg) {
          this->mag_callback(i, msg);
        });
    }

    sensor_data_.resize(kSensorCount, 0.0);
    sensor_data_[0] = 199.0;
    sensor_data_[1] = 276.0;
    sensor_data_[5] = 395.0;
    sensor_data_[6] = 276.0;

    stand_avg_.assign(kSensorCount, 0.0);
    max_values_.assign(kSensorCount, 0.0);

    sensor_candidate_values_.assign(kSensorCount, 0.0);
    sensor_candidate_counts_.assign(kSensorCount, 0);
    sensor_recent_samples_.assign(kSensorCount, std::vector<double>());

    sensor_last_stable_values_.assign(kSensorCount, 0.0);
    sensor_lowpass_values_.assign(kSensorCount, 0.0);
    sensor_lowpass_initialized_.assign(kSensorCount, false);

    max_tracking_samples_.assign(kSensorCount, std::vector<double>());
    max_candidate_values_.assign(kSensorCount, 0.0);
    max_candidate_counts_.assign(kSensorCount, 0);

    imu_initial_accel_.resize(kImuCount, geometry_msgs::msg::Vector3());
    imu_initial_mag_.resize(kImuCount, geometry_msgs::msg::Vector3());
    imu_current_accel_.resize(kImuCount, geometry_msgs::msg::Vector3());
    imu_current_mag_.resize(kImuCount, geometry_msgs::msg::Vector3());
    imu_current_gyro_.resize(kImuCount, geometry_msgs::msg::Vector3());
    imu_gyro_bias_.resize(kImuCount, geometry_msgs::msg::Vector3());
    imu_world_orientation_.resize(kImuCount);
    imu_initial_orientation_.resize(kImuCount);
    imu_filter_initialized_.assign(kImuCount, false);
    imu_last_stamp_ns_.assign(kImuCount, 0);
    imu_calib_accel_sum_.resize(kImuCount, tf2::Vector3(0.0, 0.0, 0.0));
    imu_calib_gyro_sum_.resize(kImuCount, tf2::Vector3(0.0, 0.0, 0.0));
    imu_calib_sample_counts_.assign(kImuCount, 0);

    for (int i = 0; i < kImuCount; ++i) {
      imu_world_orientation_[i] = identity_quaternion();
      imu_initial_orientation_[i] = identity_quaternion();
    }

    calib_thread_ = std::thread(&HumanSkeletonTFPublisher::start_calibration_console, this);
    calib_thread_.detach();

    RCLCPP_INFO(this->get_logger(), "上半身款传感器校准/归一化处理器已启动");
    RCLCPP_INFO(this->get_logger(), "订阅话题: /sensor/capacitance, /imu/channel_*");
    RCLCPP_INFO(this->get_logger(), "发布话题: /sensor/normalized, /imu/relative_transform");
    RCLCPP_INFO(this->get_logger(), "柔性传感器已启用：异常值确认 + 中值滤波 + 低通滤波");
    RCLCPP_INFO(this->get_logger(), "IMU相对姿态已切换为 Acc+Gyro 世界系相对姿态估计");
  }

private:
  static constexpr int kSensorCount = 10;
  static constexpr int kImuCount = 8;

  static constexpr double kGravity = 9.81;
  static constexpr double kGravityTolerance = 2.0;
  static constexpr double kMinVectorNorm = 1e-6;
  static constexpr double kMaxIntegrationDt = 0.1;
  static constexpr double kAccelCorrectionGain = 2.5;
  static constexpr double kStationaryGyroThreshold = 0.25;
  static constexpr double kGyroDeadband = 0.03;
  static constexpr double kBiasAdaptationRate = 2.0;

  static constexpr int kCalibrationSleepMs = 10;
  static constexpr int kSensorCalibrationSamples = 300;      // 3.0s
  static constexpr int kGyroBiasCalibrationSamples = 300;    // 3.0s

  // 柔性传感器抗毛刺参数
  static constexpr double kSensorSpikeAbsTolerance = 15.0;
  static constexpr double kSensorSpikeRelativeTolerance = 0.06;
  static constexpr int kSensorSpikeConfirmationFrames = 3;
  static constexpr size_t kSensorMedianWindowSize = 7;
  static constexpr double kSensorLowpassAlpha = 0.25;

  static constexpr int kMaxCalibrationConfirmationFrames = 3;
  static constexpr size_t kMaxStableSampleTailCount = 10;

  static tf2::Quaternion identity_quaternion()
  {
    return tf2::Quaternion(0.0, 0.0, 0.0, 1.0);
  }

  static tf2::Vector3 msg_to_tf(const geometry_msgs::msg::Vector3 & msg)
  {
    return tf2::Vector3(msg.x, msg.y, msg.z);
  }

  static geometry_msgs::msg::Vector3 tf_to_msg(const tf2::Vector3 & v)
  {
    geometry_msgs::msg::Vector3 msg;
    msg.x = v.x();
    msg.y = v.y();
    msg.z = v.z();
    return msg;
  }

  static double clamp_unit(double value)
  {
    return std::max(-1.0, std::min(1.0, value));
  }

  static tf2::Vector3 normalized_or_zero(const tf2::Vector3 & v)
  {
    tf2::Vector3 result = v;
    if (result.length() < kMinVectorNorm) {
      return tf2::Vector3(0.0, 0.0, 0.0);
    }
    result.normalize();
    return result;
  }

  static tf2::Vector3 rotate_vector(const tf2::Quaternion & q, const tf2::Vector3 & v)
  {
    return tf2::Matrix3x3(q) * v;
  }

  static tf2::Quaternion quaternion_from_two_vectors(const tf2::Vector3 & from_raw, const tf2::Vector3 & to_raw)
  {
    tf2::Vector3 from = normalized_or_zero(from_raw);
    tf2::Vector3 to = normalized_or_zero(to_raw);

    if (from.length() < kMinVectorNorm || to.length() < kMinVectorNorm) {
      return identity_quaternion();
    }

    const double dot = clamp_unit(from.dot(to));

    if (dot > 1.0 - 1e-8) {
      return identity_quaternion();
    }

    if (dot < -1.0 + 1e-8) {
      tf2::Vector3 axis = from.cross(tf2::Vector3(1.0, 0.0, 0.0));
      if (axis.length() < kMinVectorNorm) {
        axis = from.cross(tf2::Vector3(0.0, 1.0, 0.0));
      }
      axis.normalize();
      tf2::Quaternion q(axis, M_PI);
      q.normalize();
      return q;
    }

    tf2::Vector3 cross = from.cross(to);
    const double s = std::sqrt((1.0 + dot) * 2.0);
    const double inv_s = 1.0 / s;

    tf2::Quaternion q(
      cross.x() * inv_s,
      cross.y() * inv_s,
      cross.z() * inv_s,
      0.5 * s);
    q.normalize();
    return q;
  }

  static tf2::Quaternion scaled_quaternion(tf2::Quaternion q, double scale)
  {
    if (scale <= 0.0) {
      return identity_quaternion();
    }
    if (scale >= 1.0) {
      q.normalize();
      return q;
    }

    q.normalize();
    if (q.w() < 0.0) {
      q = tf2::Quaternion(-q.x(), -q.y(), -q.z(), -q.w());
    }

    const double w = clamp_unit(q.w());
    const double angle = 2.0 * std::acos(w);
    const double sin_half = std::sqrt(std::max(0.0, 1.0 - w * w));

    if (sin_half < 1e-8 || angle < 1e-8) {
      return identity_quaternion();
    }

    tf2::Vector3 axis(q.x() / sin_half, q.y() / sin_half, q.z() / sin_half);
    axis.normalize();

    tf2::Quaternion result(axis, angle * scale);
    result.normalize();
    return result;
  }

  static tf2::Vector3 world_down()
  {
    return tf2::Vector3(0.0, 1.0, 0.0);
  }

  static bool has_meaningful_accel(const tf2::Vector3 & accel)
  {
    return accel.length() >= 0.1;
  }

  static bool is_gravity_like(const tf2::Vector3 & accel)
  {
    const double norm = accel.length();
    return norm >= (kGravity - kGravityTolerance) && norm <= (kGravity + kGravityTolerance);
  }

  static bool is_stationary_sample(const tf2::Vector3 & accel, const tf2::Vector3 & gyro)
  {
    return is_gravity_like(accel) && gyro.length() <= kStationaryGyroThreshold;
  }

  static tf2::Vector3 apply_gyro_deadband(const tf2::Vector3 & gyro)
  {
    tf2::Vector3 filtered = gyro;

    if (std::abs(filtered.x()) < kGyroDeadband) {
      filtered.setX(0.0);
    }
    if (std::abs(filtered.y()) < kGyroDeadband) {
      filtered.setY(0.0);
    }
    if (std::abs(filtered.z()) < kGyroDeadband) {
      filtered.setZ(0.0);
    }

    return filtered;
  }

  static tf2::Quaternion world_orientation_from_accel(const tf2::Vector3 & accel)
  {
    if (!has_meaningful_accel(accel)) {
      return identity_quaternion();
    }

    tf2::Vector3 accel_dir = normalized_or_zero(accel);
    tf2::Quaternion q = quaternion_from_two_vectors(accel_dir, world_down());
    q.normalize();
    return q;
  }

  static tf2::Quaternion integrate_world_orientation(
    const tf2::Quaternion & q_world,
    const tf2::Vector3 & gyro_body,
    double dt)
  {
    if (dt <= 0.0) {
      return q_world;
    }

    tf2::Vector3 omega_world = rotate_vector(q_world, gyro_body);
    const double angle = omega_world.length() * dt;

    if (angle < 1e-9) {
      return q_world;
    }

    tf2::Vector3 axis = normalized_or_zero(omega_world);
    tf2::Quaternion delta(axis, angle);
    tf2::Quaternion result = delta * q_world;
    result.normalize();
    return result;
  }

  static tf2::Quaternion fully_align_orientation_to_accel(
    const tf2::Quaternion & q_world,
    const tf2::Vector3 & accel)
  {
    if (!has_meaningful_accel(accel)) {
      return q_world;
    }

    tf2::Vector3 accel_dir = normalized_or_zero(accel);
    tf2::Vector3 accel_world = rotate_vector(q_world, accel_dir);
    tf2::Quaternion correction = quaternion_from_two_vectors(accel_world, world_down());

    tf2::Quaternion result = correction * q_world;
    result.normalize();
    return result;
  }

  static tf2::Quaternion apply_accel_correction(
    const tf2::Quaternion & q_world,
    const tf2::Vector3 & accel,
    double dt)
  {
    if (!is_gravity_like(accel) || dt <= 0.0) {
      return q_world;
    }

    tf2::Vector3 accel_dir = normalized_or_zero(accel);
    tf2::Vector3 accel_world = rotate_vector(q_world, accel_dir);
    tf2::Quaternion correction_full = quaternion_from_two_vectors(accel_world, world_down());

    const double blend = std::max(0.0, std::min(1.0, kAccelCorrectionGain * dt));
    tf2::Quaternion correction = scaled_quaternion(correction_full, blend);

    tf2::Quaternion result = correction * q_world;
    result.normalize();
    return result;
  }

  static double compute_median(std::vector<double> values)
  {
    if (values.empty()) {
      return 0.0;
    }

    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;

    if (values.size() % 2 == 0) {
      return (values[mid - 1] + values[mid]) / 2.0;
    }
    return values[mid];
  }

  static double median_of_recent_samples(std::vector<double> values)
  {
    return compute_median(std::move(values));
  }

  static double compute_asymmetric_trimmed_mean(
    std::vector<double> values,
    double lower_trim_ratio,
    double upper_trim_ratio)
  {
    if (values.empty()) {
      return 0.0;
    }

    std::sort(values.begin(), values.end());

    const size_t lower_cut = static_cast<size_t>(std::floor(values.size() * lower_trim_ratio));
    const size_t upper_cut = static_cast<size_t>(std::floor(values.size() * upper_trim_ratio));

    const size_t begin = std::min(lower_cut, values.size());
    const size_t end = values.size() > upper_cut ? values.size() - upper_cut : 0;

    if (begin >= end) {
      return compute_median(values);
    }

    double sum = 0.0;
    for (size_t i = begin; i < end; ++i) {
      sum += values[i];
    }
    return sum / static_cast<double>(end - begin);
  }

  static double sensor_spike_tolerance(double reference)
  {
    return std::max(kSensorSpikeAbsTolerance, std::abs(reference) * kSensorSpikeRelativeTolerance);
  }

  static double max_confirmation_tolerance(double reference)
  {
    return std::max(kSensorSpikeAbsTolerance, std::abs(reference) * kSensorSpikeRelativeTolerance);
  }

  static double estimate_stable_max_from_samples(
    std::vector<double> samples,
    double fallback_value)
  {
    if (samples.empty()) {
      return fallback_value;
    }

    std::sort(samples.begin(), samples.end());

    const size_t tail_count = std::min(kMaxStableSampleTailCount, samples.size());
    const size_t tail_begin = samples.size() - tail_count;
    const size_t drop_top = static_cast<size_t>(std::floor(tail_count * 0.10));
    const size_t usable_end = samples.size() - drop_top;

    if (tail_begin >= usable_end) {
      return samples.back();
    }

    double sum = 0.0;
    for (size_t i = tail_begin; i < usable_end; ++i) {
      sum += samples[i];
    }

    return sum / static_cast<double>(usable_end - tail_begin);
  }

  void reset_sensor_spike_filter_locked()
  {
    sensor_candidate_values_.assign(kSensorCount, 0.0);
    sensor_candidate_counts_.assign(kSensorCount, 0);
    sensor_recent_samples_.assign(kSensorCount, std::vector<double>());
    sensor_last_stable_values_.assign(kSensorCount, 0.0);
    sensor_lowpass_values_.assign(kSensorCount, 0.0);
    sensor_lowpass_initialized_.assign(kSensorCount, false);
    sensor_filter_initialized_ = false;
  }

  void reset_max_tracking_state_locked(const std::vector<double> & initial_values)
  {
    max_tracking_samples_.assign(kSensorCount, std::vector<double>());
    max_candidate_values_.assign(kSensorCount, 0.0);
    max_candidate_counts_.assign(kSensorCount, 0);
    max_values_ = initial_values;

    if (max_values_.size() != kSensorCount) {
      max_values_.assign(kSensorCount, 0.0);
    }

    for (size_t i = 0; i < kSensorCount && i < initial_values.size(); ++i) {
      max_candidate_values_[i] = initial_values[i];
      max_candidate_counts_[i] = 1;
    }
  }

  double filter_single_sensor_sample_locked(size_t index, double incoming)
  {
    if (!sensor_filter_initialized_) {
      sensor_last_stable_values_[index] = incoming;
      sensor_candidate_values_[index] = incoming;
      sensor_candidate_counts_[index] = 0;

      auto & recent = sensor_recent_samples_[index];
      recent.clear();
      recent.push_back(incoming);

      sensor_lowpass_values_[index] = incoming;
      sensor_lowpass_initialized_[index] = true;
      return incoming;
    }

    double stable_reference = sensor_last_stable_values_[index];
    double accepted = stable_reference;
    const double tolerance = sensor_spike_tolerance(stable_reference);

    if (std::abs(incoming - stable_reference) <= tolerance) {
      accepted = incoming;
      sensor_last_stable_values_[index] = incoming;
      sensor_candidate_values_[index] = incoming;
      sensor_candidate_counts_[index] = 0;
    } else {
      if (sensor_candidate_counts_[index] == 0 ||
          std::abs(incoming - sensor_candidate_values_[index]) > tolerance) {
        sensor_candidate_values_[index] = incoming;
        sensor_candidate_counts_[index] = 1;
      } else {
        sensor_candidate_values_[index] =
          (sensor_candidate_values_[index] * static_cast<double>(sensor_candidate_counts_[index]) +
           incoming) /
          static_cast<double>(sensor_candidate_counts_[index] + 1);
        ++sensor_candidate_counts_[index];
      }

      if (sensor_candidate_counts_[index] >= kSensorSpikeConfirmationFrames) {
        accepted = sensor_candidate_values_[index];
        sensor_last_stable_values_[index] = accepted;
        sensor_candidate_counts_[index] = 0;
      } else {
        accepted = stable_reference;
      }
    }

    auto & recent = sensor_recent_samples_[index];
    recent.push_back(accepted);
    if (recent.size() > kSensorMedianWindowSize) {
      recent.erase(recent.begin());
    }

    double median_value = median_of_recent_samples(recent);

    if (!sensor_lowpass_initialized_[index]) {
      sensor_lowpass_values_[index] = median_value;
      sensor_lowpass_initialized_[index] = true;
    } else {
      sensor_lowpass_values_[index] =
        kSensorLowpassAlpha * median_value +
        (1.0 - kSensorLowpassAlpha) * sensor_lowpass_values_[index];
    }

    return sensor_lowpass_values_[index];
  }

  std::vector<double> filter_sensor_frame_locked(const std::vector<float> & values)
  {
    std::vector<double> filtered(kSensorCount, 0.0);
    if (values.size() != kSensorCount) {
      return filtered;
    }

    if (sensor_data_.size() != kSensorCount) {
      sensor_data_.assign(kSensorCount, 0.0);
    }
    if (sensor_recent_samples_.size() != kSensorCount) {
      sensor_recent_samples_.assign(kSensorCount, std::vector<double>());
    }
    if (sensor_last_stable_values_.size() != kSensorCount) {
      sensor_last_stable_values_.assign(kSensorCount, 0.0);
    }
    if (sensor_lowpass_values_.size() != kSensorCount) {
      sensor_lowpass_values_.assign(kSensorCount, 0.0);
    }
    if (sensor_lowpass_initialized_.size() != kSensorCount) {
      sensor_lowpass_initialized_.assign(kSensorCount, false);
    }

    for (size_t i = 0; i < kSensorCount; ++i) {
      const double incoming = static_cast<double>(values[i]);
      filtered[i] = filter_single_sensor_sample_locked(i, incoming);
    }

    sensor_data_ = filtered;
    sensor_filter_initialized_ = true;
    return filtered;
  }

  void update_max_tracking_locked(const std::vector<double> & values)
  {
    if (values.size() != kSensorCount) {
      return;
    }

    if (max_values_.size() != kSensorCount) {
      max_values_.assign(kSensorCount, 0.0);
    }
    if (max_candidate_values_.size() != kSensorCount) {
      max_candidate_values_.assign(kSensorCount, 0.0);
    }
    if (max_candidate_counts_.size() != kSensorCount) {
      max_candidate_counts_.assign(kSensorCount, 0);
    }
    if (max_tracking_samples_.size() != kSensorCount) {
      max_tracking_samples_.assign(kSensorCount, std::vector<double>());
    }

    for (size_t i = 0; i < kSensorCount; ++i) {
      const double value = values[i];

      const double reference =
        max_candidate_counts_[i] > 0 ? max_candidate_values_[i] : value;
      const double tolerance = max_confirmation_tolerance(reference);

      if (max_candidate_counts_[i] == 0 || std::abs(value - reference) > tolerance) {
        max_candidate_values_[i] = value;
        max_candidate_counts_[i] = 1;
        continue;
      }

      max_candidate_values_[i] =
        (max_candidate_values_[i] * static_cast<double>(max_candidate_counts_[i]) + value) /
        static_cast<double>(max_candidate_counts_[i] + 1);
      ++max_candidate_counts_[i];

      if (max_candidate_counts_[i] >= kMaxCalibrationConfirmationFrames) {
        max_tracking_samples_[i].push_back(max_candidate_values_[i]);
        max_values_[i] = estimate_stable_max_from_samples(max_tracking_samples_[i], max_values_[i]);
      }
    }
  }

  void reset_imu_calibration_capture_locked()
  {
    for (int i = 0; i < kImuCount; ++i) {
      imu_calib_accel_sum_[i] = tf2::Vector3(0.0, 0.0, 0.0);
      imu_calib_gyro_sum_[i] = tf2::Vector3(0.0, 0.0, 0.0);
      imu_calib_sample_counts_[i] = 0;
    }
  }

  void accumulate_imu_calibration_sample_locked(int channel, const sensor_msgs::msg::Imu & msg)
  {
    imu_calib_accel_sum_[channel] += msg_to_tf(msg.linear_acceleration);
    imu_calib_gyro_sum_[channel] += msg_to_tf(msg.angular_velocity);
    ++imu_calib_sample_counts_[channel];
  }

  void finalize_imu_initial_calibration_locked()
  {
    for (int i = 0; i < kImuCount; ++i) {
      tf2::Vector3 avg_accel = msg_to_tf(imu_current_accel_[i]);
      imu_initial_accel_[i] = tf_to_msg(avg_accel);

      tf2::Quaternion seed_orientation = imu_filter_initialized_[i] ?
        imu_world_orientation_[i] : world_orientation_from_accel(avg_accel);
      tf2::Quaternion q_initial = fully_align_orientation_to_accel(seed_orientation, avg_accel);
      q_initial.normalize();

      imu_initial_orientation_[i] = q_initial;
      imu_world_orientation_[i] = q_initial;
      imu_filter_initialized_[i] = has_meaningful_accel(avg_accel) || imu_filter_initialized_[i];
      imu_last_stamp_ns_[i] = 0;
    }
  }

  void update_imu_world_orientation_locked(int channel, const sensor_msgs::msg::Imu & msg)
  {
    const int64_t stamp_ns = rclcpp::Time(msg.header.stamp).nanoseconds();
    const tf2::Vector3 accel = msg_to_tf(msg.linear_acceleration);
    const tf2::Vector3 gyro = msg_to_tf(msg.angular_velocity);

    if (!imu_filter_initialized_[channel]) {
      if (is_gravity_like(accel)) {
        imu_world_orientation_[channel] = world_orientation_from_accel(accel);
        imu_world_orientation_[channel].normalize();
        imu_filter_initialized_[channel] = true;
      }
      imu_last_stamp_ns_[channel] = stamp_ns;
      return;
    }

    if (imu_last_stamp_ns_[channel] <= 0 || stamp_ns <= imu_last_stamp_ns_[channel]) {
      imu_last_stamp_ns_[channel] = stamp_ns;
      return;
    }

    double dt = static_cast<double>(stamp_ns - imu_last_stamp_ns_[channel]) * 1e-9;
    imu_last_stamp_ns_[channel] = stamp_ns;

    if (dt <= 0.0 || dt > kMaxIntegrationDt) {
      return;
    }

    tf2::Vector3 bias = msg_to_tf(imu_gyro_bias_[channel]);
    tf2::Vector3 gyro_unbiased = gyro - bias;

    if (is_stationary_sample(accel, gyro_unbiased)) {
      const double alpha = std::max(0.0, std::min(1.0, kBiasAdaptationRate * dt));
      tf2::Vector3 updated_bias = bias * (1.0 - alpha) + gyro * alpha;
      imu_gyro_bias_[channel] = tf_to_msg(updated_bias);
      gyro_unbiased = gyro - updated_bias;
    }

    gyro_unbiased = apply_gyro_deadband(gyro_unbiased);

    tf2::Quaternion q_world = imu_world_orientation_[channel];
    q_world = integrate_world_orientation(q_world, gyro_unbiased, dt);
    q_world = apply_accel_correction(q_world, accel, dt);
    q_world.normalize();

    imu_world_orientation_[channel] = q_world;
  }

  void sensor_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
  {
    if (msg->data.size() != kSensorCount) {
      return;
    }

    std::vector<double> filtered_sensor_data;
    bool should_publish = false;
    {
      std::lock_guard<std::mutex> lk(calib_mutex_);
      filtered_sensor_data = filter_sensor_frame_locked(msg->data);

      // 注意：最大值跟踪现在也只吃“去异常 + 滤波后”的数据
      if (track_max_) {
        update_max_tracking_locked(filtered_sensor_data);
      }

      should_publish = calibration_ready_ && calib_loaded_;
    }

    if (should_publish) {
      publish_normalized_sensors();
    }

    if (calibration_ready_) {
      static int sensor_counter = 0;
      if (++sensor_counter % 100 == 0) {
        RCLCPP_INFO(this->get_logger(),
          "上半身传感器数据(滤波后) - 0:%.1f 1:%.1f 5:%.1f 6:%.1f",
          filtered_sensor_data[0], filtered_sensor_data[1], filtered_sensor_data[5],
          filtered_sensor_data[6]);
      }
    }
  }

  void imu_callback(int channel, const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    bool should_publish = false;

    {
      std::lock_guard<std::mutex> lk(calib_mutex_);
      imu_current_accel_[channel] = msg->linear_acceleration;
      imu_current_gyro_[channel] = msg->angular_velocity;

      if (record_imu_initial_) {
        accumulate_imu_calibration_sample_locked(channel, *msg);
        imu_initial_accel_[channel] = msg->linear_acceleration;
      }

      update_imu_world_orientation_locked(channel, *msg);
      should_publish = calibration_ready_ && calib_loaded_ && channel == 0;
    }

    if (should_publish) {
      publish_imu_relative_transforms();
    }

    if (calibration_ready_) {
      static std::vector<int> counters(kImuCount, 0);
      if (++counters[channel] % 100 == 0) {
        tf2::Vector3 bias;
        tf2::Vector3 corrected;
        {
          std::lock_guard<std::mutex> lk(calib_mutex_);
          bias = msg_to_tf(imu_gyro_bias_[channel]);
          corrected = apply_gyro_deadband(msg_to_tf(imu_current_gyro_[channel]) - bias);
        }
        RCLCPP_INFO(this->get_logger(),
          "IMU%d 加速度:[%.2f, %.2f, %.2f] 原始角速度:[%.3f, %.3f, %.3f] bias:[%.3f, %.3f, %.3f] 校正后:[%.3f, %.3f, %.3f]",
          channel,
          imu_current_accel_[channel].x, imu_current_accel_[channel].y, imu_current_accel_[channel].z,
          imu_current_gyro_[channel].x, imu_current_gyro_[channel].y, imu_current_gyro_[channel].z,
          bias.x(), bias.y(), bias.z(),
          corrected.x(), corrected.y(), corrected.z());
      }
    }
  }

  void mag_callback(int channel, const sensor_msgs::msg::MagneticField::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lk(calib_mutex_);
    imu_current_mag_[channel] = msg->magnetic_field;

    if (record_imu_initial_) {
      imu_initial_mag_[channel] = msg->magnetic_field;
    }

    if (calibration_ready_) {
      static std::vector<int> counters(kImuCount, 0);
      if (++counters[channel] % 100 == 0) {
        RCLCPP_INFO(this->get_logger(),
          "IMU%d 磁场: [%.2f, %.2f, %.2f]",
          channel,
          imu_current_mag_[channel].x, imu_current_mag_[channel].y, imu_current_mag_[channel].z);
      }
    }
  }

  void publish_imu_relative_transforms()
  {
    std_msgs::msg::Float32MultiArray transform_msg;
    transform_msg.data.resize(kImuCount * 4);

    std::lock_guard<std::mutex> lk(calib_mutex_);

    for (int i = 0; i < kImuCount; ++i) {
      tf2::Quaternion q_initial = imu_initial_orientation_[i];
      tf2::Quaternion q_current = imu_filter_initialized_[i] ?
        imu_world_orientation_[i] : imu_initial_orientation_[i];

      tf2::Quaternion q_relative = q_current.inverse() * q_initial;
      q_relative.normalize();

      transform_msg.data[i * 4 + 0] = static_cast<float>(q_relative.x());
      transform_msg.data[i * 4 + 1] = static_cast<float>(q_relative.y());
      transform_msg.data[i * 4 + 2] = static_cast<float>(q_relative.z());
      transform_msg.data[i * 4 + 3] = static_cast<float>(q_relative.w());
    }

    imu_relative_transform_pub_->publish(transform_msg);

    static int imu_counter = 0;
    if (++imu_counter % 100 == 0) {
      tf2::Quaternion q0(
        transform_msg.data[0],
        transform_msg.data[1],
        transform_msg.data[2],
        transform_msg.data[3]);
      tf2::Matrix3x3 m(q0);
      double roll, pitch, yaw;
      m.getRPY(roll, pitch, yaw);
      RCLCPP_INFO(this->get_logger(),
        "IMU0世界系相对变换 - Roll:%.1f° Pitch:%.1f° Yaw:%.1f° (四元数:[%.3f,%.3f,%.3f,%.3f])",
        roll * 180.0 / M_PI, pitch * 180.0 / M_PI, yaw * 180.0 / M_PI,
        q0.x(), q0.y(), q0.z(), q0.w());
    }
  }

  void publish_normalized_sensors()
  {
    std_msgs::msg::Float32MultiArray normalized_msg;
    normalized_msg.data.resize(kSensorCount);

    std::lock_guard<std::mutex> lk(calib_mutex_);

    for (size_t i = 0; i < kSensorCount; ++i) {
      double min_val = stand_avg_[i];
      double max_val = max_values_[i];
      double current_val = sensor_data_[i];

      double normalized = 0.0;
      if (max_val > min_val) {
        normalized = (current_val - min_val) / (max_val - min_val);
        normalized = std::max(0.0, std::min(1.0, normalized));
      }

      normalized_msg.data[i] = static_cast<float>(normalized);
    }

    normalized_sensor_pub_->publish(normalized_msg);

    static int norm_counter = 0;
    if (++norm_counter % 100 == 0) {
      RCLCPP_INFO(this->get_logger(),
        "上半身归一化传感器 (0-1) - 0:%.2f 1:%.2f 5:%.2f 6:%.2f",
        normalized_msg.data[0], normalized_msg.data[1], normalized_msg.data[5],
        normalized_msg.data[6]);
    }
  }

  std::string calib_file_path() const
  {
    const char * home = std::getenv("HOME");
    std::string base = home ? std::string(home) : std::string(".");
    std::string dir = base + "/.ros";
    std::filesystem::create_directories(dir);
    return dir + "/human_upper_body_calibration.txt";
  }

  bool load_calibration_file()
  {
    std::string path = calib_file_path();
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
      RCLCPP_WARN(this->get_logger(), "未找到校准文件: %s", path.c_str());
      return false;
    }

    std::vector<double> stand(kSensorCount, 0.0), mx(kSensorCount, 0.0);
    std::vector<geometry_msgs::msg::Vector3> imu_accel(kImuCount), imu_mag(kImuCount), gyro_bias(kImuCount);
    std::vector<tf2::Quaternion> imu_q_init(kImuCount, identity_quaternion());
    std::vector<bool> ok_imu_accel(kImuCount, false), ok_imu_mag(kImuCount, false);
    std::vector<bool> ok_gyro_bias(kImuCount, false), ok_q_init(kImuCount, false);

    auto parse_list = [](const std::string & s, std::vector<double> & out) -> bool {
      auto pos = s.find(':');
      if (pos == std::string::npos) {
        return false;
      }
      std::string values = s.substr(pos + 1);
      values.erase(std::remove(values.begin(), values.end(), ' '), values.end());

      std::stringstream ss(values);
      std::string item;
      size_t idx = 0;
      while (std::getline(ss, item, ',') && idx < out.size()) {
        try {
          out[idx++] = std::stod(item);
        } catch (...) {
          return false;
        }
      }
      return idx == out.size();
    };

    auto parse_vec3 = [](const std::string & s, geometry_msgs::msg::Vector3 & out) -> bool {
      auto pos = s.find(':');
      if (pos == std::string::npos) {
        return false;
      }
      std::string values = s.substr(pos + 1);
      values.erase(std::remove(values.begin(), values.end(), ' '), values.end());

      std::stringstream ss(values);
      std::string item;
      std::vector<double> v;
      while (std::getline(ss, item, ',')) {
        try {
          v.push_back(std::stod(item));
        } catch (...) {
          return false;
        }
      }
      if (v.size() != 3) {
        return false;
      }
      out.x = v[0];
      out.y = v[1];
      out.z = v[2];
      return true;
    };

    auto parse_quat = [](const std::string & s, tf2::Quaternion & out) -> bool {
      auto pos = s.find(':');
      if (pos == std::string::npos) {
        return false;
      }
      std::string values = s.substr(pos + 1);
      values.erase(std::remove(values.begin(), values.end(), ' '), values.end());

      std::stringstream ss(values);
      std::string item;
      std::vector<double> v;
      while (std::getline(ss, item, ',')) {
        try {
          v.push_back(std::stod(item));
        } catch (...) {
          return false;
        }
      }
      if (v.size() != 4) {
        return false;
      }
      out = tf2::Quaternion(v[0], v[1], v[2], v[3]);
      out.normalize();
      return true;
    };

    bool ok_stand = false;
    bool ok_max = false;
    std::string line;

    while (std::getline(ifs, line)) {
      if (line.rfind("stand:", 0) == 0) {
        ok_stand = parse_list(line, stand);
      } else if (line.rfind("max:", 0) == 0) {
        ok_max = parse_list(line, mx);
      } else {
        for (int i = 0; i < kImuCount; ++i) {
          std::string key_accel = "imu_accel_" + std::to_string(i) + ":";
          std::string key_mag = "imu_mag_" + std::to_string(i) + ":";
          std::string key_bias = "gyro_bias_" + std::to_string(i) + ":";
          std::string key_q = "imu_q_init_" + std::to_string(i) + ":";

          if (line.rfind(key_accel, 0) == 0) {
            ok_imu_accel[i] = parse_vec3(line, imu_accel[i]);
          } else if (line.rfind(key_mag, 0) == 0) {
            ok_imu_mag[i] = parse_vec3(line, imu_mag[i]);
          } else if (line.rfind(key_bias, 0) == 0) {
            ok_gyro_bias[i] = parse_vec3(line, gyro_bias[i]);
          } else if (line.rfind(key_q, 0) == 0) {
            ok_q_init[i] = parse_quat(line, imu_q_init[i]);
          }
        }
      }
    }

    if (!ok_stand || !ok_max) {
      RCLCPP_ERROR(this->get_logger(), "校准文件解析失败，stand/max 字段不完整");
      return false;
    }

    {
      std::lock_guard<std::mutex> lk(calib_mutex_);
      stand_avg_ = stand;
      max_values_ = mx;

      for (int i = 0; i < kImuCount; ++i) {
        if (ok_imu_accel[i]) {
          imu_initial_accel_[i] = imu_accel[i];
        }
        if (ok_imu_mag[i]) {
          imu_initial_mag_[i] = imu_mag[i];
        }
        if (ok_gyro_bias[i]) {
          imu_gyro_bias_[i] = gyro_bias[i];
        }
        if (ok_q_init[i]) {
          imu_initial_orientation_[i] = imu_q_init[i];
          imu_world_orientation_[i] = imu_q_init[i];
          imu_filter_initialized_[i] = true;
          imu_last_stamp_ns_[i] = 0;
        }
      }

      reset_sensor_spike_filter_locked();
      reset_max_tracking_state_locked(max_values_);
    }

    calib_loaded_ = true;
    calibration_ready_ = true;

    RCLCPP_INFO(this->get_logger(), "已加载校准文件: %s", path.c_str());
    return true;
  }

  bool save_calibration_file()
  {
    std::string path = calib_file_path();
    std::ofstream ofs(path, std::ios::trunc);
    if (!ofs.is_open()) {
      RCLCPP_ERROR(this->get_logger(), "无法写入校准文件: %s", path.c_str());
      return false;
    }

    std::lock_guard<std::mutex> lk(calib_mutex_);

    auto write_list = [&ofs](const std::string & key, const std::vector<double> & values) {
      ofs << key << ": ";
      for (size_t i = 0; i < values.size(); ++i) {
        ofs << std::fixed << std::setprecision(6) << values[i];
        if (i + 1 < values.size()) {
          ofs << ", ";
        }
      }
      ofs << "\n";
    };

    auto write_vec3 = [&ofs](const std::string & key, const geometry_msgs::msg::Vector3 & v) {
      ofs << key << ": "
          << std::fixed << std::setprecision(6)
          << v.x << ", " << v.y << ", " << v.z << "\n";
    };

    auto write_quat = [&ofs](const std::string & key, const tf2::Quaternion & q) {
      ofs << key << ": "
          << std::fixed << std::setprecision(6)
          << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << "\n";
    };

    write_list("stand", stand_avg_);
    write_list("max", max_values_);

    for (int i = 0; i < kImuCount; ++i) {
      write_vec3("imu_accel_" + std::to_string(i), imu_initial_accel_[i]);
      write_vec3("imu_mag_" + std::to_string(i), imu_initial_mag_[i]);
      write_vec3("gyro_bias_" + std::to_string(i), imu_gyro_bias_[i]);
      write_quat("imu_q_init_" + std::to_string(i), imu_initial_orientation_[i]);
    }

    ofs.close();
    calib_loaded_ = true;

    RCLCPP_INFO(this->get_logger(), "校准文件已保存: %s", path.c_str());
    return true;
  }

  void do_stand_calibration_3s()
  {
    std::vector<std::vector<double>> all_samples;
    all_samples.reserve(kSensorCalibrationSamples);

    for (int k = 0; k < kSensorCalibrationSamples; ++k) {
      std::vector<double> snap;
      {
        std::lock_guard<std::mutex> lk(calib_mutex_);
        snap = sensor_data_;  // 已经是 去异常+滤波 后的数据
      }

      if (snap.size() != kSensorCount) {
        snap.assign(kSensorCount, 0.0);
      }
      all_samples.push_back(snap);
      std::this_thread::sleep_for(std::chrono::milliseconds(kCalibrationSleepMs));
    }

    std::vector<double> avg(kSensorCount, 0.0);
    for (int i = 0; i < kSensorCount; ++i) {
      std::vector<double> sensor_values;
      sensor_values.reserve(all_samples.size());
      for (const auto & sample : all_samples) {
        sensor_values.push_back(sample[i]);
      }
      avg[i] = compute_asymmetric_trimmed_mean(sensor_values, 0.65, 0.20);
    }

    {
      std::lock_guard<std::mutex> lk(calib_mutex_);
      stand_avg_ = avg;
    }

    RCLCPP_INFO(this->get_logger(), "站立校准完成（已基于去异常+滤波后的数据计算）。");
  }

  void do_gyro_bias_calibration_3s()
  {
    for (int k = 0; k < kGyroBiasCalibrationSamples; ++k) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kCalibrationSleepMs));
    }

    std::lock_guard<std::mutex> lk(calib_mutex_);
    for (int i = 0; i < kImuCount; ++i) {
      tf2::Vector3 avg_gyro = msg_to_tf(imu_current_gyro_[i]);
      if (imu_calib_sample_counts_[i] > 0) {
        avg_gyro = imu_calib_gyro_sum_[i] / static_cast<double>(imu_calib_sample_counts_[i]);
      }
      imu_gyro_bias_[i] = tf_to_msg(avg_gyro);
    }

    RCLCPP_INFO(this->get_logger(), "IMU陀螺零偏校准完成 (3秒平均)。");
  }

  void start_calibration_console()
  {
    std::cout << "\n=== 校准模式 ===\n";
    std::cout << "请选择一个选项:\n";
    std::cout << "  1 : 开始新的校准\n";
    std::cout << "  2 : 加载现有校准文件\n";
    std::cout << "  3 : 使用默认值 (不校准)\n";
    std::cout << "输入选择 (1/2/3): " << std::flush;

    std::string choice;
    if (!std::getline(std::cin, choice)) {
      std::cout << "[警告] 读取输入失败。使用默认值。\n\n";
      calibration_ready_ = true;
      return;
    }

    if (choice == "2") {
      if (load_calibration_file()) {
        std::cout << "[信息] 校准文件加载成功。\n";
        std::cout << "[信息] 启动TF发布器。\n\n";
        calibration_ready_ = true;
        return;
      } else {
        std::cout << "[警告] 校准文件加载失败，将进入新校准流程。\n";
      }
    } else if (choice == "3") {
      std::lock_guard<std::mutex> lk(calib_mutex_);
      stand_avg_ = std::vector<double>(kSensorCount, 0.0);
      max_values_ = std::vector<double>(kSensorCount, 1000.0);
      calib_loaded_ = true;
      calibration_ready_ = true;
      std::cout << "[信息] 使用默认值启动。\n\n";
      return;
    }

    std::cout << "\n=== 新校准流程 ===\n";
    std::cout << "命令说明:\n";
    std::cout << "  1 : 开始 3 秒站立校准\n";
    std::cout << "  2 : 开始最大值/姿态校准\n";
    std::cout << "  p : 打印当前校准结果\n";
    std::cout << "  s : 保存并结束校准\n";
    std::cout << "  q : 不保存直接退出\n\n";

    calib_running_ = true;

    while (rclcpp::ok() && calib_running_) {
      std::cout << "输入命令 (1/2/p/s/q): " << std::flush;

      std::string cmd;
      if (!std::getline(std::cin, cmd)) {
        std::cout << "[警告] 控制台输入结束，退出校准。\n";
        calib_running_ = false;
        calibration_ready_ = true;
        break;
      }

      if (cmd == "1") {
        std::cout << "[校准] 请保持标准站立 3 秒...\n";
        do_stand_calibration_3s();
      } else if (cmd == "2") {
        std::cout << "[校准] 开始最大值/IMU 初始姿态校准。\n";
        std::cout << "[校准] 请在接下来过程中做满量程动作，异常值会被自动抑制。\n";

        {
          std::lock_guard<std::mutex> lk(calib_mutex_);
          reset_max_tracking_state_locked(sensor_data_);
          reset_imu_calibration_capture_locked();
        }

        record_imu_initial_ = true;
        track_max_ = true;

        std::cout << "[校准] 正在记录，请保持/完成动作后按回车结束..." << std::flush;
        std::string dummy;
        std::getline(std::cin, dummy);

        track_max_ = false;
        record_imu_initial_ = false;

        {
          std::lock_guard<std::mutex> lk(calib_mutex_);
          finalize_imu_initial_calibration_locked();
        }
        do_gyro_bias_calibration_3s();

        std::cout << "[校准] 最大值和 IMU 初始姿态校准完成。\n";
      } else if (cmd == "p" || cmd == "P") {
        std::lock_guard<std::mutex> lk(calib_mutex_);

        std::cout << "\n[当前校准结果]\n";
        std::cout << "stand: ";
        for (size_t i = 0; i < stand_avg_.size(); ++i) {
          std::cout << std::fixed << std::setprecision(2) << stand_avg_[i];
          if (i + 1 < stand_avg_.size()) {
            std::cout << ", ";
          }
        }
        std::cout << "\nmax:   ";
        for (size_t i = 0; i < max_values_.size(); ++i) {
          std::cout << std::fixed << std::setprecision(2) << max_values_[i];
          if (i + 1 < max_values_.size()) {
            std::cout << ", ";
          }
        }
        std::cout << "\nIMU初始姿态: 已记录" << kImuCount << "个IMU的初始四元数；陀螺零偏为3秒静止平均\n\n";
      } else if (cmd == "s" || cmd == "S") {
        track_max_ = false;
        calib_running_ = false;
        save_calibration_file();
        std::cout << "[校准] 已保存并退出校准。\n";
        std::cout << "[信息] 启动TF发布器。\n\n";
        calibration_ready_ = true;
      } else if (cmd == "q" || cmd == "Q") {
        track_max_ = false;
        calib_running_ = false;
        std::cout << "[校准] 退出不保存。\n";
        std::cout << "[信息] 启动TF发布器。\n\n";
        calibration_ready_ = true;
      } else {
        std::cout << "未知命令。使用 1 / 2 / p / s / q。\n";
      }
    }
  }

  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sensor_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr normalized_sensor_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr imu_relative_transform_pub_;
  std::array<rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr, kImuCount> imu_subs_;
  std::array<rclcpp::Subscription<sensor_msgs::msg::MagneticField>::SharedPtr, kImuCount> mag_subs_;

  std::vector<double> sensor_data_;

  // 异常值确认
  std::vector<double> sensor_candidate_values_;
  std::vector<int> sensor_candidate_counts_;

  // 中值滤波窗口
  std::vector<std::vector<double>> sensor_recent_samples_;

  // 稳定参考值 + 低通滤波
  std::vector<double> sensor_last_stable_values_;
  std::vector<double> sensor_lowpass_values_;
  std::vector<bool> sensor_lowpass_initialized_;
  bool sensor_filter_initialized_{false};

  // 最大值跟踪
  std::vector<std::vector<double>> max_tracking_samples_;
  std::vector<double> max_candidate_values_;
  std::vector<int> max_candidate_counts_;

  // IMU
  std::vector<geometry_msgs::msg::Vector3> imu_initial_accel_;
  std::vector<geometry_msgs::msg::Vector3> imu_initial_mag_;
  std::vector<geometry_msgs::msg::Vector3> imu_current_accel_;
  std::vector<geometry_msgs::msg::Vector3> imu_current_mag_;
  std::vector<geometry_msgs::msg::Vector3> imu_current_gyro_;
  std::vector<geometry_msgs::msg::Vector3> imu_gyro_bias_;
  std::vector<tf2::Quaternion> imu_world_orientation_;
  std::vector<tf2::Quaternion> imu_initial_orientation_;
  std::vector<bool> imu_filter_initialized_;
  std::vector<int64_t> imu_last_stamp_ns_;
  std::vector<tf2::Vector3> imu_calib_accel_sum_;
  std::vector<tf2::Vector3> imu_calib_gyro_sum_;
  std::vector<int> imu_calib_sample_counts_;

  std::thread calib_thread_;
  std::atomic<bool> calib_running_{false};
  std::atomic<bool> track_max_{false};
  std::atomic<bool> record_imu_initial_{false};
  std::atomic<bool> calibration_ready_{false};
  std::mutex calib_mutex_;

  std::vector<double> stand_avg_;
  std::vector<double> max_values_;
  std::atomic<bool> calib_loaded_{false};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HumanSkeletonTFPublisher>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
