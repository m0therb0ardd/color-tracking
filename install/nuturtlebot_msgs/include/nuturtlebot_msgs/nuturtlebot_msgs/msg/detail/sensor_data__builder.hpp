// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from nuturtlebot_msgs:msg/SensorData.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "nuturtlebot_msgs/msg/sensor_data.hpp"


#ifndef NUTURTLEBOT_MSGS__MSG__DETAIL__SENSOR_DATA__BUILDER_HPP_
#define NUTURTLEBOT_MSGS__MSG__DETAIL__SENSOR_DATA__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "nuturtlebot_msgs/msg/detail/sensor_data__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace nuturtlebot_msgs
{

namespace msg
{

namespace builder
{

class Init_SensorData_battery_voltage
{
public:
  explicit Init_SensorData_battery_voltage(::nuturtlebot_msgs::msg::SensorData & msg)
  : msg_(msg)
  {}
  ::nuturtlebot_msgs::msg::SensorData battery_voltage(::nuturtlebot_msgs::msg::SensorData::_battery_voltage_type arg)
  {
    msg_.battery_voltage = std::move(arg);
    return std::move(msg_);
  }

private:
  ::nuturtlebot_msgs::msg::SensorData msg_;
};

class Init_SensorData_mag_z
{
public:
  explicit Init_SensorData_mag_z(::nuturtlebot_msgs::msg::SensorData & msg)
  : msg_(msg)
  {}
  Init_SensorData_battery_voltage mag_z(::nuturtlebot_msgs::msg::SensorData::_mag_z_type arg)
  {
    msg_.mag_z = std::move(arg);
    return Init_SensorData_battery_voltage(msg_);
  }

private:
  ::nuturtlebot_msgs::msg::SensorData msg_;
};

class Init_SensorData_mag_y
{
public:
  explicit Init_SensorData_mag_y(::nuturtlebot_msgs::msg::SensorData & msg)
  : msg_(msg)
  {}
  Init_SensorData_mag_z mag_y(::nuturtlebot_msgs::msg::SensorData::_mag_y_type arg)
  {
    msg_.mag_y = std::move(arg);
    return Init_SensorData_mag_z(msg_);
  }

private:
  ::nuturtlebot_msgs::msg::SensorData msg_;
};

class Init_SensorData_mag_x
{
public:
  explicit Init_SensorData_mag_x(::nuturtlebot_msgs::msg::SensorData & msg)
  : msg_(msg)
  {}
  Init_SensorData_mag_y mag_x(::nuturtlebot_msgs::msg::SensorData::_mag_x_type arg)
  {
    msg_.mag_x = std::move(arg);
    return Init_SensorData_mag_y(msg_);
  }

private:
  ::nuturtlebot_msgs::msg::SensorData msg_;
};

class Init_SensorData_gyro_z
{
public:
  explicit Init_SensorData_gyro_z(::nuturtlebot_msgs::msg::SensorData & msg)
  : msg_(msg)
  {}
  Init_SensorData_mag_x gyro_z(::nuturtlebot_msgs::msg::SensorData::_gyro_z_type arg)
  {
    msg_.gyro_z = std::move(arg);
    return Init_SensorData_mag_x(msg_);
  }

private:
  ::nuturtlebot_msgs::msg::SensorData msg_;
};

class Init_SensorData_gyro_y
{
public:
  explicit Init_SensorData_gyro_y(::nuturtlebot_msgs::msg::SensorData & msg)
  : msg_(msg)
  {}
  Init_SensorData_gyro_z gyro_y(::nuturtlebot_msgs::msg::SensorData::_gyro_y_type arg)
  {
    msg_.gyro_y = std::move(arg);
    return Init_SensorData_gyro_z(msg_);
  }

private:
  ::nuturtlebot_msgs::msg::SensorData msg_;
};

class Init_SensorData_gyro_x
{
public:
  explicit Init_SensorData_gyro_x(::nuturtlebot_msgs::msg::SensorData & msg)
  : msg_(msg)
  {}
  Init_SensorData_gyro_y gyro_x(::nuturtlebot_msgs::msg::SensorData::_gyro_x_type arg)
  {
    msg_.gyro_x = std::move(arg);
    return Init_SensorData_gyro_y(msg_);
  }

private:
  ::nuturtlebot_msgs::msg::SensorData msg_;
};

class Init_SensorData_accel_z
{
public:
  explicit Init_SensorData_accel_z(::nuturtlebot_msgs::msg::SensorData & msg)
  : msg_(msg)
  {}
  Init_SensorData_gyro_x accel_z(::nuturtlebot_msgs::msg::SensorData::_accel_z_type arg)
  {
    msg_.accel_z = std::move(arg);
    return Init_SensorData_gyro_x(msg_);
  }

private:
  ::nuturtlebot_msgs::msg::SensorData msg_;
};

class Init_SensorData_accel_y
{
public:
  explicit Init_SensorData_accel_y(::nuturtlebot_msgs::msg::SensorData & msg)
  : msg_(msg)
  {}
  Init_SensorData_accel_z accel_y(::nuturtlebot_msgs::msg::SensorData::_accel_y_type arg)
  {
    msg_.accel_y = std::move(arg);
    return Init_SensorData_accel_z(msg_);
  }

private:
  ::nuturtlebot_msgs::msg::SensorData msg_;
};

class Init_SensorData_accel_x
{
public:
  explicit Init_SensorData_accel_x(::nuturtlebot_msgs::msg::SensorData & msg)
  : msg_(msg)
  {}
  Init_SensorData_accel_y accel_x(::nuturtlebot_msgs::msg::SensorData::_accel_x_type arg)
  {
    msg_.accel_x = std::move(arg);
    return Init_SensorData_accel_y(msg_);
  }

private:
  ::nuturtlebot_msgs::msg::SensorData msg_;
};

class Init_SensorData_right_encoder
{
public:
  explicit Init_SensorData_right_encoder(::nuturtlebot_msgs::msg::SensorData & msg)
  : msg_(msg)
  {}
  Init_SensorData_accel_x right_encoder(::nuturtlebot_msgs::msg::SensorData::_right_encoder_type arg)
  {
    msg_.right_encoder = std::move(arg);
    return Init_SensorData_accel_x(msg_);
  }

private:
  ::nuturtlebot_msgs::msg::SensorData msg_;
};

class Init_SensorData_left_encoder
{
public:
  explicit Init_SensorData_left_encoder(::nuturtlebot_msgs::msg::SensorData & msg)
  : msg_(msg)
  {}
  Init_SensorData_right_encoder left_encoder(::nuturtlebot_msgs::msg::SensorData::_left_encoder_type arg)
  {
    msg_.left_encoder = std::move(arg);
    return Init_SensorData_right_encoder(msg_);
  }

private:
  ::nuturtlebot_msgs::msg::SensorData msg_;
};

class Init_SensorData_stamp
{
public:
  Init_SensorData_stamp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SensorData_left_encoder stamp(::nuturtlebot_msgs::msg::SensorData::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return Init_SensorData_left_encoder(msg_);
  }

private:
  ::nuturtlebot_msgs::msg::SensorData msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::nuturtlebot_msgs::msg::SensorData>()
{
  return nuturtlebot_msgs::msg::builder::Init_SensorData_stamp();
}

}  // namespace nuturtlebot_msgs

#endif  // NUTURTLEBOT_MSGS__MSG__DETAIL__SENSOR_DATA__BUILDER_HPP_
