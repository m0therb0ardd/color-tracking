// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from nuturtlebot_msgs:msg/SensorData.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "nuturtlebot_msgs/msg/sensor_data.hpp"


#ifndef NUTURTLEBOT_MSGS__MSG__DETAIL__SENSOR_DATA__STRUCT_HPP_
#define NUTURTLEBOT_MSGS__MSG__DETAIL__SENSOR_DATA__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__nuturtlebot_msgs__msg__SensorData __attribute__((deprecated))
#else
# define DEPRECATED__nuturtlebot_msgs__msg__SensorData __declspec(deprecated)
#endif

namespace nuturtlebot_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SensorData_
{
  using Type = SensorData_<ContainerAllocator>;

  explicit SensorData_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->left_encoder = 0l;
      this->right_encoder = 0l;
      this->accel_x = 0;
      this->accel_y = 0;
      this->accel_z = 0;
      this->gyro_x = 0;
      this->gyro_y = 0;
      this->gyro_z = 0;
      this->mag_x = 0;
      this->mag_y = 0;
      this->mag_z = 0;
      this->battery_voltage = 0.0f;
    }
  }

  explicit SensorData_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->left_encoder = 0l;
      this->right_encoder = 0l;
      this->accel_x = 0;
      this->accel_y = 0;
      this->accel_z = 0;
      this->gyro_x = 0;
      this->gyro_y = 0;
      this->gyro_z = 0;
      this->mag_x = 0;
      this->mag_y = 0;
      this->mag_z = 0;
      this->battery_voltage = 0.0f;
    }
  }

  // field types and members
  using _stamp_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _stamp_type stamp;
  using _left_encoder_type =
    int32_t;
  _left_encoder_type left_encoder;
  using _right_encoder_type =
    int32_t;
  _right_encoder_type right_encoder;
  using _accel_x_type =
    int16_t;
  _accel_x_type accel_x;
  using _accel_y_type =
    int16_t;
  _accel_y_type accel_y;
  using _accel_z_type =
    int16_t;
  _accel_z_type accel_z;
  using _gyro_x_type =
    int16_t;
  _gyro_x_type gyro_x;
  using _gyro_y_type =
    int16_t;
  _gyro_y_type gyro_y;
  using _gyro_z_type =
    int16_t;
  _gyro_z_type gyro_z;
  using _mag_x_type =
    int16_t;
  _mag_x_type mag_x;
  using _mag_y_type =
    int16_t;
  _mag_y_type mag_y;
  using _mag_z_type =
    int16_t;
  _mag_z_type mag_z;
  using _battery_voltage_type =
    float;
  _battery_voltage_type battery_voltage;

  // setters for named parameter idiom
  Type & set__stamp(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->stamp = _arg;
    return *this;
  }
  Type & set__left_encoder(
    const int32_t & _arg)
  {
    this->left_encoder = _arg;
    return *this;
  }
  Type & set__right_encoder(
    const int32_t & _arg)
  {
    this->right_encoder = _arg;
    return *this;
  }
  Type & set__accel_x(
    const int16_t & _arg)
  {
    this->accel_x = _arg;
    return *this;
  }
  Type & set__accel_y(
    const int16_t & _arg)
  {
    this->accel_y = _arg;
    return *this;
  }
  Type & set__accel_z(
    const int16_t & _arg)
  {
    this->accel_z = _arg;
    return *this;
  }
  Type & set__gyro_x(
    const int16_t & _arg)
  {
    this->gyro_x = _arg;
    return *this;
  }
  Type & set__gyro_y(
    const int16_t & _arg)
  {
    this->gyro_y = _arg;
    return *this;
  }
  Type & set__gyro_z(
    const int16_t & _arg)
  {
    this->gyro_z = _arg;
    return *this;
  }
  Type & set__mag_x(
    const int16_t & _arg)
  {
    this->mag_x = _arg;
    return *this;
  }
  Type & set__mag_y(
    const int16_t & _arg)
  {
    this->mag_y = _arg;
    return *this;
  }
  Type & set__mag_z(
    const int16_t & _arg)
  {
    this->mag_z = _arg;
    return *this;
  }
  Type & set__battery_voltage(
    const float & _arg)
  {
    this->battery_voltage = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    nuturtlebot_msgs::msg::SensorData_<ContainerAllocator> *;
  using ConstRawPtr =
    const nuturtlebot_msgs::msg::SensorData_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<nuturtlebot_msgs::msg::SensorData_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<nuturtlebot_msgs::msg::SensorData_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      nuturtlebot_msgs::msg::SensorData_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<nuturtlebot_msgs::msg::SensorData_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      nuturtlebot_msgs::msg::SensorData_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<nuturtlebot_msgs::msg::SensorData_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<nuturtlebot_msgs::msg::SensorData_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<nuturtlebot_msgs::msg::SensorData_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__nuturtlebot_msgs__msg__SensorData
    std::shared_ptr<nuturtlebot_msgs::msg::SensorData_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__nuturtlebot_msgs__msg__SensorData
    std::shared_ptr<nuturtlebot_msgs::msg::SensorData_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SensorData_ & other) const
  {
    if (this->stamp != other.stamp) {
      return false;
    }
    if (this->left_encoder != other.left_encoder) {
      return false;
    }
    if (this->right_encoder != other.right_encoder) {
      return false;
    }
    if (this->accel_x != other.accel_x) {
      return false;
    }
    if (this->accel_y != other.accel_y) {
      return false;
    }
    if (this->accel_z != other.accel_z) {
      return false;
    }
    if (this->gyro_x != other.gyro_x) {
      return false;
    }
    if (this->gyro_y != other.gyro_y) {
      return false;
    }
    if (this->gyro_z != other.gyro_z) {
      return false;
    }
    if (this->mag_x != other.mag_x) {
      return false;
    }
    if (this->mag_y != other.mag_y) {
      return false;
    }
    if (this->mag_z != other.mag_z) {
      return false;
    }
    if (this->battery_voltage != other.battery_voltage) {
      return false;
    }
    return true;
  }
  bool operator!=(const SensorData_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SensorData_

// alias to use template instance with default allocator
using SensorData =
  nuturtlebot_msgs::msg::SensorData_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace nuturtlebot_msgs

#endif  // NUTURTLEBOT_MSGS__MSG__DETAIL__SENSOR_DATA__STRUCT_HPP_
