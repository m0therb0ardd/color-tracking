// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from nuturtlebot_msgs:msg/WheelCommands.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "nuturtlebot_msgs/msg/wheel_commands.hpp"


#ifndef NUTURTLEBOT_MSGS__MSG__DETAIL__WHEEL_COMMANDS__STRUCT_HPP_
#define NUTURTLEBOT_MSGS__MSG__DETAIL__WHEEL_COMMANDS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__nuturtlebot_msgs__msg__WheelCommands __attribute__((deprecated))
#else
# define DEPRECATED__nuturtlebot_msgs__msg__WheelCommands __declspec(deprecated)
#endif

namespace nuturtlebot_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct WheelCommands_
{
  using Type = WheelCommands_<ContainerAllocator>;

  explicit WheelCommands_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->left_velocity = 0l;
      this->right_velocity = 0l;
    }
  }

  explicit WheelCommands_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->left_velocity = 0l;
      this->right_velocity = 0l;
    }
  }

  // field types and members
  using _left_velocity_type =
    int32_t;
  _left_velocity_type left_velocity;
  using _right_velocity_type =
    int32_t;
  _right_velocity_type right_velocity;

  // setters for named parameter idiom
  Type & set__left_velocity(
    const int32_t & _arg)
  {
    this->left_velocity = _arg;
    return *this;
  }
  Type & set__right_velocity(
    const int32_t & _arg)
  {
    this->right_velocity = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    nuturtlebot_msgs::msg::WheelCommands_<ContainerAllocator> *;
  using ConstRawPtr =
    const nuturtlebot_msgs::msg::WheelCommands_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<nuturtlebot_msgs::msg::WheelCommands_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<nuturtlebot_msgs::msg::WheelCommands_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      nuturtlebot_msgs::msg::WheelCommands_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<nuturtlebot_msgs::msg::WheelCommands_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      nuturtlebot_msgs::msg::WheelCommands_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<nuturtlebot_msgs::msg::WheelCommands_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<nuturtlebot_msgs::msg::WheelCommands_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<nuturtlebot_msgs::msg::WheelCommands_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__nuturtlebot_msgs__msg__WheelCommands
    std::shared_ptr<nuturtlebot_msgs::msg::WheelCommands_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__nuturtlebot_msgs__msg__WheelCommands
    std::shared_ptr<nuturtlebot_msgs::msg::WheelCommands_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const WheelCommands_ & other) const
  {
    if (this->left_velocity != other.left_velocity) {
      return false;
    }
    if (this->right_velocity != other.right_velocity) {
      return false;
    }
    return true;
  }
  bool operator!=(const WheelCommands_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct WheelCommands_

// alias to use template instance with default allocator
using WheelCommands =
  nuturtlebot_msgs::msg::WheelCommands_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace nuturtlebot_msgs

#endif  // NUTURTLEBOT_MSGS__MSG__DETAIL__WHEEL_COMMANDS__STRUCT_HPP_
