// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from nuturtlebot_msgs:msg/WheelCommands.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "nuturtlebot_msgs/msg/wheel_commands.hpp"


#ifndef NUTURTLEBOT_MSGS__MSG__DETAIL__WHEEL_COMMANDS__BUILDER_HPP_
#define NUTURTLEBOT_MSGS__MSG__DETAIL__WHEEL_COMMANDS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "nuturtlebot_msgs/msg/detail/wheel_commands__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace nuturtlebot_msgs
{

namespace msg
{

namespace builder
{

class Init_WheelCommands_right_velocity
{
public:
  explicit Init_WheelCommands_right_velocity(::nuturtlebot_msgs::msg::WheelCommands & msg)
  : msg_(msg)
  {}
  ::nuturtlebot_msgs::msg::WheelCommands right_velocity(::nuturtlebot_msgs::msg::WheelCommands::_right_velocity_type arg)
  {
    msg_.right_velocity = std::move(arg);
    return std::move(msg_);
  }

private:
  ::nuturtlebot_msgs::msg::WheelCommands msg_;
};

class Init_WheelCommands_left_velocity
{
public:
  Init_WheelCommands_left_velocity()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_WheelCommands_right_velocity left_velocity(::nuturtlebot_msgs::msg::WheelCommands::_left_velocity_type arg)
  {
    msg_.left_velocity = std::move(arg);
    return Init_WheelCommands_right_velocity(msg_);
  }

private:
  ::nuturtlebot_msgs::msg::WheelCommands msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::nuturtlebot_msgs::msg::WheelCommands>()
{
  return nuturtlebot_msgs::msg::builder::Init_WheelCommands_left_velocity();
}

}  // namespace nuturtlebot_msgs

#endif  // NUTURTLEBOT_MSGS__MSG__DETAIL__WHEEL_COMMANDS__BUILDER_HPP_
