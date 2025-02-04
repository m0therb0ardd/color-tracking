// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from nuturtlebot_msgs:msg/WheelCommands.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "nuturtlebot_msgs/msg/wheel_commands.hpp"


#ifndef NUTURTLEBOT_MSGS__MSG__DETAIL__WHEEL_COMMANDS__TRAITS_HPP_
#define NUTURTLEBOT_MSGS__MSG__DETAIL__WHEEL_COMMANDS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "nuturtlebot_msgs/msg/detail/wheel_commands__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace nuturtlebot_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const WheelCommands & msg,
  std::ostream & out)
{
  out << "{";
  // member: left_velocity
  {
    out << "left_velocity: ";
    rosidl_generator_traits::value_to_yaml(msg.left_velocity, out);
    out << ", ";
  }

  // member: right_velocity
  {
    out << "right_velocity: ";
    rosidl_generator_traits::value_to_yaml(msg.right_velocity, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const WheelCommands & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: left_velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "left_velocity: ";
    rosidl_generator_traits::value_to_yaml(msg.left_velocity, out);
    out << "\n";
  }

  // member: right_velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "right_velocity: ";
    rosidl_generator_traits::value_to_yaml(msg.right_velocity, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const WheelCommands & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace nuturtlebot_msgs

namespace rosidl_generator_traits
{

[[deprecated("use nuturtlebot_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const nuturtlebot_msgs::msg::WheelCommands & msg,
  std::ostream & out, size_t indentation = 0)
{
  nuturtlebot_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use nuturtlebot_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const nuturtlebot_msgs::msg::WheelCommands & msg)
{
  return nuturtlebot_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<nuturtlebot_msgs::msg::WheelCommands>()
{
  return "nuturtlebot_msgs::msg::WheelCommands";
}

template<>
inline const char * name<nuturtlebot_msgs::msg::WheelCommands>()
{
  return "nuturtlebot_msgs/msg/WheelCommands";
}

template<>
struct has_fixed_size<nuturtlebot_msgs::msg::WheelCommands>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<nuturtlebot_msgs::msg::WheelCommands>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<nuturtlebot_msgs::msg::WheelCommands>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // NUTURTLEBOT_MSGS__MSG__DETAIL__WHEEL_COMMANDS__TRAITS_HPP_
