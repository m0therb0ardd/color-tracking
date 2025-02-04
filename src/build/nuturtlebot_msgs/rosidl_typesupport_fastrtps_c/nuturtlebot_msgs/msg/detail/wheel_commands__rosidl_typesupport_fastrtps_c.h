// generated from rosidl_typesupport_fastrtps_c/resource/idl__rosidl_typesupport_fastrtps_c.h.em
// with input from nuturtlebot_msgs:msg/WheelCommands.idl
// generated code does not contain a copyright notice
#ifndef NUTURTLEBOT_MSGS__MSG__DETAIL__WHEEL_COMMANDS__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
#define NUTURTLEBOT_MSGS__MSG__DETAIL__WHEEL_COMMANDS__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_


#include <stddef.h>
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "nuturtlebot_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "nuturtlebot_msgs/msg/detail/wheel_commands__struct.h"
#include "fastcdr/Cdr.h"

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
bool cdr_serialize_nuturtlebot_msgs__msg__WheelCommands(
  const nuturtlebot_msgs__msg__WheelCommands * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
bool cdr_deserialize_nuturtlebot_msgs__msg__WheelCommands(
  eprosima::fastcdr::Cdr &,
  nuturtlebot_msgs__msg__WheelCommands * ros_message);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
size_t get_serialized_size_nuturtlebot_msgs__msg__WheelCommands(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
size_t max_serialized_size_nuturtlebot_msgs__msg__WheelCommands(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
bool cdr_serialize_key_nuturtlebot_msgs__msg__WheelCommands(
  const nuturtlebot_msgs__msg__WheelCommands * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
size_t get_serialized_size_key_nuturtlebot_msgs__msg__WheelCommands(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
size_t max_serialized_size_key_nuturtlebot_msgs__msg__WheelCommands(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, nuturtlebot_msgs, msg, WheelCommands)();

#ifdef __cplusplus
}
#endif

#endif  // NUTURTLEBOT_MSGS__MSG__DETAIL__WHEEL_COMMANDS__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
