// generated from rosidl_typesupport_fastrtps_c/resource/idl__rosidl_typesupport_fastrtps_c.h.em
// with input from nuturtlebot_msgs:msg/SensorData.idl
// generated code does not contain a copyright notice
#ifndef NUTURTLEBOT_MSGS__MSG__DETAIL__SENSOR_DATA__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
#define NUTURTLEBOT_MSGS__MSG__DETAIL__SENSOR_DATA__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_


#include <stddef.h>
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "nuturtlebot_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "nuturtlebot_msgs/msg/detail/sensor_data__struct.h"
#include "fastcdr/Cdr.h"

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
bool cdr_serialize_nuturtlebot_msgs__msg__SensorData(
  const nuturtlebot_msgs__msg__SensorData * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
bool cdr_deserialize_nuturtlebot_msgs__msg__SensorData(
  eprosima::fastcdr::Cdr &,
  nuturtlebot_msgs__msg__SensorData * ros_message);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
size_t get_serialized_size_nuturtlebot_msgs__msg__SensorData(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
size_t max_serialized_size_nuturtlebot_msgs__msg__SensorData(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
bool cdr_serialize_key_nuturtlebot_msgs__msg__SensorData(
  const nuturtlebot_msgs__msg__SensorData * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
size_t get_serialized_size_key_nuturtlebot_msgs__msg__SensorData(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
size_t max_serialized_size_key_nuturtlebot_msgs__msg__SensorData(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, nuturtlebot_msgs, msg, SensorData)();

#ifdef __cplusplus
}
#endif

#endif  // NUTURTLEBOT_MSGS__MSG__DETAIL__SENSOR_DATA__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
