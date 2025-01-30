// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from nuturtlebot_msgs:msg/WheelCommands.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "nuturtlebot_msgs/msg/wheel_commands.h"


#ifndef NUTURTLEBOT_MSGS__MSG__DETAIL__WHEEL_COMMANDS__STRUCT_H_
#define NUTURTLEBOT_MSGS__MSG__DETAIL__WHEEL_COMMANDS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

/// Struct defined in msg/WheelCommands in the package nuturtlebot_msgs.
/**
  * Left and right wheel velocity, in "motor command units" (mcu)
  * For the turtlebot, each motor can be command with an integer velocity of between
  * -265 mcu and 265 mcu, and 1 mcu = 0.024 rad/sec
 */
typedef struct nuturtlebot_msgs__msg__WheelCommands
{
  int32_t left_velocity;
  int32_t right_velocity;
} nuturtlebot_msgs__msg__WheelCommands;

// Struct for a sequence of nuturtlebot_msgs__msg__WheelCommands.
typedef struct nuturtlebot_msgs__msg__WheelCommands__Sequence
{
  nuturtlebot_msgs__msg__WheelCommands * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} nuturtlebot_msgs__msg__WheelCommands__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // NUTURTLEBOT_MSGS__MSG__DETAIL__WHEEL_COMMANDS__STRUCT_H_
