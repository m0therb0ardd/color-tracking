// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from nuturtlebot_msgs:msg/SensorData.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "nuturtlebot_msgs/msg/sensor_data.h"


#ifndef NUTURTLEBOT_MSGS__MSG__DETAIL__SENSOR_DATA__STRUCT_H_
#define NUTURTLEBOT_MSGS__MSG__DETAIL__SENSOR_DATA__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in msg/SensorData in the package nuturtlebot_msgs.
/**
  * Time the data was recorded
 */
typedef struct nuturtlebot_msgs__msg__SensorData
{
  builtin_interfaces__msg__Time stamp;
  /// Left encoder data, in ticks
  int32_t left_encoder;
  /// Left encoder data, in ticks
  int32_t right_encoder;
  /// Raw acceleromter x, y, and z data
  int16_t accel_x;
  int16_t accel_y;
  int16_t accel_z;
  /// Raw gyroscope x, y, and z data
  int16_t gyro_x;
  int16_t gyro_y;
  int16_t gyro_z;
  /// raw magnetometer x, y, and z components
  int16_t mag_x;
  int16_t mag_y;
  int16_t mag_z;
  /// battery voltage
  float battery_voltage;
} nuturtlebot_msgs__msg__SensorData;

// Struct for a sequence of nuturtlebot_msgs__msg__SensorData.
typedef struct nuturtlebot_msgs__msg__SensorData__Sequence
{
  nuturtlebot_msgs__msg__SensorData * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} nuturtlebot_msgs__msg__SensorData__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // NUTURTLEBOT_MSGS__MSG__DETAIL__SENSOR_DATA__STRUCT_H_
