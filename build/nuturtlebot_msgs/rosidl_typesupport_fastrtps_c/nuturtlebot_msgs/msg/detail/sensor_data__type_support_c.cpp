// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from nuturtlebot_msgs:msg/SensorData.idl
// generated code does not contain a copyright notice
#include "nuturtlebot_msgs/msg/detail/sensor_data__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <cstddef>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/serialization_helpers.hpp"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "nuturtlebot_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "nuturtlebot_msgs/msg/detail/sensor_data__struct.h"
#include "nuturtlebot_msgs/msg/detail/sensor_data__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif

#include "builtin_interfaces/msg/detail/time__functions.h"  // stamp

// forward declare type support functions

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_nuturtlebot_msgs
bool cdr_serialize_builtin_interfaces__msg__Time(
  const builtin_interfaces__msg__Time * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_nuturtlebot_msgs
bool cdr_deserialize_builtin_interfaces__msg__Time(
  eprosima::fastcdr::Cdr & cdr,
  builtin_interfaces__msg__Time * ros_message);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_nuturtlebot_msgs
size_t get_serialized_size_builtin_interfaces__msg__Time(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_nuturtlebot_msgs
size_t max_serialized_size_builtin_interfaces__msg__Time(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_nuturtlebot_msgs
bool cdr_serialize_key_builtin_interfaces__msg__Time(
  const builtin_interfaces__msg__Time * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_nuturtlebot_msgs
size_t get_serialized_size_key_builtin_interfaces__msg__Time(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_nuturtlebot_msgs
size_t max_serialized_size_key_builtin_interfaces__msg__Time(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_nuturtlebot_msgs
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, builtin_interfaces, msg, Time)();


using _SensorData__ros_msg_type = nuturtlebot_msgs__msg__SensorData;


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
bool cdr_serialize_nuturtlebot_msgs__msg__SensorData(
  const nuturtlebot_msgs__msg__SensorData * ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Field name: stamp
  {
    cdr_serialize_builtin_interfaces__msg__Time(
      &ros_message->stamp, cdr);
  }

  // Field name: left_encoder
  {
    cdr << ros_message->left_encoder;
  }

  // Field name: right_encoder
  {
    cdr << ros_message->right_encoder;
  }

  // Field name: accel_x
  {
    cdr << ros_message->accel_x;
  }

  // Field name: accel_y
  {
    cdr << ros_message->accel_y;
  }

  // Field name: accel_z
  {
    cdr << ros_message->accel_z;
  }

  // Field name: gyro_x
  {
    cdr << ros_message->gyro_x;
  }

  // Field name: gyro_y
  {
    cdr << ros_message->gyro_y;
  }

  // Field name: gyro_z
  {
    cdr << ros_message->gyro_z;
  }

  // Field name: mag_x
  {
    cdr << ros_message->mag_x;
  }

  // Field name: mag_y
  {
    cdr << ros_message->mag_y;
  }

  // Field name: mag_z
  {
    cdr << ros_message->mag_z;
  }

  // Field name: battery_voltage
  {
    cdr << ros_message->battery_voltage;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
bool cdr_deserialize_nuturtlebot_msgs__msg__SensorData(
  eprosima::fastcdr::Cdr & cdr,
  nuturtlebot_msgs__msg__SensorData * ros_message)
{
  // Field name: stamp
  {
    cdr_deserialize_builtin_interfaces__msg__Time(cdr, &ros_message->stamp);
  }

  // Field name: left_encoder
  {
    cdr >> ros_message->left_encoder;
  }

  // Field name: right_encoder
  {
    cdr >> ros_message->right_encoder;
  }

  // Field name: accel_x
  {
    cdr >> ros_message->accel_x;
  }

  // Field name: accel_y
  {
    cdr >> ros_message->accel_y;
  }

  // Field name: accel_z
  {
    cdr >> ros_message->accel_z;
  }

  // Field name: gyro_x
  {
    cdr >> ros_message->gyro_x;
  }

  // Field name: gyro_y
  {
    cdr >> ros_message->gyro_y;
  }

  // Field name: gyro_z
  {
    cdr >> ros_message->gyro_z;
  }

  // Field name: mag_x
  {
    cdr >> ros_message->mag_x;
  }

  // Field name: mag_y
  {
    cdr >> ros_message->mag_y;
  }

  // Field name: mag_z
  {
    cdr >> ros_message->mag_z;
  }

  // Field name: battery_voltage
  {
    cdr >> ros_message->battery_voltage;
  }

  return true;
}  // NOLINT(readability/fn_size)


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
size_t get_serialized_size_nuturtlebot_msgs__msg__SensorData(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _SensorData__ros_msg_type * ros_message = static_cast<const _SensorData__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Field name: stamp
  current_alignment += get_serialized_size_builtin_interfaces__msg__Time(
    &(ros_message->stamp), current_alignment);

  // Field name: left_encoder
  {
    size_t item_size = sizeof(ros_message->left_encoder);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: right_encoder
  {
    size_t item_size = sizeof(ros_message->right_encoder);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: accel_x
  {
    size_t item_size = sizeof(ros_message->accel_x);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: accel_y
  {
    size_t item_size = sizeof(ros_message->accel_y);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: accel_z
  {
    size_t item_size = sizeof(ros_message->accel_z);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: gyro_x
  {
    size_t item_size = sizeof(ros_message->gyro_x);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: gyro_y
  {
    size_t item_size = sizeof(ros_message->gyro_y);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: gyro_z
  {
    size_t item_size = sizeof(ros_message->gyro_z);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: mag_x
  {
    size_t item_size = sizeof(ros_message->mag_x);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: mag_y
  {
    size_t item_size = sizeof(ros_message->mag_y);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: mag_z
  {
    size_t item_size = sizeof(ros_message->mag_z);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: battery_voltage
  {
    size_t item_size = sizeof(ros_message->battery_voltage);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
size_t max_serialized_size_nuturtlebot_msgs__msg__SensorData(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // Field name: stamp
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_builtin_interfaces__msg__Time(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Field name: left_encoder
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: right_encoder
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: accel_x
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Field name: accel_y
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Field name: accel_z
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Field name: gyro_x
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Field name: gyro_y
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Field name: gyro_z
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Field name: mag_x
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Field name: mag_y
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Field name: mag_z
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Field name: battery_voltage
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }


  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = nuturtlebot_msgs__msg__SensorData;
    is_plain =
      (
      offsetof(DataType, battery_voltage) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
bool cdr_serialize_key_nuturtlebot_msgs__msg__SensorData(
  const nuturtlebot_msgs__msg__SensorData * ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Field name: stamp
  {
    cdr_serialize_key_builtin_interfaces__msg__Time(
      &ros_message->stamp, cdr);
  }

  // Field name: left_encoder
  {
    cdr << ros_message->left_encoder;
  }

  // Field name: right_encoder
  {
    cdr << ros_message->right_encoder;
  }

  // Field name: accel_x
  {
    cdr << ros_message->accel_x;
  }

  // Field name: accel_y
  {
    cdr << ros_message->accel_y;
  }

  // Field name: accel_z
  {
    cdr << ros_message->accel_z;
  }

  // Field name: gyro_x
  {
    cdr << ros_message->gyro_x;
  }

  // Field name: gyro_y
  {
    cdr << ros_message->gyro_y;
  }

  // Field name: gyro_z
  {
    cdr << ros_message->gyro_z;
  }

  // Field name: mag_x
  {
    cdr << ros_message->mag_x;
  }

  // Field name: mag_y
  {
    cdr << ros_message->mag_y;
  }

  // Field name: mag_z
  {
    cdr << ros_message->mag_z;
  }

  // Field name: battery_voltage
  {
    cdr << ros_message->battery_voltage;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
size_t get_serialized_size_key_nuturtlebot_msgs__msg__SensorData(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _SensorData__ros_msg_type * ros_message = static_cast<const _SensorData__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;

  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Field name: stamp
  current_alignment += get_serialized_size_key_builtin_interfaces__msg__Time(
    &(ros_message->stamp), current_alignment);

  // Field name: left_encoder
  {
    size_t item_size = sizeof(ros_message->left_encoder);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: right_encoder
  {
    size_t item_size = sizeof(ros_message->right_encoder);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: accel_x
  {
    size_t item_size = sizeof(ros_message->accel_x);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: accel_y
  {
    size_t item_size = sizeof(ros_message->accel_y);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: accel_z
  {
    size_t item_size = sizeof(ros_message->accel_z);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: gyro_x
  {
    size_t item_size = sizeof(ros_message->gyro_x);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: gyro_y
  {
    size_t item_size = sizeof(ros_message->gyro_y);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: gyro_z
  {
    size_t item_size = sizeof(ros_message->gyro_z);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: mag_x
  {
    size_t item_size = sizeof(ros_message->mag_x);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: mag_y
  {
    size_t item_size = sizeof(ros_message->mag_y);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: mag_z
  {
    size_t item_size = sizeof(ros_message->mag_z);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: battery_voltage
  {
    size_t item_size = sizeof(ros_message->battery_voltage);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_nuturtlebot_msgs
size_t max_serialized_size_key_nuturtlebot_msgs__msg__SensorData(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;
  // Field name: stamp
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_key_builtin_interfaces__msg__Time(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Field name: left_encoder
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: right_encoder
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: accel_x
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Field name: accel_y
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Field name: accel_z
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Field name: gyro_x
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Field name: gyro_y
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Field name: gyro_z
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Field name: mag_x
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Field name: mag_y
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Field name: mag_z
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Field name: battery_voltage
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = nuturtlebot_msgs__msg__SensorData;
    is_plain =
      (
      offsetof(DataType, battery_voltage) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}


static bool _SensorData__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const nuturtlebot_msgs__msg__SensorData * ros_message = static_cast<const nuturtlebot_msgs__msg__SensorData *>(untyped_ros_message);
  (void)ros_message;
  return cdr_serialize_nuturtlebot_msgs__msg__SensorData(ros_message, cdr);
}

static bool _SensorData__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  nuturtlebot_msgs__msg__SensorData * ros_message = static_cast<nuturtlebot_msgs__msg__SensorData *>(untyped_ros_message);
  (void)ros_message;
  return cdr_deserialize_nuturtlebot_msgs__msg__SensorData(cdr, ros_message);
}

static uint32_t _SensorData__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_nuturtlebot_msgs__msg__SensorData(
      untyped_ros_message, 0));
}

static size_t _SensorData__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_nuturtlebot_msgs__msg__SensorData(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_SensorData = {
  "nuturtlebot_msgs::msg",
  "SensorData",
  _SensorData__cdr_serialize,
  _SensorData__cdr_deserialize,
  _SensorData__get_serialized_size,
  _SensorData__max_serialized_size,
  nullptr
};

static rosidl_message_type_support_t _SensorData__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_SensorData,
  get_message_typesupport_handle_function,
  &nuturtlebot_msgs__msg__SensorData__get_type_hash,
  &nuturtlebot_msgs__msg__SensorData__get_type_description,
  &nuturtlebot_msgs__msg__SensorData__get_type_description_sources,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, nuturtlebot_msgs, msg, SensorData)() {
  return &_SensorData__type_support;
}

#if defined(__cplusplus)
}
#endif
