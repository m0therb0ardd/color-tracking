// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from nuturtlebot_msgs:msg/WheelCommands.idl
// generated code does not contain a copyright notice
#include "nuturtlebot_msgs/msg/detail/wheel_commands__rosidl_typesupport_fastrtps_cpp.hpp"
#include "nuturtlebot_msgs/msg/detail/wheel_commands__functions.h"
#include "nuturtlebot_msgs/msg/detail/wheel_commands__struct.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/serialization_helpers.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions

namespace nuturtlebot_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{


bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_nuturtlebot_msgs
cdr_serialize(
  const nuturtlebot_msgs::msg::WheelCommands & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: left_velocity
  cdr << ros_message.left_velocity;

  // Member: right_velocity
  cdr << ros_message.right_velocity;

  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_nuturtlebot_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  nuturtlebot_msgs::msg::WheelCommands & ros_message)
{
  // Member: left_velocity
  cdr >> ros_message.left_velocity;

  // Member: right_velocity
  cdr >> ros_message.right_velocity;

  return true;
}


size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_nuturtlebot_msgs
get_serialized_size(
  const nuturtlebot_msgs::msg::WheelCommands & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: left_velocity
  {
    size_t item_size = sizeof(ros_message.left_velocity);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Member: right_velocity
  {
    size_t item_size = sizeof(ros_message.right_velocity);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}


size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_nuturtlebot_msgs
max_serialized_size_WheelCommands(
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

  // Member: left_velocity
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // Member: right_velocity
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
    using DataType = nuturtlebot_msgs::msg::WheelCommands;
    is_plain =
      (
      offsetof(DataType, right_velocity) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_nuturtlebot_msgs
cdr_serialize_key(
  const nuturtlebot_msgs::msg::WheelCommands & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: left_velocity
  cdr << ros_message.left_velocity;

  // Member: right_velocity
  cdr << ros_message.right_velocity;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_nuturtlebot_msgs
get_serialized_size_key(
  const nuturtlebot_msgs::msg::WheelCommands & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: left_velocity
  {
    size_t item_size = sizeof(ros_message.left_velocity);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Member: right_velocity
  {
    size_t item_size = sizeof(ros_message.right_velocity);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_nuturtlebot_msgs
max_serialized_size_key_WheelCommands(
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

  // Member: left_velocity
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: right_velocity
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
    using DataType = nuturtlebot_msgs::msg::WheelCommands;
    is_plain =
      (
      offsetof(DataType, right_velocity) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}


static bool _WheelCommands__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const nuturtlebot_msgs::msg::WheelCommands *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _WheelCommands__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<nuturtlebot_msgs::msg::WheelCommands *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _WheelCommands__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const nuturtlebot_msgs::msg::WheelCommands *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _WheelCommands__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_WheelCommands(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _WheelCommands__callbacks = {
  "nuturtlebot_msgs::msg",
  "WheelCommands",
  _WheelCommands__cdr_serialize,
  _WheelCommands__cdr_deserialize,
  _WheelCommands__get_serialized_size,
  _WheelCommands__max_serialized_size,
  nullptr
};

static rosidl_message_type_support_t _WheelCommands__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_WheelCommands__callbacks,
  get_message_typesupport_handle_function,
  &nuturtlebot_msgs__msg__WheelCommands__get_type_hash,
  &nuturtlebot_msgs__msg__WheelCommands__get_type_description,
  &nuturtlebot_msgs__msg__WheelCommands__get_type_description_sources,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace nuturtlebot_msgs

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_nuturtlebot_msgs
const rosidl_message_type_support_t *
get_message_type_support_handle<nuturtlebot_msgs::msg::WheelCommands>()
{
  return &nuturtlebot_msgs::msg::typesupport_fastrtps_cpp::_WheelCommands__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, nuturtlebot_msgs, msg, WheelCommands)() {
  return &nuturtlebot_msgs::msg::typesupport_fastrtps_cpp::_WheelCommands__handle;
}

#ifdef __cplusplus
}
#endif
