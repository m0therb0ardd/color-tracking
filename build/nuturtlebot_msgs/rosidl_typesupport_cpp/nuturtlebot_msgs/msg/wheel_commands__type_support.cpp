// generated from rosidl_typesupport_cpp/resource/idl__type_support.cpp.em
// with input from nuturtlebot_msgs:msg/WheelCommands.idl
// generated code does not contain a copyright notice

#include "cstddef"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "nuturtlebot_msgs/msg/detail/wheel_commands__functions.h"
#include "nuturtlebot_msgs/msg/detail/wheel_commands__struct.hpp"
#include "rosidl_typesupport_cpp/identifier.hpp"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_cpp/message_type_support_dispatch.hpp"
#include "rosidl_typesupport_cpp/visibility_control.h"
#include "rosidl_typesupport_interface/macros.h"

namespace nuturtlebot_msgs
{

namespace msg
{

namespace rosidl_typesupport_cpp
{

typedef struct _WheelCommands_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _WheelCommands_type_support_ids_t;

static const _WheelCommands_type_support_ids_t _WheelCommands_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  }
};

typedef struct _WheelCommands_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _WheelCommands_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _WheelCommands_type_support_symbol_names_t _WheelCommands_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, nuturtlebot_msgs, msg, WheelCommands)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, nuturtlebot_msgs, msg, WheelCommands)),
  }
};

typedef struct _WheelCommands_type_support_data_t
{
  void * data[2];
} _WheelCommands_type_support_data_t;

static _WheelCommands_type_support_data_t _WheelCommands_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _WheelCommands_message_typesupport_map = {
  2,
  "nuturtlebot_msgs",
  &_WheelCommands_message_typesupport_ids.typesupport_identifier[0],
  &_WheelCommands_message_typesupport_symbol_names.symbol_name[0],
  &_WheelCommands_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t WheelCommands_message_type_support_handle = {
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_WheelCommands_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
  &nuturtlebot_msgs__msg__WheelCommands__get_type_hash,
  &nuturtlebot_msgs__msg__WheelCommands__get_type_description,
  &nuturtlebot_msgs__msg__WheelCommands__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace msg

}  // namespace nuturtlebot_msgs

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<nuturtlebot_msgs::msg::WheelCommands>()
{
  return &::nuturtlebot_msgs::msg::rosidl_typesupport_cpp::WheelCommands_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, nuturtlebot_msgs, msg, WheelCommands)() {
  return get_message_type_support_handle<nuturtlebot_msgs::msg::WheelCommands>();
}

#ifdef __cplusplus
}
#endif
}  // namespace rosidl_typesupport_cpp
