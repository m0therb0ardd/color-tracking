// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from nuturtlebot_msgs:msg/WheelCommands.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "nuturtlebot_msgs/msg/detail/wheel_commands__functions.h"
#include "nuturtlebot_msgs/msg/detail/wheel_commands__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace nuturtlebot_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void WheelCommands_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) nuturtlebot_msgs::msg::WheelCommands(_init);
}

void WheelCommands_fini_function(void * message_memory)
{
  auto typed_message = static_cast<nuturtlebot_msgs::msg::WheelCommands *>(message_memory);
  typed_message->~WheelCommands();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember WheelCommands_message_member_array[2] = {
  {
    "left_velocity",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(nuturtlebot_msgs::msg::WheelCommands, left_velocity),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "right_velocity",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(nuturtlebot_msgs::msg::WheelCommands, right_velocity),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers WheelCommands_message_members = {
  "nuturtlebot_msgs::msg",  // message namespace
  "WheelCommands",  // message name
  2,  // number of fields
  sizeof(nuturtlebot_msgs::msg::WheelCommands),
  false,  // has_any_key_member_
  WheelCommands_message_member_array,  // message members
  WheelCommands_init_function,  // function to initialize message memory (memory has to be allocated)
  WheelCommands_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t WheelCommands_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &WheelCommands_message_members,
  get_message_typesupport_handle_function,
  &nuturtlebot_msgs__msg__WheelCommands__get_type_hash,
  &nuturtlebot_msgs__msg__WheelCommands__get_type_description,
  &nuturtlebot_msgs__msg__WheelCommands__get_type_description_sources,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace nuturtlebot_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<nuturtlebot_msgs::msg::WheelCommands>()
{
  return &::nuturtlebot_msgs::msg::rosidl_typesupport_introspection_cpp::WheelCommands_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, nuturtlebot_msgs, msg, WheelCommands)() {
  return &::nuturtlebot_msgs::msg::rosidl_typesupport_introspection_cpp::WheelCommands_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
