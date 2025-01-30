// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from nuturtlebot_msgs:msg/WheelCommands.idl
// generated code does not contain a copyright notice

#include "nuturtlebot_msgs/msg/detail/wheel_commands__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_nuturtlebot_msgs
const rosidl_type_hash_t *
nuturtlebot_msgs__msg__WheelCommands__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x65, 0x6f, 0xa9, 0x54, 0xe4, 0x27, 0x66, 0x3c,
      0x7a, 0x86, 0x20, 0xab, 0x34, 0xba, 0xcb, 0x54,
      0xa3, 0x5f, 0xf2, 0x78, 0x47, 0x0d, 0xdf, 0x2b,
      0x7c, 0xac, 0xfa, 0x77, 0xc2, 0x25, 0xf7, 0x80,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char nuturtlebot_msgs__msg__WheelCommands__TYPE_NAME[] = "nuturtlebot_msgs/msg/WheelCommands";

// Define type names, field names, and default values
static char nuturtlebot_msgs__msg__WheelCommands__FIELD_NAME__left_velocity[] = "left_velocity";
static char nuturtlebot_msgs__msg__WheelCommands__FIELD_NAME__right_velocity[] = "right_velocity";

static rosidl_runtime_c__type_description__Field nuturtlebot_msgs__msg__WheelCommands__FIELDS[] = {
  {
    {nuturtlebot_msgs__msg__WheelCommands__FIELD_NAME__left_velocity, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {nuturtlebot_msgs__msg__WheelCommands__FIELD_NAME__right_velocity, 14, 14},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
nuturtlebot_msgs__msg__WheelCommands__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {nuturtlebot_msgs__msg__WheelCommands__TYPE_NAME, 34, 34},
      {nuturtlebot_msgs__msg__WheelCommands__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "# Left and right wheel velocity, in \"motor command units\" (mcu)\n"
  "# For the turtlebot, each motor can be command with an integer velocity of between\n"
  "# -265 mcu and 265 mcu, and 1 mcu = 0.024 rad/sec\n"
  "int32 left_velocity\n"
  "int32 right_velocity";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
nuturtlebot_msgs__msg__WheelCommands__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {nuturtlebot_msgs__msg__WheelCommands__TYPE_NAME, 34, 34},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 238, 238},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
nuturtlebot_msgs__msg__WheelCommands__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *nuturtlebot_msgs__msg__WheelCommands__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
