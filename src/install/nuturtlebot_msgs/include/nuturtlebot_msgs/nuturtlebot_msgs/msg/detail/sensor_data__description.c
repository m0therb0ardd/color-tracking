// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from nuturtlebot_msgs:msg/SensorData.idl
// generated code does not contain a copyright notice

#include "nuturtlebot_msgs/msg/detail/sensor_data__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_nuturtlebot_msgs
const rosidl_type_hash_t *
nuturtlebot_msgs__msg__SensorData__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x3f, 0x6c, 0x41, 0x0e, 0x30, 0x29, 0x1e, 0x7b,
      0x96, 0xc2, 0x38, 0xf1, 0x00, 0xec, 0xd8, 0xc5,
      0x14, 0x0b, 0x81, 0xcc, 0x12, 0x79, 0x8d, 0x37,
      0xf2, 0x6e, 0xdb, 0x95, 0xdf, 0xa4, 0x9b, 0xbf,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "builtin_interfaces/msg/detail/time__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t builtin_interfaces__msg__Time__EXPECTED_HASH = {1, {
    0xb1, 0x06, 0x23, 0x5e, 0x25, 0xa4, 0xc5, 0xed,
    0x35, 0x09, 0x8a, 0xa0, 0xa6, 0x1a, 0x3e, 0xe9,
    0xc9, 0xb1, 0x8d, 0x19, 0x7f, 0x39, 0x8b, 0x0e,
    0x42, 0x06, 0xce, 0xa9, 0xac, 0xf9, 0xc1, 0x97,
  }};
#endif

static char nuturtlebot_msgs__msg__SensorData__TYPE_NAME[] = "nuturtlebot_msgs/msg/SensorData";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";

// Define type names, field names, and default values
static char nuturtlebot_msgs__msg__SensorData__FIELD_NAME__stamp[] = "stamp";
static char nuturtlebot_msgs__msg__SensorData__FIELD_NAME__left_encoder[] = "left_encoder";
static char nuturtlebot_msgs__msg__SensorData__FIELD_NAME__right_encoder[] = "right_encoder";
static char nuturtlebot_msgs__msg__SensorData__FIELD_NAME__accel_x[] = "accel_x";
static char nuturtlebot_msgs__msg__SensorData__FIELD_NAME__accel_y[] = "accel_y";
static char nuturtlebot_msgs__msg__SensorData__FIELD_NAME__accel_z[] = "accel_z";
static char nuturtlebot_msgs__msg__SensorData__FIELD_NAME__gyro_x[] = "gyro_x";
static char nuturtlebot_msgs__msg__SensorData__FIELD_NAME__gyro_y[] = "gyro_y";
static char nuturtlebot_msgs__msg__SensorData__FIELD_NAME__gyro_z[] = "gyro_z";
static char nuturtlebot_msgs__msg__SensorData__FIELD_NAME__mag_x[] = "mag_x";
static char nuturtlebot_msgs__msg__SensorData__FIELD_NAME__mag_y[] = "mag_y";
static char nuturtlebot_msgs__msg__SensorData__FIELD_NAME__mag_z[] = "mag_z";
static char nuturtlebot_msgs__msg__SensorData__FIELD_NAME__battery_voltage[] = "battery_voltage";

static rosidl_runtime_c__type_description__Field nuturtlebot_msgs__msg__SensorData__FIELDS[] = {
  {
    {nuturtlebot_msgs__msg__SensorData__FIELD_NAME__stamp, 5, 5},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    },
    {NULL, 0, 0},
  },
  {
    {nuturtlebot_msgs__msg__SensorData__FIELD_NAME__left_encoder, 12, 12},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {nuturtlebot_msgs__msg__SensorData__FIELD_NAME__right_encoder, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {nuturtlebot_msgs__msg__SensorData__FIELD_NAME__accel_x, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT16,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {nuturtlebot_msgs__msg__SensorData__FIELD_NAME__accel_y, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT16,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {nuturtlebot_msgs__msg__SensorData__FIELD_NAME__accel_z, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT16,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {nuturtlebot_msgs__msg__SensorData__FIELD_NAME__gyro_x, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT16,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {nuturtlebot_msgs__msg__SensorData__FIELD_NAME__gyro_y, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT16,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {nuturtlebot_msgs__msg__SensorData__FIELD_NAME__gyro_z, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT16,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {nuturtlebot_msgs__msg__SensorData__FIELD_NAME__mag_x, 5, 5},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT16,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {nuturtlebot_msgs__msg__SensorData__FIELD_NAME__mag_y, 5, 5},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT16,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {nuturtlebot_msgs__msg__SensorData__FIELD_NAME__mag_z, 5, 5},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT16,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {nuturtlebot_msgs__msg__SensorData__FIELD_NAME__battery_voltage, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription nuturtlebot_msgs__msg__SensorData__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
nuturtlebot_msgs__msg__SensorData__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {nuturtlebot_msgs__msg__SensorData__TYPE_NAME, 31, 31},
      {nuturtlebot_msgs__msg__SensorData__FIELDS, 13, 13},
    },
    {nuturtlebot_msgs__msg__SensorData__REFERENCED_TYPE_DESCRIPTIONS, 1, 1},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "# Time the data was recorded\n"
  "builtin_interfaces/Time stamp\n"
  "# Left encoder data, in ticks\n"
  "int32 left_encoder\n"
  "# Left encoder data, in ticks\n"
  "int32 right_encoder\n"
  "\n"
  "# Raw acceleromter x, y, and z data\n"
  "int16 accel_x\n"
  "int16 accel_y\n"
  "int16 accel_z\n"
  "\n"
  "# Raw gyroscope x, y, and z data\n"
  "int16 gyro_x\n"
  "int16 gyro_y\n"
  "int16 gyro_z\n"
  "\n"
  "# raw magnetometer x, y, and z components\n"
  "int16 mag_x\n"
  "int16 mag_y\n"
  "int16 mag_z\n"
  "\n"
  "#battery voltage\n"
  "float32 battery_voltage";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
nuturtlebot_msgs__msg__SensorData__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {nuturtlebot_msgs__msg__SensorData__TYPE_NAME, 31, 31},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 431, 431},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
nuturtlebot_msgs__msg__SensorData__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[2];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 2, 2};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *nuturtlebot_msgs__msg__SensorData__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
