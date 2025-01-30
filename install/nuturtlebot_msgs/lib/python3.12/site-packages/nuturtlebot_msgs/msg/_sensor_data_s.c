// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from nuturtlebot_msgs:msg/SensorData.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "nuturtlebot_msgs/msg/detail/sensor_data__struct.h"
#include "nuturtlebot_msgs/msg/detail/sensor_data__functions.h"

ROSIDL_GENERATOR_C_IMPORT
bool builtin_interfaces__msg__time__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * builtin_interfaces__msg__time__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool nuturtlebot_msgs__msg__sensor_data__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[45];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("nuturtlebot_msgs.msg._sensor_data.SensorData", full_classname_dest, 44) == 0);
  }
  nuturtlebot_msgs__msg__SensorData * ros_message = _ros_message;
  {  // stamp
    PyObject * field = PyObject_GetAttrString(_pymsg, "stamp");
    if (!field) {
      return false;
    }
    if (!builtin_interfaces__msg__time__convert_from_py(field, &ros_message->stamp)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // left_encoder
    PyObject * field = PyObject_GetAttrString(_pymsg, "left_encoder");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->left_encoder = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // right_encoder
    PyObject * field = PyObject_GetAttrString(_pymsg, "right_encoder");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->right_encoder = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // accel_x
    PyObject * field = PyObject_GetAttrString(_pymsg, "accel_x");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->accel_x = (int16_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // accel_y
    PyObject * field = PyObject_GetAttrString(_pymsg, "accel_y");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->accel_y = (int16_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // accel_z
    PyObject * field = PyObject_GetAttrString(_pymsg, "accel_z");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->accel_z = (int16_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // gyro_x
    PyObject * field = PyObject_GetAttrString(_pymsg, "gyro_x");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->gyro_x = (int16_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // gyro_y
    PyObject * field = PyObject_GetAttrString(_pymsg, "gyro_y");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->gyro_y = (int16_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // gyro_z
    PyObject * field = PyObject_GetAttrString(_pymsg, "gyro_z");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->gyro_z = (int16_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // mag_x
    PyObject * field = PyObject_GetAttrString(_pymsg, "mag_x");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->mag_x = (int16_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // mag_y
    PyObject * field = PyObject_GetAttrString(_pymsg, "mag_y");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->mag_y = (int16_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // mag_z
    PyObject * field = PyObject_GetAttrString(_pymsg, "mag_z");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->mag_z = (int16_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // battery_voltage
    PyObject * field = PyObject_GetAttrString(_pymsg, "battery_voltage");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->battery_voltage = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * nuturtlebot_msgs__msg__sensor_data__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of SensorData */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("nuturtlebot_msgs.msg._sensor_data");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "SensorData");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  nuturtlebot_msgs__msg__SensorData * ros_message = (nuturtlebot_msgs__msg__SensorData *)raw_ros_message;
  {  // stamp
    PyObject * field = NULL;
    field = builtin_interfaces__msg__time__convert_to_py(&ros_message->stamp);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "stamp", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // left_encoder
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->left_encoder);
    {
      int rc = PyObject_SetAttrString(_pymessage, "left_encoder", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // right_encoder
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->right_encoder);
    {
      int rc = PyObject_SetAttrString(_pymessage, "right_encoder", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // accel_x
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->accel_x);
    {
      int rc = PyObject_SetAttrString(_pymessage, "accel_x", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // accel_y
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->accel_y);
    {
      int rc = PyObject_SetAttrString(_pymessage, "accel_y", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // accel_z
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->accel_z);
    {
      int rc = PyObject_SetAttrString(_pymessage, "accel_z", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // gyro_x
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->gyro_x);
    {
      int rc = PyObject_SetAttrString(_pymessage, "gyro_x", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // gyro_y
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->gyro_y);
    {
      int rc = PyObject_SetAttrString(_pymessage, "gyro_y", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // gyro_z
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->gyro_z);
    {
      int rc = PyObject_SetAttrString(_pymessage, "gyro_z", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // mag_x
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->mag_x);
    {
      int rc = PyObject_SetAttrString(_pymessage, "mag_x", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // mag_y
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->mag_y);
    {
      int rc = PyObject_SetAttrString(_pymessage, "mag_y", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // mag_z
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->mag_z);
    {
      int rc = PyObject_SetAttrString(_pymessage, "mag_z", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // battery_voltage
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->battery_voltage);
    {
      int rc = PyObject_SetAttrString(_pymessage, "battery_voltage", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
