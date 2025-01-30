// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from nuturtlebot_msgs:msg/WheelCommands.idl
// generated code does not contain a copyright notice
#include "nuturtlebot_msgs/msg/detail/wheel_commands__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
nuturtlebot_msgs__msg__WheelCommands__init(nuturtlebot_msgs__msg__WheelCommands * msg)
{
  if (!msg) {
    return false;
  }
  // left_velocity
  // right_velocity
  return true;
}

void
nuturtlebot_msgs__msg__WheelCommands__fini(nuturtlebot_msgs__msg__WheelCommands * msg)
{
  if (!msg) {
    return;
  }
  // left_velocity
  // right_velocity
}

bool
nuturtlebot_msgs__msg__WheelCommands__are_equal(const nuturtlebot_msgs__msg__WheelCommands * lhs, const nuturtlebot_msgs__msg__WheelCommands * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // left_velocity
  if (lhs->left_velocity != rhs->left_velocity) {
    return false;
  }
  // right_velocity
  if (lhs->right_velocity != rhs->right_velocity) {
    return false;
  }
  return true;
}

bool
nuturtlebot_msgs__msg__WheelCommands__copy(
  const nuturtlebot_msgs__msg__WheelCommands * input,
  nuturtlebot_msgs__msg__WheelCommands * output)
{
  if (!input || !output) {
    return false;
  }
  // left_velocity
  output->left_velocity = input->left_velocity;
  // right_velocity
  output->right_velocity = input->right_velocity;
  return true;
}

nuturtlebot_msgs__msg__WheelCommands *
nuturtlebot_msgs__msg__WheelCommands__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  nuturtlebot_msgs__msg__WheelCommands * msg = (nuturtlebot_msgs__msg__WheelCommands *)allocator.allocate(sizeof(nuturtlebot_msgs__msg__WheelCommands), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(nuturtlebot_msgs__msg__WheelCommands));
  bool success = nuturtlebot_msgs__msg__WheelCommands__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
nuturtlebot_msgs__msg__WheelCommands__destroy(nuturtlebot_msgs__msg__WheelCommands * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    nuturtlebot_msgs__msg__WheelCommands__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
nuturtlebot_msgs__msg__WheelCommands__Sequence__init(nuturtlebot_msgs__msg__WheelCommands__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  nuturtlebot_msgs__msg__WheelCommands * data = NULL;

  if (size) {
    data = (nuturtlebot_msgs__msg__WheelCommands *)allocator.zero_allocate(size, sizeof(nuturtlebot_msgs__msg__WheelCommands), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = nuturtlebot_msgs__msg__WheelCommands__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        nuturtlebot_msgs__msg__WheelCommands__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
nuturtlebot_msgs__msg__WheelCommands__Sequence__fini(nuturtlebot_msgs__msg__WheelCommands__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      nuturtlebot_msgs__msg__WheelCommands__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

nuturtlebot_msgs__msg__WheelCommands__Sequence *
nuturtlebot_msgs__msg__WheelCommands__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  nuturtlebot_msgs__msg__WheelCommands__Sequence * array = (nuturtlebot_msgs__msg__WheelCommands__Sequence *)allocator.allocate(sizeof(nuturtlebot_msgs__msg__WheelCommands__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = nuturtlebot_msgs__msg__WheelCommands__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
nuturtlebot_msgs__msg__WheelCommands__Sequence__destroy(nuturtlebot_msgs__msg__WheelCommands__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    nuturtlebot_msgs__msg__WheelCommands__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
nuturtlebot_msgs__msg__WheelCommands__Sequence__are_equal(const nuturtlebot_msgs__msg__WheelCommands__Sequence * lhs, const nuturtlebot_msgs__msg__WheelCommands__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!nuturtlebot_msgs__msg__WheelCommands__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
nuturtlebot_msgs__msg__WheelCommands__Sequence__copy(
  const nuturtlebot_msgs__msg__WheelCommands__Sequence * input,
  nuturtlebot_msgs__msg__WheelCommands__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(nuturtlebot_msgs__msg__WheelCommands);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    nuturtlebot_msgs__msg__WheelCommands * data =
      (nuturtlebot_msgs__msg__WheelCommands *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!nuturtlebot_msgs__msg__WheelCommands__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          nuturtlebot_msgs__msg__WheelCommands__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!nuturtlebot_msgs__msg__WheelCommands__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
