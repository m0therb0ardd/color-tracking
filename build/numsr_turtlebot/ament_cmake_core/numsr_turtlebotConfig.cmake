# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_numsr_turtlebot_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED numsr_turtlebot_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(numsr_turtlebot_FOUND FALSE)
  elseif(NOT numsr_turtlebot_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(numsr_turtlebot_FOUND FALSE)
  endif()
  return()
endif()
set(_numsr_turtlebot_CONFIG_INCLUDED TRUE)

# output package information
if(NOT numsr_turtlebot_FIND_QUIETLY)
  message(STATUS "Found numsr_turtlebot: 0.0.1 (${numsr_turtlebot_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'numsr_turtlebot' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT numsr_turtlebot_DEPRECATED_QUIET)
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(numsr_turtlebot_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${numsr_turtlebot_DIR}/${_extra}")
endforeach()
