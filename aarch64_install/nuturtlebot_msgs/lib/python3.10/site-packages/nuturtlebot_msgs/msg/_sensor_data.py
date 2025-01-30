# generated from rosidl_generator_py/resource/_idl.py.em
# with input from nuturtlebot_msgs:msg/SensorData.idl
# generated code does not contain a copyright notice

# This is being done at the module level and not on the instance level to avoid looking
# for the same variable multiple times on each instance. This variable is not supposed to
# change during runtime so it makes sense to only look for it once.
from os import getenv

ros_python_check_fields = getenv('ROS_PYTHON_CHECK_FIELDS', default='')


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_SensorData(type):
    """Metaclass of message 'SensorData'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('nuturtlebot_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'nuturtlebot_msgs.msg.SensorData')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__sensor_data
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__sensor_data
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__sensor_data
            cls._TYPE_SUPPORT = module.type_support_msg__msg__sensor_data
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__sensor_data

            from builtin_interfaces.msg import Time
            if Time.__class__._TYPE_SUPPORT is None:
                Time.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class SensorData(metaclass=Metaclass_SensorData):
    """Message class 'SensorData'."""

    __slots__ = [
        '_stamp',
        '_left_encoder',
        '_right_encoder',
        '_accel_x',
        '_accel_y',
        '_accel_z',
        '_gyro_x',
        '_gyro_y',
        '_gyro_z',
        '_mag_x',
        '_mag_y',
        '_mag_z',
        '_battery_voltage',
        '_check_fields',
    ]

    _fields_and_field_types = {
        'stamp': 'builtin_interfaces/Time',
        'left_encoder': 'int32',
        'right_encoder': 'int32',
        'accel_x': 'int16',
        'accel_y': 'int16',
        'accel_z': 'int16',
        'gyro_x': 'int16',
        'gyro_y': 'int16',
        'gyro_z': 'int16',
        'mag_x': 'int16',
        'mag_y': 'int16',
        'mag_z': 'int16',
        'battery_voltage': 'float',
    }

    # This attribute is used to store an rosidl_parser.definition variable
    # related to the data type of each of the components the message.
    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        if 'check_fields' in kwargs:
            self._check_fields = kwargs['check_fields']
        else:
            self._check_fields = ros_python_check_fields == '1'
        if self._check_fields:
            assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
                'Invalid arguments passed to constructor: %s' % \
                ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from builtin_interfaces.msg import Time
        self.stamp = kwargs.get('stamp', Time())
        self.left_encoder = kwargs.get('left_encoder', int())
        self.right_encoder = kwargs.get('right_encoder', int())
        self.accel_x = kwargs.get('accel_x', int())
        self.accel_y = kwargs.get('accel_y', int())
        self.accel_z = kwargs.get('accel_z', int())
        self.gyro_x = kwargs.get('gyro_x', int())
        self.gyro_y = kwargs.get('gyro_y', int())
        self.gyro_z = kwargs.get('gyro_z', int())
        self.mag_x = kwargs.get('mag_x', int())
        self.mag_y = kwargs.get('mag_y', int())
        self.mag_z = kwargs.get('mag_z', int())
        self.battery_voltage = kwargs.get('battery_voltage', float())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.get_fields_and_field_types().keys(), self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    if self._check_fields:
                        assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.stamp != other.stamp:
            return False
        if self.left_encoder != other.left_encoder:
            return False
        if self.right_encoder != other.right_encoder:
            return False
        if self.accel_x != other.accel_x:
            return False
        if self.accel_y != other.accel_y:
            return False
        if self.accel_z != other.accel_z:
            return False
        if self.gyro_x != other.gyro_x:
            return False
        if self.gyro_y != other.gyro_y:
            return False
        if self.gyro_z != other.gyro_z:
            return False
        if self.mag_x != other.mag_x:
            return False
        if self.mag_y != other.mag_y:
            return False
        if self.mag_z != other.mag_z:
            return False
        if self.battery_voltage != other.battery_voltage:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def stamp(self):
        """Message field 'stamp'."""
        return self._stamp

    @stamp.setter
    def stamp(self, value):
        if self._check_fields:
            from builtin_interfaces.msg import Time
            assert \
                isinstance(value, Time), \
                "The 'stamp' field must be a sub message of type 'Time'"
        self._stamp = value

    @builtins.property
    def left_encoder(self):
        """Message field 'left_encoder'."""
        return self._left_encoder

    @left_encoder.setter
    def left_encoder(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'left_encoder' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'left_encoder' field must be an integer in [-2147483648, 2147483647]"
        self._left_encoder = value

    @builtins.property
    def right_encoder(self):
        """Message field 'right_encoder'."""
        return self._right_encoder

    @right_encoder.setter
    def right_encoder(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'right_encoder' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'right_encoder' field must be an integer in [-2147483648, 2147483647]"
        self._right_encoder = value

    @builtins.property
    def accel_x(self):
        """Message field 'accel_x'."""
        return self._accel_x

    @accel_x.setter
    def accel_x(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'accel_x' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'accel_x' field must be an integer in [-32768, 32767]"
        self._accel_x = value

    @builtins.property
    def accel_y(self):
        """Message field 'accel_y'."""
        return self._accel_y

    @accel_y.setter
    def accel_y(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'accel_y' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'accel_y' field must be an integer in [-32768, 32767]"
        self._accel_y = value

    @builtins.property
    def accel_z(self):
        """Message field 'accel_z'."""
        return self._accel_z

    @accel_z.setter
    def accel_z(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'accel_z' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'accel_z' field must be an integer in [-32768, 32767]"
        self._accel_z = value

    @builtins.property
    def gyro_x(self):
        """Message field 'gyro_x'."""
        return self._gyro_x

    @gyro_x.setter
    def gyro_x(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'gyro_x' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'gyro_x' field must be an integer in [-32768, 32767]"
        self._gyro_x = value

    @builtins.property
    def gyro_y(self):
        """Message field 'gyro_y'."""
        return self._gyro_y

    @gyro_y.setter
    def gyro_y(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'gyro_y' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'gyro_y' field must be an integer in [-32768, 32767]"
        self._gyro_y = value

    @builtins.property
    def gyro_z(self):
        """Message field 'gyro_z'."""
        return self._gyro_z

    @gyro_z.setter
    def gyro_z(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'gyro_z' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'gyro_z' field must be an integer in [-32768, 32767]"
        self._gyro_z = value

    @builtins.property
    def mag_x(self):
        """Message field 'mag_x'."""
        return self._mag_x

    @mag_x.setter
    def mag_x(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'mag_x' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'mag_x' field must be an integer in [-32768, 32767]"
        self._mag_x = value

    @builtins.property
    def mag_y(self):
        """Message field 'mag_y'."""
        return self._mag_y

    @mag_y.setter
    def mag_y(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'mag_y' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'mag_y' field must be an integer in [-32768, 32767]"
        self._mag_y = value

    @builtins.property
    def mag_z(self):
        """Message field 'mag_z'."""
        return self._mag_z

    @mag_z.setter
    def mag_z(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'mag_z' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'mag_z' field must be an integer in [-32768, 32767]"
        self._mag_z = value

    @builtins.property
    def battery_voltage(self):
        """Message field 'battery_voltage'."""
        return self._battery_voltage

    @battery_voltage.setter
    def battery_voltage(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'battery_voltage' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'battery_voltage' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._battery_voltage = value
