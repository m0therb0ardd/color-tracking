import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/catherine-maglione/yolo_home/install_me495_yolo/me495_yolo'
