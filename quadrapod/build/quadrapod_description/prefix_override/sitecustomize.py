import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/dream-hell/sleipnir/quadrapod/install/quadrapod_description'
