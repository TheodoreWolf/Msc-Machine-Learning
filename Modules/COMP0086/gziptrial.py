import zlib
import struct
import ctypes
like = -3495.29176507


def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

