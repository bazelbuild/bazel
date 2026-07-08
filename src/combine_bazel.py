#!/usr/bin/env python3

import struct
import sys
from pathlib import Path

CENTRAL_DIRECTORY_SIGNATURE = b"PK\x01\x02"
END_OF_CENTRAL_DIRECTORY_SIGNATURE = b"PK\x05\x06"


def main():
    client = Path(sys.argv[1])
    package = Path(sys.argv[2])
    output = Path(sys.argv[3])

    client_data = client.read_bytes()
    package_data = bytearray(package.read_bytes())
    prefix_length = len(client_data)
    end_offset = package_data.rfind(END_OF_CENTRAL_DIRECTORY_SIGNATURE)
    if end_offset < 0:
        raise ValueError("package does not contain a ZIP end record")

    entry_count = struct.unpack_from("<H", package_data, end_offset + 10)[0]
    directory_offset = struct.unpack_from("<I", package_data, end_offset + 16)[0]
    cursor = directory_offset
    for _ in range(entry_count):
        if package_data[cursor:cursor + 4] != CENTRAL_DIRECTORY_SIGNATURE:
            raise ValueError("invalid ZIP central directory")
        local_header_offset = struct.unpack_from("<I", package_data, cursor + 42)[0]
        struct.pack_into("<I", package_data, cursor + 42, local_header_offset + prefix_length)
        name_length, extra_length, comment_length = struct.unpack_from("<HHH", package_data, cursor + 28)
        cursor += 46 + name_length + extra_length + comment_length

    struct.pack_into("<I", package_data, end_offset + 16, directory_offset + prefix_length)
    output.write_bytes(client_data + package_data)
    output.chmod(client.stat().st_mode | 0o111)


if __name__ == "__main__":
    main()
