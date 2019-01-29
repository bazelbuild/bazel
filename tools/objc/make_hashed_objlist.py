# pylint: disable=g-bad-file-header
# Copyright 2016 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Creates symbolic links for .o files with hashcode.

This script reads the file list containing the input files, creates
symbolic links with a path-hash appended to their original name (foo.o
becomes foo_{md5sum}.o), then saves the list of symbolic links to another
file.

The symbolic links are created into the given temporary directory.  There is
no guarantee that we can write to the directory that contained the inputs to
this script.

This is to circumvent a bug in the original libtool that arises when two
input files have the same base name (even if they are in different
directories).
"""

import hashlib
import os
import sys


def main():
  outdir = sys.argv[3]
  with open(sys.argv[1]) as obj_file_list:
    with open(sys.argv[2], 'w') as hashed_obj_file_list:
      for line in obj_file_list:
        obj_file_path = line.rstrip('\n')

        hashed_obj_file_name = '%s_%s.o' % (
            os.path.basename(os.path.splitext(obj_file_path)[0]),
            hashlib.md5(obj_file_path.encode('utf-8')).hexdigest())
        hashed_obj_file_path = os.path.join(outdir, hashed_obj_file_name)

        hashed_obj_file_list.write(hashed_obj_file_path + '\n')

        # Create symlink only if the symlink doesn't exist.
        if not os.path.exists(hashed_obj_file_path):
          os.symlink(os.path.abspath(obj_file_path),
                     hashed_obj_file_path)


if __name__ == '__main__':
  main()
