# Copyright 2015 The Bazel Authors. All rights reserved.
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
"""A simple cross-platform helper to create a timestamped tar file."""
import datetime
import sys
import tarfile

from tools.build_defs.pkg import archive

if __name__ == '__main__':
  mtime = int(datetime.datetime.now().strftime('%s'))
  with archive.TarFileWriter(sys.argv[1]) as f:
    f.add_file('./', tarfile.DIRTYPE,
               uname='root', gname='root', mtime=mtime)
    f.add_file('./usr/', tarfile.DIRTYPE,
               uname='root', gname='root', mtime=mtime)
    f.add_file('./usr/bin/', tarfile.DIRTYPE,
               uname='root', gname='root', mtime=mtime)
    f.add_file('./usr/bin/java', tarfile.SYMTYPE,
               link='/path/to/bin/java',
               uname='root', gname='root', mtime=mtime)
    f.add_file('./etc/', tarfile.DIRTYPE,
               uname='root', gname='root', mtime=mtime)
    f.add_file('./etc/nsswitch.conf',
               content='hosts:          files dns\n',
               uname='root', gname='root', mtime=mtime)
    f.add_file('./tmp/', tarfile.DIRTYPE,
               uname='root', gname='root', mtime=mtime)
