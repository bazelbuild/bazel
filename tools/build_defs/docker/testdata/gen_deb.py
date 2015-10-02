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
"""A simple cross-platform helper to create a dummy debian package."""
from StringIO import StringIO
import sys
import tarfile


def AddArFileEntry(fileobj, filename, content=''):
  """Add a AR file entry to fileobj."""
  fileobj.write((filename + '/').ljust(16))    # filename (SysV)
  fileobj.write('0'.ljust(12))                 # timestamp
  fileobj.write('0'.ljust(6))                  # owner id
  fileobj.write('0'.ljust(6))                  # group id
  fileobj.write('0644'.ljust(8))               # mode
  fileobj.write(str(len(content)).ljust(10))   # size
  fileobj.write('\x60\x0a')                    # end of file entry
  fileobj.write(content)
  if len(content) % 2 != 0:
    fileobj.write('\n')  # 2-byte alignment padding

if __name__ == '__main__':
  # Create data.tar
  tar = StringIO()
  with tarfile.open('data.tar', mode='w', fileobj=tar) as f:
    tarinfo = tarfile.TarInfo('usr/')
    tarinfo.type = tarfile.DIRTYPE
    f.addfile(tarinfo)
    tarinfo = tarfile.TarInfo('usr/titi')
    f.addfile(tarinfo, fileobj=StringIO('toto\n'))
  data = tar.getvalue()
  tar.close()
  # Create control.tar
  tar = StringIO()
  with tarfile.open('control.tar', mode='w', fileobj=tar) as f:
    tarinfo = tarfile.TarInfo('control')
    f.addfile(tarinfo, fileobj=StringIO('\n'.join([
        'Package: test'
        'Description: Just a dummy test'
        ])))
  control = tar.getvalue()
  tar.close()

  # Write the final AR archive (the deb package)
  with open(sys.argv[1], 'w') as f:
    f.write('!<arch>\n')  # Magic AR header
    AddArFileEntry(f, 'debian-binary', '2.0')
    AddArFileEntry(f, 'control.tar', control)
    AddArFileEntry(f, 'data.tar', data)
