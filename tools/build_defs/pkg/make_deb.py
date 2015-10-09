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
"""A simple cross-platform helper to create a debian package."""

import os.path
from StringIO import StringIO
import sys
import tarfile
import textwrap

from third_party.py import gflags

# list of debian fields : (name, mandatory, wrap[, default])
# see http://www.debian.org/doc/debian-policy/ch-controlfields.html
DEBIAN_FIELDS = [
    ('Package', True, False),
    ('Version', True, False),
    ('Section', False, False, 'contrib/devel'),
    ('Priority', False, False, 'optional'),
    ('Architecture', True, False, 'all'),
    ('Depends', False, True, []),
    ('Recommends', False, True, []),
    ('Suggests', False, True, []),
    ('Enhances', False, True, []),
    ('Pre-Depends', False, True, []),
    ('Installed-Size', False, False),
    ('Maintainer', True, False),
    ('Description', True, True),
    ('Homepage', False, False),
    ('Built-Using', False, False, 'Bazel')
    ]

gflags.DEFINE_string(
    'output', None,
    'The output file, mandatory')
gflags.MarkFlagAsRequired('output')

gflags.DEFINE_string('data', None,
                     'Path to the data tarball, mandatory')
gflags.MarkFlagAsRequired('data')

gflags.DEFINE_string('preinst', None,
                     'The preinst script (prefix with @ to provide a path).')
gflags.DEFINE_string('postinst', None,
                     'The postinst script (prefix with @ to provide a path).')
gflags.DEFINE_string('prerm', None,
                     'The prerm script (prefix with @ to provide a path).')
gflags.DEFINE_string('postrm', None,
                     'The postrm script (prefix with @ to provide a path).')


def MakeGflags():
  for field in DEBIAN_FIELDS:
    fieldname = field[0].replace('-', '_').lower()
    msg = 'The value for the %s content header entry.' % field[0]
    if len(field) > 3:
      if type(field[3]) is list:
        gflags.DEFINE_multistring(fieldname, field[3], msg)
      else:
        gflags.DEFINE_string(fieldname, field[3], msg)
    else:
      gflags.DEFINE_string(fieldname, None, msg)
    if field[1]:
      gflags.MarkFlagAsRequired(fieldname)


def AddArFileEntry(fileobj, filename,
                   content='', timestamp=0,
                   owner_id=0, group_id=0, mode=0644):
  """Add a AR file entry to fileobj."""
  fileobj.write((filename + '/').ljust(16))    # filename (SysV)
  fileobj.write(str(timestamp).ljust(12))      # timestamp
  fileobj.write(str(owner_id).ljust(6))        # owner id
  fileobj.write(str(group_id).ljust(6))        # group id
  fileobj.write(oct(mode).ljust(8))            # mode
  fileobj.write(str(len(content)).ljust(10))   # size
  fileobj.write('\x60\x0a')                    # end of file entry
  fileobj.write(content)
  if len(content) % 2 != 0:
    fileobj.write('\n')  # 2-byte alignment padding


def MakeDebianControlField(name, value, wrap=False):
  """Add a field to a debian control file."""
  result = name + ': '
  if type(value) is list:
    value = ', '.join(value)
  if wrap:
    result += ' '.join(value.split('\n'))
    result = textwrap.fill(result)
  else:
    result += value
  return result.replace('\n', '\n ') + '\n'


def CreateDebControl(extrafiles=None, **kwargs):
  """Create the control.tar.gz file."""
  # create the control file
  controlfile = ''
  for values in DEBIAN_FIELDS:
    fieldname = values[0]
    key = fieldname[0].lower() + fieldname[1:].replace('-', '')
    if values[1] or (key in kwargs and kwargs[key]):
      controlfile += MakeDebianControlField(fieldname, kwargs[key], values[2])
  # Create the control.tar file
  tar = StringIO()
  with tarfile.open('control.tar.gz', mode='w:gz', fileobj=tar) as f:
    tarinfo = tarfile.TarInfo('control')
    tarinfo.size = len(controlfile)
    f.addfile(tarinfo, fileobj=StringIO(controlfile))
    if extrafiles:
      for name in extrafiles:
        tarinfo = tarfile.TarInfo(name)
        tarinfo.size = len(extrafiles[name])
        f.addfile(tarinfo, fileobj=StringIO(extrafiles[name]))
  control = tar.getvalue()
  tar.close()
  return control


def CreateDeb(output, data,
              preinst=None, postinst=None, prerm=None, postrm=None, **kwargs):
  """Create a full debian package."""
  extrafiles = {}
  if preinst:
    extrafiles['preinst'] = preinst
  if postinst:
    extrafiles['postinst'] = postinst
  if prerm:
    extrafiles['prerm'] = prerm
  if postrm:
    extrafiles['postrm'] = postrm
  control = CreateDebControl(extrafiles=extrafiles, **kwargs)

  # Write the final AR archive (the deb package)
  with open(output, 'w') as f:
    f.write('!<arch>\n')  # Magic AR header
    AddArFileEntry(f, 'debian-binary', '2.0\n')
    AddArFileEntry(f, 'control.tar.gz', control)
    # Tries to presever the extension name
    ext = os.path.basename(data).split('.')[-2:]
    if len(ext) < 2:
      ext = 'tar'
    elif ext[1] == 'tgz':
      ext = 'tar.gz'
    elif ext[1] == 'tar.bzip2':
      ext = 'tar.bz2'
    else:
      ext = '.'.join(ext)
      if ext not in ['tar.bz2', 'tar.gz', 'tar.xz', 'tar.lzma']:
        ext = 'tar'
    with open(data, 'r') as datafile:
      data = datafile.read()
    AddArFileEntry(f, 'data.' + ext, data)


def GetFlagValue(flagvalue, strip=True):
  if flagvalue:
    if flagvalue[0] == '@':
      with open(flagvalue[1:], 'r') as f:
        flagvalue = f.read()
    if strip:
      return flagvalue.strip()
  return flagvalue


def main(unused_argv):
  CreateDeb(FLAGS.output, FLAGS.data,
            preinst=GetFlagValue(FLAGS.preinst, False),
            postinst=GetFlagValue(FLAGS.postinst, False),
            prerm=GetFlagValue(FLAGS.prerm, False),
            postrm=GetFlagValue(FLAGS.postrm, False),
            package=FLAGS.package, version=GetFlagValue(FLAGS.version),
            description=GetFlagValue(FLAGS.description),
            maintainer=FLAGS.maintainer,
            section=FLAGS.section, architecture=FLAGS.architecture,
            depends=FLAGS.depends, suggests=FLAGS.suggests,
            enhances=FLAGS.enhances, preDepends=FLAGS.pre_depends,
            recommends=FLAGS.recommends, homepage=FLAGS.homepage,
            builtUsing=GetFlagValue(FLAGS.built_using),
            priority=FLAGS.priority,
            installedSize=GetFlagValue(FLAGS.installed_size))

if __name__ == '__main__':
  MakeGflags()
  FLAGS = gflags.FLAGS
  main(FLAGS(sys.argv))
