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

import hashlib
import os.path
from StringIO import StringIO
import sys
import tarfile
import textwrap
import time

from third_party.py import gflags

# list of debian fields : (name, mandatory, wrap[, default])
# see http://www.debian.org/doc/debian-policy/ch-controlfields.html
DEBIAN_FIELDS = [
    ('Package', True, False), ('Version', True, False),
    ('Section', False, False, 'contrib/devel'),
    ('Priority', False, False, 'optional'),
    ('Architecture', True, False, 'all'), ('Depends', False, True, []),
    ('Recommends', False, True, []), ('Suggests', False, True, []),
    ('Enhances', False, True, []), ('Conflicts', False, True, []),
    ('Pre-Depends', False, True, []), ('Installed-Size', False, False),
    ('Maintainer', True, False), ('Description', True, True),
    ('Homepage', False, False), ('Built-Using', False, False, 'Bazel')
]

gflags.DEFINE_string('output', None, 'The output file, mandatory')
gflags.MarkFlagAsRequired('output')

gflags.DEFINE_string('changes', None, 'The changes output file, mandatory.')
gflags.MarkFlagAsRequired('changes')

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
                   owner_id=0, group_id=0, mode=0o644):
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
    result = textwrap.fill(result,
                           break_on_hyphens=False,
                           break_long_words=False)
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
        tarinfo.mode = 0o755
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
    # Tries to preserve the extension name
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


def GetChecksumsFromFile(filename, hash_fns=None):
  """Computes MD5 and/or other checksums of a file.

  Args:
    filename: Name of the file.
    hash_fns: Mapping of hash functions.
              Default is {'md5': hashlib.md5}

  Returns:
    Mapping of hash names to hexdigest strings.
    { <hashname>: <hexdigest>, ... }
  """
  hash_fns = hash_fns or {'md5': hashlib.md5}
  checksums = {k: fn() for (k, fn) in hash_fns.items()}

  with open(filename) as file_handle:
    while True:
      buf = file_handle.read(1048576)  # 1 MiB
      if not buf:
        break
      for hashfn in checksums.values():
        hashfn.update(buf)

  return {k: fn.hexdigest() for (k, fn) in checksums.items()}


def CreateChanges(output,
                  deb_file,
                  architecture,
                  short_description,
                  maintainer,
                  package,
                  version,
                  section,
                  priority,
                  timestamp=0,
                  distro='unstable',
                  urgency='medium'):
  """Create the changes file."""
  checksums = GetChecksumsFromFile(deb_file, {'md5': hashlib.md5,
                                              'sha1': hashlib.sha1,
                                              'sha256': hashlib.sha256})
  debsize = str(os.path.getsize(deb_file))
  deb_basename = os.path.basename(deb_file)

  changesdata = ''.join(MakeDebianControlField(*x) for x in [
      ('Format', '1.8'),
      ('Date', time.ctime(timestamp)),
      ('Source', package),
      ('Binary', package),
      ('Architecture', architecture),
      ('Version', version),
      ('Distribution', distro),
      ('Urgency', urgency),
      ('Maintainer', maintainer),
      ('Changed-By', maintainer),
      ('Description', '\n%s - %s' % (package, short_description)),
      ('Changes',
       ('\n%s (%s) %s; urgency=%s'
        '\nChanges are tracked in revision control.') % (
            package, version, distro, urgency)),
      ('Files', '\n' + ' '.join(
          [checksums['md5'], debsize, section, priority, deb_basename])),
      ('Checksums-Sha1', '\n' + ' '.join(
          [checksums['sha1'], debsize, deb_basename])),
      ('Checksums-Sha256', '\n' + ' '.join(
          [checksums['sha256'], debsize, deb_basename]))
      ])
  with open(output, 'w') as changes_fh:
    changes_fh.write(changesdata)


def GetFlagValue(flagvalue, strip=True):
  if flagvalue:
    if flagvalue[0] == '@':
      with open(flagvalue[1:], 'r') as f:
        flagvalue = f.read()
    if strip:
      return flagvalue.strip()
  return flagvalue


def main(unused_argv):
  CreateDeb(FLAGS.output,
            FLAGS.data,
            preinst=GetFlagValue(FLAGS.preinst, False),
            postinst=GetFlagValue(FLAGS.postinst, False),
            prerm=GetFlagValue(FLAGS.prerm, False),
            postrm=GetFlagValue(FLAGS.postrm, False),
            package=FLAGS.package,
            version=GetFlagValue(FLAGS.version),
            description=GetFlagValue(FLAGS.description),
            maintainer=FLAGS.maintainer,
            section=FLAGS.section,
            architecture=FLAGS.architecture,
            depends=FLAGS.depends,
            suggests=FLAGS.suggests,
            enhances=FLAGS.enhances,
            preDepends=FLAGS.pre_depends,
            recommends=FLAGS.recommends,
            homepage=FLAGS.homepage,
            builtUsing=GetFlagValue(FLAGS.built_using),
            priority=FLAGS.priority,
            conflicts=FLAGS.conflicts,
            installedSize=GetFlagValue(FLAGS.installed_size))
  CreateChanges(
      FLAGS.changes,
      FLAGS.output,
      architecture=FLAGS.architecture,
      short_description=GetFlagValue(FLAGS.description).split('\n')[0],
      maintainer=FLAGS.maintainer,
      package=FLAGS.package,
      version=GetFlagValue(FLAGS.version),
      section=FLAGS.section,
      priority=FLAGS.priority)

if __name__ == '__main__':
  MakeGflags()
  FLAGS = gflags.FLAGS
  main(FLAGS(sys.argv))
