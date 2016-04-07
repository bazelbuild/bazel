# Copyright (c) 2015, Google Inc.
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
# OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""Extracts archives."""


import optparse
import os
import os.path
import tarfile
import shutil
import sys
import zipfile


def CheckedJoin(output, path):
  """
  CheckedJoin returns os.path.join(output, path). It does sanity checks to
  ensure the resulting path is under output, but shouldn't be used on untrusted
  input.
  """
  path = os.path.normpath(path)
  if os.path.isabs(path) or path.startswith('.'):
    raise ValueError(path)
  return os.path.join(output, path)


def IterateZip(path):
  """
  IterateZip opens the zip file at path and returns a generator of
  (filename, mode, fileobj) tuples for each file in it.
  """
  with zipfile.ZipFile(path, 'r') as zip_file:
    for info in zip_file.infolist():
      if info.filename.endswith('/'):
        continue
      yield (info.filename, None, zip_file.open(info))


def IterateTar(path):
  """
  IterateTar opens the tar.gz file at path and returns a generator of
  (filename, mode, fileobj) tuples for each file in it.
  """
  with tarfile.open(path, 'r:gz') as tar_file:
    for info in tar_file:
      if info.isdir():
        continue
      if not info.isfile():
        raise ValueError('Unknown entry type "%s"' % (info.name, ))
      yield (info.name, info.mode, tar_file.extractfile(info))


def main(args):
  parser = optparse.OptionParser(usage='Usage: %prog ARCHIVE OUTPUT')
  parser.add_option('--no-prefix', dest='no_prefix', action='store_true',
                    help='Do not remove a prefix from paths in the archive.')
  options, args = parser.parse_args(args)

  if len(args) != 2:
    parser.print_help()
    return 1

  archive, output = args

  if not os.path.exists(archive):
    # Skip archives that weren't downloaded.
    return 0

  if archive.endswith('.zip'):
    entries = IterateZip(archive)
  elif archive.endswith('.tar.gz'):
    entries = IterateTar(archive)
  else:
    raise ValueError(archive)

  try:
    if os.path.exists(output):
      print "Removing %s" % (output, )
      shutil.rmtree(output)

    print "Extracting %s to %s" % (archive, output)
    prefix = None
    num_extracted = 0
    for path, mode, inp in entries:
      # Even on Windows, zip files must always use forward slashes.
      if '\\' in path or path.startswith('/'):
        raise ValueError(path)

      if not options.no_prefix:
        new_prefix, rest = path.split('/', 1)

        # Ensure the archive is consistent.
        if prefix is None:
          prefix = new_prefix
        if prefix != new_prefix:
          raise ValueError((prefix, new_prefix))
      else:
        rest = path

      # Extract the file into the output directory.
      fixed_path = CheckedJoin(output, rest)
      if not os.path.isdir(os.path.dirname(fixed_path)):
        os.makedirs(os.path.dirname(fixed_path))
      with open(fixed_path, 'wb') as out:
        shutil.copyfileobj(inp, out)

      # Fix up permissions if needbe.
      # TODO(davidben): To be extra tidy, this should only track the execute bit
      # as in git.
      if mode is not None:
        os.chmod(fixed_path, mode)

      # Print every 100 files, so bots do not time out on large archives.
      num_extracted += 1
      if num_extracted % 100 == 0:
        print "Extracted %d files..." % (num_extracted,)
  finally:
    entries.close()

  if num_extracted % 100 == 0:
    print "Done. Extracted %d files." % (num_extracted,)

  return 0


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
