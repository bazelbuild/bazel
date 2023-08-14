#!/usr/bin/python3

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

"""A script for J2ObjC dead code removal in Blaze.

This script removes unused J2ObjC-translated classes from compilation and
linking by:
  1. Build a class dependency tree among translated source files.
  2. Use user-provided Java class entry points to get a list of reachable
     classes.
  3. Go through all translated source files and rewrite unreachable ones with
     dummy content.
"""

import argparse
import collections
import multiprocessing
import os
import queue
import re
import shlex
import shutil
import subprocess
import threading


PRUNED_SRC_CONTENT = 'static int DUMMY_unused __attribute__((unused,used)) = 0;'


def BuildReachabilityTree(dependency_mapping_files, file_open=open):
  """Builds a reachability tree using entries from dependency mapping files.

  Args:
    dependency_mapping_files: A comma separated list of J2ObjC-generated
        dependency mapping files.
    file_open: Reference to the builtin open function so it may be
        overridden for testing.
  Returns:
    A dict mapping J2ObjC-generated source files to the corresponding direct
    dependent source files.
  """
  return BuildArtifactSourceTree(dependency_mapping_files, file_open)


def BuildHeaderMapping(header_mapping_files, file_open=open):
  """Builds a mapping between Java classes and J2ObjC-translated header files.

  Args:
    header_mapping_files: A comma separated list of J2ObjC-generated
        header mapping files.
    file_open: Reference to the builtin open function so it may be
        overridden for testing.
  Returns:
    An ordered dict mapping Java class names to corresponding J2ObjC-translated
    source files.
  """
  header_mapping = collections.OrderedDict()
  for header_mapping_file in header_mapping_files.split(','):
    with file_open(header_mapping_file, 'r') as f:
      for line in f:
        java_class_name = line.strip().split('=')[0]
        transpiled_file_name = os.path.splitext(line.strip().split('=')[1])[0]
        header_mapping[java_class_name] = transpiled_file_name
  return header_mapping


def BuildReachableFileSet(entry_classes, reachability_tree, header_mapping,
                          archive_source_file_mapping=None):
  """Builds a set of reachable translated files from entry Java classes.

  Args:
    entry_classes: A comma separated list of Java entry classes.
    reachability_tree: A dict mapping translated files to their direct
        dependencies.
    header_mapping: A dict mapping Java class names to translated source files.
    archive_source_file_mapping: A dict mapping source files to the associated
        archive file that contains them.
  Returns:
    A set of reachable translated files from the given list of entry classes.
  Raises:
    Exception: If there is an entry class that is not being transpiled in this
        j2objc_library.
  """
  transpiled_entry_files = []
  for entry_class in entry_classes.split(','):
    if entry_class not in header_mapping:
      raise Exception(
          entry_class +
          ' is not in the transitive Java deps of included ' +
          'j2objc_library rules.')
    transpiled_entry_files.append(header_mapping[entry_class])

  # Translated files going into the same static library archive with duplicated
  # base names also need to be added to the set of entry files.
  #
  # This edge case is ignored because we currently cannot correctly perform
  # dead code removal in this case. The object file entries in static library
  # archives are named by the base names of the original source files. If two
  # source files (e.g., foo/bar.m, bar/bar.m) go into the same archive and
  # share the same base name (bar.m), their object file entries inside the
  # archive will have the same name (bar.o). We cannot correctly handle this
  # case because current archive tools (ar, ranlib, etc.) do not handle this
  # case very well.
  if archive_source_file_mapping:
    transpiled_entry_files.extend(_DuplicatedFiles(archive_source_file_mapping))

  # Translated files from package-info.java are also added to the entry files
  # because they are needed to resolve ObjC class names with prefixes and these
  # files may also have dependencies.
  for transpiled_file in reachability_tree:
    if transpiled_file.endswith('package-info'):
      transpiled_entry_files.append(transpiled_file)

  reachable_files = set()
  for transpiled_entry_file in transpiled_entry_files:
    reachable_files.add(transpiled_entry_file)
    current_level_deps = []
    # We need to check if the transpiled file is in the reachability tree
    # because J2ObjC protos are not analyzed for dead code stripping and
    # therefore are not in the reachability tree at all.
    if transpiled_entry_file in reachability_tree:
      current_level_deps = reachability_tree[transpiled_entry_file]
    while current_level_deps:
      next_level_deps = []
      for dep in current_level_deps:
        if dep not in reachable_files:
          reachable_files.add(dep)
          if dep in reachability_tree:
            next_level_deps.extend(reachability_tree[dep])
      current_level_deps = next_level_deps
  return reachable_files


def PruneFiles(input_files, output_files, objc_file_path, reachable_files,
               file_open=open, file_shutil=shutil):
  """Copies over translated files and remove the contents of unreachable files.

  Args:
    input_files: A comma separated list of input source files to prune. It has
        a one-on-one pair mapping with the output_file list.
    output_files: A comma separated list of output source files to write pruned
        source files to. It has a one-on-one pair mapping with the input_file
        list.
    objc_file_path: The file path which represents a directory where the
        generated ObjC files reside.
    reachable_files: A set of reachable source files.
    file_open: Reference to the builtin open function so it may be
        overridden for testing.
    file_shutil: Reference to the builtin shutil module so it may be
        overridden for testing.
  Returns:
    None.
  """
  file_queue = queue.queue()
  for input_file, output_file in zip(
      input_files.split(','),
      output_files.split(',')):
    file_queue.put((input_file, output_file))

  for _ in range(multiprocessing.cpu_count()):
    t = threading.Thread(target=_PruneFile, args=(file_queue,
                                                  reachable_files,
                                                  objc_file_path,
                                                  file_open,
                                                  file_shutil))
    t.start()

  file_queue.join()


def _PruneFile(file_queue, reachable_files, objc_file_path, file_open=open,
               file_shutil=shutil):
  while True:
    try:
      input_file, output_file = file_queue.get_nowait()
    except queue.Empty:
      return
    file_name = os.path.relpath(os.path.splitext(input_file)[0],
                                objc_file_path)
    if file_name in reachable_files:
      file_shutil.copy(input_file, output_file)
    else:
      with file_open(output_file, 'w') as f:
        # Use a static variable scoped to the source file to suppress
        # the "has no symbols" linker warning for empty object files.
        f.write(PRUNED_SRC_CONTENT)
    file_queue.task_done()


def _DuplicatedFiles(archive_source_file_mapping):
  """Returns a list of file with duplicated base names in each archive file.

  Args:
    archive_source_file_mapping: A dict mapping source files to the associated
        archive file that contains them.
  Returns:
    A list containing files with duplicated base names.
  """
  duplicated_files = []
  dict_with_duplicates = dict()

  for source_files in archive_source_file_mapping.values():
    for source_file in source_files:
      file_basename = os.path.basename(source_file)
      file_without_ext = os.path.splitext(source_file)[0]
      if file_basename in dict_with_duplicates:
        dict_with_duplicates[file_basename].append(file_without_ext)
      else:
        dict_with_duplicates[file_basename] = [file_without_ext]
    for basename in dict_with_duplicates:
      if len(dict_with_duplicates[basename]) > 1:
        duplicated_files.extend(dict_with_duplicates[basename])
    dict_with_duplicates = dict()

  return duplicated_files


def BuildArchiveSourceFileMapping(archive_source_mapping_files, file_open):
  """Builds a mapping between archive files and their associated source files.

  Args:
    archive_source_mapping_files: A comma separated list of J2ObjC-generated
        mapping between archive files and their associated source files.
    file_open: Reference to the builtin open function so it may be
        overridden for testing.
  Returns:
    A dict mapping between archive files and their associated source files.
  """
  return BuildArtifactSourceTree(archive_source_mapping_files, file_open)


def PruneSourceFiles(input_files, output_files, dependency_mapping_files,
                     header_mapping_files, entry_classes, objc_file_path,
                     file_open=open, file_shutil=shutil):
  """Copies over translated files and remove the contents of unreachable files.

  Args:
    input_files: A comma separated list of input source files to prune. It has
        a one-on-one pair mapping with the output_file list.
    output_files: A comma separated list of output source files to write pruned
        source files to. It has a one-on-one pair mapping with the input_file
        list.
    dependency_mapping_files: A comma separated list of J2ObjC-generated
        dependency mapping files.
    header_mapping_files: A comma separated list of J2ObjC-generated
        header mapping files.
    entry_classes: A comma separated list of Java entry classes.
    objc_file_path: The file path which represents a directory where the
        generated ObjC files reside.
    file_open: Reference to the builtin open function so it may be
        overridden for testing.
    file_shutil: Reference to the builtin shutil module so it may be
        overridden for testing.
  """
  reachability_file_mapping = BuildReachabilityTree(
      dependency_mapping_files, file_open)
  header_map = BuildHeaderMapping(header_mapping_files, file_open)
  reachable_files_set = BuildReachableFileSet(entry_classes,
                                              reachability_file_mapping,
                                              header_map)
  PruneFiles(input_files,
             output_files,
             objc_file_path,
             reachable_files_set,
             file_open,
             file_shutil)


def MatchObjectNamesInArchive(archive, object_names):
  """Returns object names matching their identity in an archive file.

  The linker that blaze uses appends an md5 hash to object file
  names prior to inclusion in the archive file. Thus, object names
  such as 'foo.o' need to be matched to their appropriate name in
  the archive file, such as 'foo_<hash>.o'.

  Args:
    archive: The location of the archive file.
    object_names: The expected basenames of object files to match,
        sans extension. For example 'foo' (not 'foo.o').
  Returns:
    A list of basenames of matching members of the given archive
  """
  ar_contents_cmd = ['/usr/bin/xcrun', 'ar', '-t', archive]
  real_object_names_output = subprocess.check_output(ar_contents_cmd)
  real_object_names = real_object_names_output.decode('utf-8')
  expected_object_name_regex = r'^(?:%s)(?:_[0-9a-f]{32}(?:-[0-9]+)?)?\.o$' % (
      '|'.join([re.escape(name) for name in object_names]))
  return re.findall(
      expected_object_name_regex,
      real_object_names,
      flags=re.MULTILINE)


def PruneArchiveFile(
    input_archive,
    output_archive,
    dummy_archive,
    dependency_mapping_files,
    header_mapping_files,
    archive_source_mapping_files,
    entry_classes,
    file_open=open,
):
  """Remove unreachable objects from archive file.

  Args:
    input_archive: The source archive file to prune.
    output_archive: The location of the pruned archive file.
    dummy_archive: A dummy archive file that contains no object.
    dependency_mapping_files: A comma separated list of J2ObjC-generated
        dependency mapping files.
    header_mapping_files: A comma separated list of J2ObjC-generated
        header mapping files.
    archive_source_mapping_files: A comma separated list of J2ObjC-generated
        mapping between archive files and their associated source files.
    entry_classes: A comma separated list of Java entry classes.
    file_open: Reference to the builtin open function so it may be
        overridden for testing.
  """
  reachability_file_mapping = BuildReachabilityTree(
      dependency_mapping_files, file_open)
  header_map = BuildHeaderMapping(header_mapping_files, file_open)
  archive_source_file_mapping = BuildArchiveSourceFileMapping(
      archive_source_mapping_files, file_open)
  reachable_files_set = BuildReachableFileSet(entry_classes,
                                              reachability_file_mapping,
                                              header_map,
                                              archive_source_file_mapping)

  # Copy the current processes' environment.
  cmd_env = dict(os.environ)
  j2objc_cmd = ''
  if input_archive in archive_source_file_mapping:
    source_files = archive_source_file_mapping[input_archive]
    unreachable_object_names = []

    for source_file in source_files:
      if os.path.splitext(source_file)[0] not in reachable_files_set:
        unreachable_object_names.append(
            os.path.basename(os.path.splitext(source_file)[0]))

    # There are unreachable objects in the archive to prune
    if unreachable_object_names:
      # If all objects in the archive are unreachable, just copy over a dummy
      # archive that contains no object
      if len(unreachable_object_names) == len(source_files):
        j2objc_cmd = 'cp %s %s' % (shlex.quote(dummy_archive),
                                   shlex.quote(output_archive))
      # Else we need to prune the archive of unreachable objects
      else:
        cmd_env['ZERO_AR_DATE'] = '1'
        # Copy the input archive to the output location
        j2objc_cmd += 'cp %s %s && ' % (shlex.quote(input_archive),
                                        shlex.quote(output_archive))
        # Make the output archive editable
        j2objc_cmd += 'chmod +w %s && ' % (shlex.quote(output_archive))
        # Remove the unreachable objects from the archive
        unreachable_object_names = MatchObjectNamesInArchive(
            input_archive, unreachable_object_names
        )
        j2objc_cmd += '/usr/bin/xcrun ar -d -s %s %s && ' % (
            shlex.quote(output_archive),
            ' '.join(shlex.quote(uon) for uon in unreachable_object_names),
        )
        # Update the table of content of the archive file
        j2objc_cmd += '/usr/bin/xcrun ranlib %s' % shlex.quote(output_archive)
    # There are no unreachable objects, we just copy over the original archive
    else:
      j2objc_cmd = 'cp %s %s' % (shlex.quote(input_archive),
                                 shlex.quote(output_archive))
  # The archive cannot be pruned by J2ObjC dead code removal, just copy over
  # the original archive
  else:
    j2objc_cmd = 'cp %s %s' % (shlex.quote(input_archive),
                               shlex.quote(output_archive))

  try:
    subprocess.check_output(
        j2objc_cmd, stderr=subprocess.STDOUT, shell=True, env=cmd_env)
  except OSError as e:
    raise Exception(
        'executing command failed: %s (%s)' % (j2objc_cmd, e.strerror))

  # "Touch" the output file.
  # Prevents a pre-Xcode-8 bug in which passing zero-date archive files to ld
  # would cause ld to error.
  os.utime(output_archive, None)


def BuildArtifactSourceTree(files, file_open=open):
  """Builds a dependency tree using from dependency mapping files.

  Args:
   files: A comma separated list of dependency mapping files.
   file_open: Reference to the builtin open function so it may be overridden for
     testing.

  Returns:
   A dict mapping build artifacts (possibly generated source files) to the
   corresponding direct dependent source files.
  """
  tree = dict()
  if not files:
    return tree
  for filename in files.split(','):
    with file_open(filename, 'r') as f:
      for line in f:
        split = line.strip().split(':')
        entry = split[0]
        if len(split) == 1:
          # The build system allows for adding just the entry if the dependency
          # is the same name
          dep = split[0]
        else:
          dep = split[1]
        if entry in tree:
          tree[entry].append(dep)
        else:
          tree[entry] = [dep]
  return tree


if __name__ == '__main__':
  parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

  # TODO(rduan): Remove these three flags once J2ObjC compile actions are fully
  # moved to the edges.
  parser.add_argument(
      '--input_files',
      help=('The comma-separated file paths of translated source files to '
            'prune.'))
  parser.add_argument(
      '--output_files',
      help='The comma-separated file paths of pruned source files to write to.')
  parser.add_argument(
      '--objc_file_path',
      help='The file path which represents a directory where the generated ObjC'
      ' files reside')

  parser.add_argument(
      '--input_archive',
      help=('The path of the translated archive to prune.'))
  parser.add_argument(
      '--output_archive',
      help='The path of the pruned archive file to write to.')
  parser.add_argument(
      '--dummy_archive',
      help='The dummy archive file that contains no symbol.')
  parser.add_argument(
      '--dependency_mapping_files',
      help='The comma-separated file paths of dependency mapping files.')
  parser.add_argument(
      '--header_mapping_files',
      help='The comma-separated file paths of header mapping files.')
  parser.add_argument(
      '--archive_source_mapping_files',
      help='The comma-separated file paths of archive to source mapping files.'
           'These mapping files should contain mappings between the '
           'translated source files and the archive file compiled from those '
           'source files.')
  parser.add_argument(
      '--entry_classes',
      help=('The comma-separated list of Java entry classes to be used as entry'
            ' point of the dead code analysis.'))

  args = parser.parse_args()

  if not args.entry_classes:
    raise Exception('J2objC dead code removal is on but no entry class is ',
                    'specified in any j2objc_library targets in the transitive',
                    ' closure')
  if args.input_archive and args.output_archive:
    PruneArchiveFile(
        args.input_archive,
        args.output_archive,
        args.dummy_archive,
        args.dependency_mapping_files,
        args.header_mapping_files,
        args.archive_source_mapping_files,
        args.entry_classes,
    )
  else:
    # TODO(rduan): Remove once J2ObjC compile actions are fully moved to the
    # edges.
    PruneSourceFiles(
        args.input_files,
        args.output_files,
        args.dependency_mapping_files,
        args.header_mapping_files,
        args.entry_classes,
        args.objc_file_path)
