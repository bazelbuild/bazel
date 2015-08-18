#!/usr/bin/python2.7

# Copyright 2015 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http:#www.apache.org/licenses/LICENSE-2.0
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
from collections import OrderedDict
import multiprocessing
import os
import Queue
import shutil
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
  tree = dict()
  for dependency_mapping_file in dependency_mapping_files.split(','):
    with file_open(dependency_mapping_file, 'r') as f:
      for line in f:
        entry = line.strip().split(':')[0]
        dep = line.strip().split(':')[1]
        if entry in tree:
          tree[entry].append(dep)
        else:
          tree[entry] = [dep]
  return tree


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
  header_mapping = OrderedDict()
  for header_mapping_file in header_mapping_files.split(','):
    with file_open(header_mapping_file, 'r') as f:
      for line in f:
        java_class_name = line.strip().split('=')[0]
        transpiled_file_name = os.path.splitext(line.strip().split('=')[1])[0]
        header_mapping[java_class_name] = transpiled_file_name
  return header_mapping


def BuildReachableFileSet(entry_classes, reachability_tree, header_mapping):
  """Builds a set of reachable translated files from entry Java classes.

  Args:
    entry_classes: A comma separated list of Java entry classes.
    reachability_tree: A dict mapping translated files to their direct
        dependencies.
    header_mapping: A dict mapping Java class names to translated source files.
  Returns:
    A set of reachable translated files from the given list of entry classes.
  Raises:
    Exception: If there is an entry class that is not being transpiled in this
        j2objc_library.
  """
  reachable_files = set()
  for entry_class in entry_classes.split(','):
    if entry_class not in header_mapping:
      raise Exception(entry_class +
                      'is not in the transitive Java deps of included ' +
                      'j2objc_library rules.')
    transpiled_file_name = header_mapping[entry_class]
    reachable_files.add(transpiled_file_name)
    current_level_deps = reachability_tree[transpiled_file_name]
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
  file_queue = Queue.Queue()
  for input_file, output_file in zip(input_files.split(','),
                                     output_files.split(',')):
    file_queue.put((input_file, output_file))

  for _ in xrange(multiprocessing.cpu_count()):
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
    except Queue.Empty:
      return
    file_name = os.path.relpath(os.path.splitext(input_file)[0],
                                objc_file_path)
    # Translated files from package-info.java are also preserved because
    # they are needed to resolve ObjC class names with prefixes.
    if file_name in reachable_files or file_name.endswith('package-info'):
      file_shutil.copy(input_file, output_file)
    else:
      f = file_open(output_file, 'w')
      # Use a static variable scoped to the source file to supress
      # the "has no symbols" linker warning for empty object files.
      f.write(PRUNED_SRC_CONTENT)
      f.close()
    file_queue.task_done()


def PruneDeadCode(input_files, output_files, dependency_mapping_files,
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
  Returns:
    None.
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


if __name__ == '__main__':
  parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
  parser.add_argument(
      '--input_files',
      required=True,
      help=('The comma-separated file paths of translated source files to '
            'prune.'))
  parser.add_argument(
      '--output_files',
      required=True,
      help='The comma-separated file paths of pruned source files to write to.')
  parser.add_argument(
      '--dependency_mapping_files',
      required=True,
      help='The comma-separated file paths of dependency mapping files.')
  parser.add_argument(
      '--header_mapping_files',
      required=True,
      help='The comma-separated file paths of header mapping files.')
  parser.add_argument(
      '--entry_classes',
      required=True,
      help=('The comma-separated list of Java entry classes to be used as entry'
            ' point of the dead code anlysis.'))
  parser.add_argument(
      '--objc_file_path',
      required=True,
      help='The file path which represents a directory where the generated ObjC'
      ' files reside')
  args = parser.parse_args()

  if not args.entry_classes:
    raise Exception('J2objC dead code removal is on but no entry class is ',
                    'specified in any j2objc_library targets in the transitive',
                    ' closure')
  PruneDeadCode(
      args.input_files,
      args.output_files,
      args.dependency_mapping_files,
      args.header_mapping_files,
      args.entry_classes,
      args.objc_file_path)
