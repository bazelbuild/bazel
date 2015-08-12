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

"""A wrapper script for J2ObjC transpiler.

This script wraps around J2ObjC transpiler to also output a dependency mapping
file by scanning the import and include directives of the J2ObjC-translated
files.
"""

import argparse
import multiprocessing
import os
import Queue
import re
import subprocess
import threading

INCLUDE_RE = re.compile('#(include|import) "([^"]+)"')


def RunJ2ObjC(java, jvm_flags, j2objc, main_class, flags):
  """Runs J2ObjC transpiler to translate Java source files to ObjC.

  Args:
    java: The path of the Java executable.
    jvm_flags: A comma-separated list of flags to pass to JVM.
    j2objc: The deploy jar of J2ObjC.
    main_class: The J2ObjC main class to invoke.
    flags: A list of flags to pass to J2ObjC transpiler.
  Returns:
    None.
  """
  j2objc_args = [java]
  j2objc_args.extend(filter(None, jvm_flags.split(',')))
  j2objc_args.extend(['-cp', j2objc, main_class])
  j2objc_args.extend(flags)

  subprocess.check_call(j2objc_args, stderr=subprocess.STDOUT)


def WriteDepMappingFile(translated_source_files,
                        objc_file_path,
                        output_dependency_mapping_file,
                        file_open=open):
  """Scans J2ObjC-translated files and outputs a dependency mapping file.

  The mapping file contains mappings between translated source files and their
  imported source files scanned from the import and include directives.

  Args:
    translated_source_files: A comma-separated list of files translated by
        J2ObjC.
    objc_file_path: The file path which represents a directory where the
        generated ObjC files reside.
    output_dependency_mapping_file: The path of the dependency mapping file to
        write to.
    file_open: Reference to the builtin open function so it may be
        overridden for testing.
  Returns:
    None.
  """
  dep_mapping = dict()
  input_file_queue = Queue.Queue()
  output_dep_mapping_queue = Queue.Queue()
  for output_file in translated_source_files.split(','):
    input_file_queue.put(output_file)

  for _ in xrange(multiprocessing.cpu_count()):
    t = threading.Thread(target=_ReadDepMapping, args=(input_file_queue,
                                                       output_dep_mapping_queue,
                                                       objc_file_path,
                                                       file_open))
    t.start()

  input_file_queue.join()

  while not output_dep_mapping_queue.empty():
    entry_file, deps = output_dep_mapping_queue.get()
    dep_mapping[entry_file] = deps

  f = file_open(output_dependency_mapping_file, 'w')
  for entry in sorted(dep_mapping):
    for dep in dep_mapping[entry]:
      f.write(entry + ':' + dep + '\n')
  f.close()


def _ReadDepMapping(input_file_queue, output_dep_mapping_queue, objc_file_path,
                    file_open=open):
  while True:
    try:
      input_file = input_file_queue.get_nowait()
    except Queue.Empty:
      return
    deps = []
    entry = os.path.relpath(os.path.splitext(input_file)[0], objc_file_path)
    with file_open(input_file, 'r') as f:
      for line in f:
        include = INCLUDE_RE.match(line)
        if include:
          include_path = include.group(2)
          dep = os.path.splitext(include_path)[0]
          if dep != entry:
            deps.append(dep)
    output_dep_mapping_queue.put((entry, deps))
    input_file_queue.task_done()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--java',
      required=True,
      help='The path to the Java executable.')
  parser.add_argument(
      '--jvm_flags',
      default='',
      help='A comma-separated list of flags to pass to the JVM.')
  parser.add_argument(
      '--j2objc',
      required=True,
      help='The path to the J2ObjC deploy jar.')
  parser.add_argument(
      '--main_class',
      required=True,
      help='The main class of the J2ObjC deploy jar to execute.')
  parser.add_argument(
      '--translated_source_files',
      required=True,
      help=('A comma-separated list of file paths where J2ObjC will write the '
            'translated files to.'))
  parser.add_argument(
      '--output_dependency_mapping_file',
      required=True,
      help='The file path of the dependency mapping file to write to.')
  parser.add_argument(
      '--objc_file_path',
      required=True,
      help=('The file path which represents a directory where the generated '
            'ObjC files reside.'))
  args, pass_through_flags = parser.parse_known_args()

  RunJ2ObjC(args.java,
            args.jvm_flags,
            args.j2objc,
            args.main_class,
            pass_through_flags)
  WriteDepMappingFile(args.translated_source_files,
                      args.objc_file_path,
                      args.output_dependency_mapping_file)
