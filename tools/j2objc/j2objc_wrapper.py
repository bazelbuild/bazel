#!/usr/bin/python2.7

# Copyright 2015 The Bazel Authors. All rights reserved.
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
import tempfile
import threading

INCLUDE_RE = re.compile('#(include|import) "([^"]+)"')


def RunJ2ObjC(java, jvm_flags, j2objc, main_class, j2objc_args):
  """Runs J2ObjC transpiler to translate Java source files to ObjC.

  Args:
    java: The path of the Java executable.
    jvm_flags: A comma-separated list of flags to pass to JVM.
    j2objc: The deploy jar of J2ObjC.
    main_class: The J2ObjC main class to invoke.
    j2objc_args: A list of args to pass to J2ObjC transpiler.
  Returns:
    None.
  """
  source_files, flags = _ParseArgs(j2objc_args)
  source_file_manifest_content = ' '.join(source_files)
  fd = None
  param_filename = None
  try:
    fd, param_filename = tempfile.mkstemp(text=True)
    os.write(fd, source_file_manifest_content)
  finally:
    if fd:
      os.close(fd)
  try:
    j2objc_cmd = [java]
    j2objc_cmd.extend(filter(None, jvm_flags.split(',')))
    j2objc_cmd.extend(['-cp', j2objc, main_class])
    j2objc_cmd.extend(flags)
    j2objc_cmd.extend(['@%s' % param_filename])
    subprocess.check_call(j2objc_cmd, stderr=subprocess.STDOUT)
  finally:
    if param_filename:
      os.remove(param_filename)


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
  Raises:
    RuntimeError: If spawned threads throw errors during processing.
  Returns:
    None.
  """
  dep_mapping = dict()
  input_file_queue = Queue.Queue()
  output_dep_mapping_queue = Queue.Queue()
  error_message_queue = Queue.Queue()
  for output_file in translated_source_files.split(','):
    input_file_queue.put(output_file)

  for _ in xrange(multiprocessing.cpu_count()):
    t = threading.Thread(target=_ReadDepMapping, args=(input_file_queue,
                                                       output_dep_mapping_queue,
                                                       error_message_queue,
                                                       objc_file_path,
                                                       file_open))
    t.start()

  input_file_queue.join()

  if not error_message_queue.empty():
    error_messages = [error_message for error_message in
                      error_message_queue.queue]
    raise RuntimeError('\n'.join(error_messages))

  while not output_dep_mapping_queue.empty():
    entry_file, deps = output_dep_mapping_queue.get()
    dep_mapping[entry_file] = deps

  f = file_open(output_dependency_mapping_file, 'w')
  for entry in sorted(dep_mapping):
    for dep in dep_mapping[entry]:
      f.write(entry + ':' + dep + '\n')
  f.close()


def _ReadDepMapping(input_file_queue, output_dep_mapping_queue,
                    error_message_queue, objc_file_path, file_open=open):
  while True:
    try:
      input_file = input_file_queue.get_nowait()
    except Queue.Empty:
      # No more work left in the queue.
      return

    try:
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
    except Exception as e:  # pylint: disable=broad-except
      error_message_queue.put(str(e))
    finally:
      # We need to mark the task done to prevent blocking the main process
      # indefinitely.
      input_file_queue.task_done()


def WriteArchiveSourceMappingFile(compiled_archive_file_path,
                                  output_archive_source_mapping_file,
                                  translated_source_files,
                                  objc_file_path,
                                  file_open=open):
  """Writes a mapping file between archive file to associated ObjC source files.

  Args:
    compiled_archive_file_path: The path of the archive file.
    output_archive_source_mapping_file: A path of the mapping file to write to.
    translated_source_files: A comma-separated list of source files translated
        by J2ObjC.
    objc_file_path: The file path which represents a directory where the
        generated ObjC files reside.
    file_open: Reference to the builtin open function so it may be
        overridden for testing.
  Returns:
    None.
  """
  with file_open(output_archive_source_mapping_file, 'w') as f:
    for translated_source_file in translated_source_files.split(','):
      file_path = os.path.relpath(translated_source_file, objc_file_path)
      f.write(compiled_archive_file_path + ':' + file_path + '\n')


def _ParseArgs(j2objc_args):
  """Separate arguments passed to J2ObjC into source files and J2ObjC flags.

  Args:
    j2objc_args: A list of args to pass to J2ObjC transpiler.
  Returns:
    A tuple containing source files and J2ObjC flags
  """
  source_files = []
  flags = []
  is_next_flag_value = False
  for j2objc_arg in j2objc_args:
    if j2objc_arg.startswith('-'):
      flags.append(j2objc_arg)
      is_next_flag_value = True
    elif is_next_flag_value:
      flags.append(j2objc_arg)
      is_next_flag_value = False
    else:
      source_files.append(j2objc_arg)
  return (source_files, flags)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
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
  parser.add_argument(
      '--output_archive_source_mapping_file',
      help='The file path of the mapping file containing mappings between the '
           'translated source files and the to-be-generated archive file '
           'compiled from those source files. --compile_archive_file_path must '
           'be specified if this option is specified.')
  parser.add_argument(
      '--compiled_archive_file_path',
      required=False,
      help=('The archive file path that will be produced by ObjC compile action'
            ' later'))
  args, pass_through_args = parser.parse_known_args()

  RunJ2ObjC(args.java,
            args.jvm_flags,
            args.j2objc,
            args.main_class,
            pass_through_args)
  WriteDepMappingFile(args.translated_source_files,
                      args.objc_file_path,
                      args.output_dependency_mapping_file)

  if args.output_archive_source_mapping_file:
    WriteArchiveSourceMappingFile(args.compiled_archive_file_path,
                                  args.output_archive_source_mapping_file,
                                  args.translated_source_files,
                                  args.objc_file_path)
