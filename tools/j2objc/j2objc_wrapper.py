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
import errno
import multiprocessing
import os
import Queue
import re
import shutil
import subprocess
import tempfile
import threading
import zipfile

_INCLUDE_RE = re.compile('#(include|import) "([^"]+)"')
_CONST_DATE_TIME = [1980, 1, 1, 0, 0, 0]


def RunJ2ObjC(java, jvm_flags, j2objc, main_class, output_file_path,
              j2objc_args, source_paths, files_to_translate):
  """Runs J2ObjC transpiler to translate Java source files to ObjC.

  Args:
    java: The path of the Java executable.
    jvm_flags: A comma-separated list of flags to pass to JVM.
    j2objc: The deploy jar of J2ObjC.
    main_class: The J2ObjC main class to invoke.
    output_file_path: The output file directory.
    j2objc_args: A list of args to pass to J2ObjC transpiler.
    source_paths: A list of directories that contain sources to translate.
    files_to_translate: A list of relative paths (relative to source_paths) that
        point to sources to translate.
  Returns:
    None.
  """
  j2objc_args.extend(['-sourcepath', ':'.join(source_paths)])
  j2objc_args.extend(['-d', output_file_path])
  j2objc_args.extend(files_to_translate)
  param_file_content = ' '.join(j2objc_args)
  fd = None
  param_filename = None
  try:
    fd, param_filename = tempfile.mkstemp(text=True)
    os.write(fd, param_file_content)
  finally:
    if fd:
      os.close(fd)
  try:
    j2objc_cmd = [java]
    j2objc_cmd.extend(filter(None, jvm_flags.split(',')))
    j2objc_cmd.extend(['-cp', j2objc, main_class])
    j2objc_cmd.extend(['@%s' % param_filename])
    subprocess.check_call(j2objc_cmd, stderr=subprocess.STDOUT)
  finally:
    if param_filename:
      os.remove(param_filename)


def WriteDepMappingFile(objc_files,
                        objc_file_root,
                        output_dependency_mapping_file,
                        file_open=open):
  """Scans J2ObjC-translated files and outputs a dependency mapping file.

  The mapping file contains mappings between translated source files and their
  imported source files scanned from the import and include directives.

  Args:
    objc_files: A list of ObjC files translated by J2ObjC.
    objc_file_root: The file path which represents a directory where the
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
  for objc_file in objc_files:
    input_file_queue.put(os.path.join(objc_file_root, objc_file))

  for _ in xrange(multiprocessing.cpu_count()):
    t = threading.Thread(target=_ReadDepMapping, args=(input_file_queue,
                                                       output_dep_mapping_queue,
                                                       error_message_queue,
                                                       objc_file_root,
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

  with file_open(output_dependency_mapping_file, 'w') as f:
    for entry in sorted(dep_mapping):
      for dep in dep_mapping[entry]:
        f.write(entry + ':' + dep + '\n')


def _ReadDepMapping(input_file_queue, output_dep_mapping_queue,
                    error_message_queue, output_root, file_open=open):
  while True:
    try:
      input_file = input_file_queue.get_nowait()
    except Queue.Empty:
      # No more work left in the queue.
      return

    try:
      deps = set()
      input_file_name = os.path.splitext(input_file)[0]
      entry = os.path.relpath(input_file_name, output_root)
      for file_ext in ['.m', '.h']:
        with file_open(input_file_name + file_ext, 'r') as f:
          for line in f:
            include = _INCLUDE_RE.match(line)
            if include:
              include_path = include.group(2)
              dep = os.path.splitext(include_path)[0]
              if dep != entry:
                deps.add(dep)

      output_dep_mapping_queue.put((entry, sorted(deps)))
    except Exception as e:  # pylint: disable=broad-except
      error_message_queue.put(str(e))
    finally:
      # We need to mark the task done to prevent blocking the main process
      # indefinitely.
      input_file_queue.task_done()


def WriteArchiveSourceMappingFile(compiled_archive_file_path,
                                  output_archive_source_mapping_file,
                                  objc_files,
                                  file_open=open):
  """Writes a mapping file between archive file to associated ObjC source files.

  Args:
    compiled_archive_file_path: The path of the archive file.
    output_archive_source_mapping_file: A path of the mapping file to write to.
    objc_files: A list of ObjC files translated by J2ObjC.
    file_open: Reference to the builtin open function so it may be
        overridden for testing.
  Returns:
    None.
  """
  with file_open(output_archive_source_mapping_file, 'w') as f:
    for objc_file in objc_files:
      f.write(compiled_archive_file_path + ':' + objc_file + '\n')


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


def _J2ObjcOutputObjcFiles(java_files):
  """Returns the relative paths of the associated output ObjC source files.

  Args:
    java_files: The list of Java files to translate.
  Returns:
    A list of associated output ObjC source files.
  """
  return [os.path.splitext(java_file)[0] + '.m' for java_file in java_files]


def UnzipSourceJarSources(source_jars):
  """Unzips the source jars containing Java source files.

  Args:
    source_jars: The list of input Java source jars.
  Returns:
    A tuple of the temporary output root and a list of root-relative paths of
    unzipped Java files
  """
  srcjar_java_files = []
  if source_jars:
    tmp_input_root = tempfile.mkdtemp()
    for source_jar in source_jars:
      zip_ref = zipfile.ZipFile(source_jar, 'r')
      zip_entries = []

      for file_entry in zip_ref.namelist():
        # We only care about Java source files.
        if file_entry.endswith('.java'):
          zip_entries.append(file_entry)

      zip_ref.extractall(tmp_input_root, zip_entries)
      zip_ref.close()
      srcjar_java_files.extend(zip_entries)

    return (tmp_input_root, srcjar_java_files)
  else:
    return None


def RenameGenJarObjcFileRootInFileContent(tmp_objc_file_root,
                                          j2objc_source_paths,
                                          gen_src_jar, genjar_objc_files,
                                          execute=subprocess.check_call):
  """Renames references to temporary root inside ObjC sources from gen srcjar.

  Args:
    tmp_objc_file_root: The temporary output root containing ObjC sources.
    j2objc_source_paths: The source paths used by J2ObjC.
    gen_src_jar: The path of the gen srcjar.
    genjar_objc_files: The list of ObjC sources translated from the gen srcjar.
    execute: The function used to execute shell commands.
  Returns:
    None.
  """
  if genjar_objc_files:
    abs_genjar_objc_source_files = [
        os.path.join(tmp_objc_file_root, genjar_objc_file)
        for genjar_objc_file in genjar_objc_files
    ]
    abs_genjar_objc_header_files = [
        os.path.join(tmp_objc_file_root,
                     os.path.splitext(genjar_objc_file)[0] + '.h')
        for genjar_objc_file in genjar_objc_files
    ]

    # We execute a command to change all references of the temporary Java root
    # where we unzipped the gen srcjar sources, to the actual gen srcjar that
    # contains the original Java sources.
    cmd = [
        'sed',
        '-i',
        '-e',
        's|%s/|%s::|g' % (j2objc_source_paths[1], gen_src_jar)
    ]
    cmd.extend(abs_genjar_objc_source_files)
    cmd.extend(abs_genjar_objc_header_files)
    execute(cmd, stderr=subprocess.STDOUT)


def MoveObjcFileToFinalOutputRoot(objc_files,
                                  tmp_objc_file_root,
                                  final_objc_file_root,
                                  suffix,
                                  os_module=os,
                                  shutil_module=shutil):
  """Moves ObjC files from temporary location to the final output location.

  Args:
    objc_files: The list of objc files to move.
    tmp_objc_file_root: The temporary output root containing ObjC sources.
    final_objc_file_root: The final output root.
    suffix: The suffix of the files to move.
    os_module: The os python module.
    shutil_module: The shutil python module.
  Returns:
    None.
  """
  for objc_file in objc_files:
    file_with_suffix = os_module.path.splitext(objc_file)[0] + suffix
    dest_path = os_module.path.join(
        final_objc_file_root, file_with_suffix)
    dest_path_dir = os_module.path.dirname(dest_path)

    if not os_module.path.isdir(dest_path_dir):
      try:
        os_module.makedirs(dest_path_dir)
      except OSError as e:
        if e.errno != errno.EEXIST or not os_module.path.isdir(dest_path_dir):
          raise

    shutil_module.move(
        os_module.path.join(tmp_objc_file_root, file_with_suffix),
        dest_path)


def PostJ2ObjcFileProcessing(normal_objc_files, genjar_objc_files,
                             tmp_objc_file_root, final_objc_file_root,
                             j2objc_source_paths, gen_src_jar,
                             output_gen_source_dir, output_gen_header_dir):
  """Performs cleanups on ObjC files and moves them to final output location.

  Args:
    normal_objc_files: The list of objc files translated from normal Java files.
    genjar_objc_files: The list of ObjC sources translated from the gen srcjar.
    tmp_objc_file_root: The temporary output root containing ObjC sources.
    final_objc_file_root: The final output root.
    j2objc_source_paths: The source paths used by J2ObjC.
    gen_src_jar: The path of the gen srcjar.
    output_gen_source_dir: The final output directory of ObjC source files
        translated from gen srcjar. Maybe null.
    output_gen_header_dir: The final output directory of ObjC header files
        translated from gen srcjar. Maybe null.
  Returns:
    None.
  """
  RenameGenJarObjcFileRootInFileContent(tmp_objc_file_root,
                                        j2objc_source_paths,
                                        gen_src_jar,
                                        genjar_objc_files)
  MoveObjcFileToFinalOutputRoot(normal_objc_files,
                                tmp_objc_file_root,
                                final_objc_file_root,
                                '.m')
  MoveObjcFileToFinalOutputRoot(normal_objc_files,
                                tmp_objc_file_root,
                                final_objc_file_root,
                                '.h')

  if output_gen_source_dir:
    MoveObjcFileToFinalOutputRoot(
        genjar_objc_files,
        tmp_objc_file_root,
        output_gen_source_dir,
        '.m')

  if output_gen_header_dir:
    MoveObjcFileToFinalOutputRoot(
        genjar_objc_files,
        tmp_objc_file_root,
        output_gen_header_dir,
        '.h')


def GenerateJ2objcMappingFiles(normal_objc_files,
                               genjar_objc_files,
                               tmp_objc_file_root,
                               output_dependency_mapping_file,
                               output_archive_source_mapping_file,
                               compiled_archive_file_path):
  """Generates J2ObjC mapping files.

  Args:
    normal_objc_files: The list of objc files translated from normal Java files.
    genjar_objc_files: The list of ObjC sources translated from the gen srcjar.
    tmp_objc_file_root: The temporary output root containing ObjC sources.
    output_dependency_mapping_file: The path of the dependency mapping file to
        write to.
    output_archive_source_mapping_file: A path of the mapping file to write to.
    compiled_archive_file_path: The path of the archive file.
  Returns:
    None.
  """
  WriteDepMappingFile(normal_objc_files + genjar_objc_files,
                      tmp_objc_file_root,
                      output_dependency_mapping_file)

  if output_archive_source_mapping_file:
    WriteArchiveSourceMappingFile(compiled_archive_file_path,
                                  output_archive_source_mapping_file,
                                  normal_objc_files + genjar_objc_files)


def main():
  parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
  parser.add_argument(
      '--java',
      required=True,
      help='The path to the Java executable.')
  parser.add_argument(
      '--jvm_flags',
      default='-Xss4m,-XX:+UseParallelGC',
      help='A comma-separated list of flags to pass to the JVM.')
  parser.add_argument(
      '--j2objc',
      required=True,
      help='The path to the J2ObjC deploy jar.')
  parser.add_argument(
      '--main_class',
      required=True,
      help='The main class of the J2ObjC deploy jar to execute.')
  # TODO(rduan): Remove, no longer needed.
  parser.add_argument(
      '--translated_source_files',
      required=False,
      help=('A comma-separated list of file paths where J2ObjC will write the '
            'translated files to.'))
  parser.add_argument(
      '--output_dependency_mapping_file',
      required=True,
      help='The file path of the dependency mapping file to write to.')
  parser.add_argument(
      '--objc_file_path', '-d',
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
  # TODO(rduan): Remove this flag once it is fully replaced by flag --src_jars.
  parser.add_argument(
      '--gen_src_jar',
      required=False,
      help='The jar containing Java sources generated by annotation processor.')
  parser.add_argument(
      '--src_jars',
      required=False,
      help='The list of Java source jars containing Java sources to translate.')
  parser.add_argument(
      '--output_gen_source_dir',
      required=False,
      help='The output directory of ObjC source files translated from the gen'
           ' srcjar')
  parser.add_argument(
      '--output_gen_header_dir',
      required=False,
      help='The output directory of ObjC header files translated from the gen'
           ' srcjar')

  args, pass_through_args = parser.parse_known_args()
  normal_java_files, j2objc_flags = _ParseArgs(pass_through_args)
  srcjar_java_files = []
  j2objc_source_paths = [os.getcwd()]

  # Unzip the source jars, so J2ObjC can translate the contained sources.
  # Also add the temporary directory containing the unzipped sources as a source
  # path for J2ObjC, so it can find these sources.
  source_jars = []
  if args.gen_src_jar:
    source_jars.append(args.gen_src_jar)
  if args.src_jars:
    source_jars.extend(args.src_jars.split(','))

  srcjar_source_tuple = UnzipSourceJarSources(source_jars)
  if srcjar_source_tuple:
    j2objc_source_paths.append(srcjar_source_tuple[0])
    srcjar_java_files = srcjar_source_tuple[1]

  # Run J2ObjC over the normal input Java files and unzipped gen jar Java files.
  # The output is stored in a temporary directory.
  tmp_objc_file_root = tempfile.mkdtemp()

  # If we do not generate the header mapping from J2ObjC, we still
  # need to specify --output-header-mapping, as it signals to J2ObjC that we
  # are using source paths as import paths, not package paths.
  # TODO(rduan): Make another flag in J2ObjC to specify using source paths.
  if '--output-header-mapping' not in j2objc_flags:
    j2objc_flags.extend(['--output-header-mapping', '/dev/null'])

  RunJ2ObjC(args.java,
            args.jvm_flags,
            args.j2objc,
            args.main_class,
            tmp_objc_file_root,
            j2objc_flags,
            j2objc_source_paths,
            normal_java_files + srcjar_java_files)

  # Calculate the relative paths of generated objc files.
  normal_objc_files = _J2ObjcOutputObjcFiles(normal_java_files)
  genjar_objc_files = _J2ObjcOutputObjcFiles(srcjar_java_files)

  # Generate J2ObjC mapping files needed for distributed builds.
  GenerateJ2objcMappingFiles(normal_objc_files,
                             genjar_objc_files,
                             tmp_objc_file_root,
                             args.output_dependency_mapping_file,
                             args.output_archive_source_mapping_file,
                             args.compiled_archive_file_path)

  # Post J2ObjC-run processing, involving file editing, zipping and moving
  # files to their final output locations.
  PostJ2ObjcFileProcessing(
      normal_objc_files,
      genjar_objc_files,
      tmp_objc_file_root,
      args.objc_file_path,
      j2objc_source_paths,
      args.gen_src_jar,
      args.output_gen_source_dir,
      args.output_gen_header_dir)

if __name__ == '__main__':
  main()
