# Copyright 2016 The Bazel Authors. All rights reserved.
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

"""Script to extract resources from a jar and put them into a separate zip file.

The input jar will be opened as a zip file, and its entries will be copied into
a new zip file, excepting any which are suspected not to be resources
 (e.g., Java class files, etc).

Usage:
  python resource_extractor.py <input jar> <output zip>
"""

from __future__ import print_function

import sys
import zipfile

USAGE = """Error, invalid arguments.
Usage: resource_extractor.py <input jar> <output zip>"""

EXCLUDED_EXTENSIONS = (
    '.aidl',  # Android interface definition files
    '.rs',  # RenderScript files
    '.fs',  # FilterScript files
    '.rsh',  # RenderScript header files
    '.d',  # Dependency files
    '.java',  # Java source files
    '.scala',  # Scala source files
    '.class',  # Java class files
    '.scc',  # Visual SourceSafe
    '.swp',  # vi swap file
    '.gwt.xml',  # Google Web Toolkit modules
    '~',  # backup files
    '/',  # empty directory entries
)

EXCLUDED_FILENAMES = (
    'thumbs.db',  # image index file
    'picasa.ini',  # image index file
    'package.html',  # Javadoc
    'overview.html',  # Javadoc
    'protobuf.meta',  # protocol buffer metadata
    'flags.xml',  # Google flags metadata
)

EXCLUDED_DIRECTORIES = (
    'cvs',  # CVS repository files
    '.svn',  # SVN repository files
    'sccs',  # SourceSafe repository files
    'meta-inf'  # jar metadata
)


def IsValidPath(path):
  """Checks if the provided path describes a resource.

  Args:
    path: the path to check

  Returns:
    True if the path is a resource.
  """
  path = path.lower()
  if any(path.endswith(extension) for extension in EXCLUDED_EXTENSIONS):
    return False

  segments = path.split('/')
  filename = segments[-1]
  if filename.startswith('.') or filename in EXCLUDED_FILENAMES:
    return False

  dirs = segments[:-1]
  # allow META-INF/services at the root to support ServiceLoader
  if dirs[:2] == ['meta-inf', 'services']:
    return True

  return not any(dir in EXCLUDED_DIRECTORIES for dir in dirs)


def ExtractResources(input_jar, output_zip):
  for path in input_jar.namelist():
    if IsValidPath(path):
      output_zip.writestr(input_jar.getinfo(path), input_jar.read(path))


def main(argv):
  if len(argv) != 3:
    print(USAGE)
    sys.exit(1)
  with zipfile.ZipFile(argv[1], 'r') as input_jar:
    with zipfile.ZipFile(argv[2], 'w') as output_zip:
      ExtractResources(input_jar, output_zip)


if __name__ == '__main__':
  main(sys.argv)
