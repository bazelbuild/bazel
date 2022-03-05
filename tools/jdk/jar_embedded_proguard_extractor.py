# Copyright 2022 The Bazel Authors. All rights reserved.
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

"""A tool for extracting proguard specs from JARs.

Usage:
Takes positional arguments: input_jar1 input_jar2 <...> output_proguard_spec

Background: 
A JAR may contain proguard specs under /META-INF/proguard/
[https://developer.android.com/studio/build/shrink-code]

This tool extracts them all into a single spec for easy usage."""


import sys
import zipfile


if __name__ == "__main__":
    with open(sys.argv[-1], 'wb') as output_proguard_spec:
        for jar_path in sys.argv[1:-1]:
            with zipfile.ZipFile(jar_path) as jar:
                for entry in jar.namelist():
                    if entry.startswith('META-INF/proguard'):
                        # zip directories are empty and therefore ok to output
                        output_proguard_spec.write(jar.read(entry))
                        # gracefully handle any lack of trailing newline
                        output_proguard_spec.write(b'\n')
