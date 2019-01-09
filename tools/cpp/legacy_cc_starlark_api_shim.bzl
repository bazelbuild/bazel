# pylint: disable=g-bad-file-header
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

"""
Utility functions that replace old C++ Starlark API using the new API.

See migration instructions in https://github.com/bazelbuild/bazel/issues/7036

Replacements:
dep.cc.transitive_headers -> dep[CcInfo].compilation_context.headers
dep.cc.defines -> dep[CcInfo].compilation_context.defines.to_list()
dep.cc.system_include_directories -> dep[CcInfo].compilation_context.system_includes.to_list()
dep.cc.include_directories -> dep[CcInfo].compilation_context.includes.to_list()
dep.cc.quote_include_directories -> dep[CcInfo].compilation_context.quote_includes.to_list()
dep.cc.link_flags = dep[CcInfo].linking_context.user_link_flags
dep.cc.libs = get_libs_for_static_executable(dep)
dep.cc.compile_flags = get_compile_flags(dep)
"""

def get_libs_for_static_executable(dep):
    """
    Finds the libraries used for linking an executable statically.

    This replaces the old API dep.cc.libs

    Args:
      dep: Target

    Returns:
      A list of File instances, these are the libraries used for linking.
    """
    libraries_to_link = dep[CcInfo].linking_context.libraries_to_link
    libs = []
    for library_to_link in libraries_to_link:
        if library_to_link.static_library != None:
            libs.append(library_to_link.static_library)
        elif library_to_link.pic_static_library != None:
            libs.append(library_to_link.pic_static_library)
        elif library_to_link.interface_library != None:
            libs.append(library_to_link.interface_library)
        elif library_to_link.dynamic_library != None:
            libs.append(library_to_link.dynamic_library)
    return depset(libs)

def get_compile_flags(dep):
    """
    Builds compilation flags. This replaces the old API dep.cc.compile_flags

    This is not the command line that C++ rules will use. For that the toolchain API should be
    used (feature configuration and variables).

    Args:
      dep: Target

    Returns:
      A list of strings
    """
    options = []
    compilation_context = dep[CcInfo].compilation_context
    for define in compilation_context.defines.to_list():
        options.append("-D{}".format(define))

    for system_include in compilation_context.system_includes.to_list():
        if len(system_include) == 0:
            system_include = "."
        options.append("-isystem {}".format(system_include))

    for include in compilation_context.includes.to_list():
        if len(include) == 0:
            include = "."
        options.append("-I {}".format(include))

    for quote_include in compilation_context.quote_includes.to_list():
        if len(quote_include) == 0:
            quote_include = "."
        options.append("-iquote {}".format(quote_include))

    return options
