// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.rules.cpp;

/**
 * This class represents the different linking modes.
 */
public enum LinkingMode {

  /**
   * Everything is linked statically; e.g. {@code gcc -static x.o libfoo.a
   * libbar.a -lm}. Specified by {@code -static} in linkopts.
   */
  FULLY_STATIC,

  /**
   * Link binaries statically except for system libraries
   * e.g. {@code gcc x.o libfoo.a libbar.a -lm}. Specified by {@code linkstatic=1}.
   *
   * <p>This mode applies to executables.
   */
  MOSTLY_STATIC,

  /**
   * Same as MOSTLY_STATIC, but for shared libraries.
   */
  MOSTLY_STATIC_LIBRARIES,

  /**
   * All libraries are linked dynamically (if a dynamic version is available),
   * e.g. {@code gcc x.o libfoo.so libbar.so -lm}. Specified by {@code
   * linkstatic=0}.
   */
  DYNAMIC;
}
