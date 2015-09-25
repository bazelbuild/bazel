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
package com.google.devtools.build.lib.rules.python;

import java.util.Arrays;

/**
 * Python version for Python rules.
 */
public enum PythonVersion {
  PY2,
  PY3,
  PY2AND3,
  PY2ONLY,
  PY3ONLY;

  static final PythonVersion[] ALL_VALUES =
      new PythonVersion[] { PY2, PY3, PY2AND3, PY2ONLY, PY3ONLY };

  public static PythonVersion defaultValue() {
    return PY2;
  }

  public static PythonVersion[] getAllValues() {
    return ALL_VALUES;
  }

  /**
   * Converts the string to PythonVersion, if it is one of the allowed values.
   * Returns null if the input is not valid.
   */
  public static PythonVersion parse(String str, PythonVersion... allowed) {
    if (str == null) {
      return null;
    }
    try {
      PythonVersion version = PythonVersion.valueOf(str);
      if (Arrays.asList(allowed).contains(version)) {
        return version;
      }
      return null;
    } catch (IllegalArgumentException e) {
      return null;
    }
  }
}

