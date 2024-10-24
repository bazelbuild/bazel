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

import com.google.devtools.build.lib.packages.Attribute.AllowedValueSet;
import com.google.devtools.build.lib.util.FileType;

/** Rule definitions for Python rules. */
public class PyRuleClasses {

  public static final FileType PYTHON_SOURCE = FileType.of(".py", ".py3");

  /**
   * A value set of the target and sentinel values that doesn't mention the sentinel in error
   * messages.
   */
  public static final AllowedValueSet TARGET_PYTHON_ATTR_VALUE_SET =
      new AllowedValueSet(PythonVersion.TARGET_AND_SENTINEL_STRINGS) {
        @Override
        public String getErrorReason(Object value) {
          return String.format("has to be one of 'PY2' or 'PY3' instead of '%s'", value);
        }
      };
}
