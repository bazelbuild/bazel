// Copyright 2018 The Bazel Authors. All rights reserved.
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

package net.starlark.java.annot.processor.testsources;

import com.google.devtools.build.lib.syntax.StarlarkValue;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkMethod;

/**
 * Test case for a StarlarkMethod method which has a parameter with both type and allowedTypes
 * specified.
 */
public class ParamTypeConflict implements StarlarkValue {

  @StarlarkMethod(
      name = "param_type_conflict",
      documented = false,
      parameters = {
        @Param(
            name = "a_parameter",
            type = String.class,
            named = true,
            allowedTypes = {
              @ParamType(type = String.class),
            })
      })
  public Integer paramTypeConflict(String x) {
    return 42;
  }
}
