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

import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkValue;

/**
 * Test case for a StarlarkMethod which does not have an appropriate StarlarkThread parameter
 * despite having useStarlarkThread set.
 */
public class StarlarkThreadMissing implements StarlarkValue {

  @StarlarkMethod(
      name = "three_arg_method_missing_env",
      documented = false,
      parameters = {
        @Param(name = "one", type = String.class, named = true),
        @Param(name = "two", type = StarlarkInt.class, named = true),
      },
      useStarlarkThread = true)
  public String threeArgMethod(String one, StarlarkInt two, String shouldBeThread) {
    return "bar";
  }
}
