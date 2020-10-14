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
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/**
 * Test case for a StarlarkCallable method which specifies StarlarkThread before other parameters.
 */
public class StarlarkInfoBeforeParams implements StarlarkValue {

  @StarlarkMethod(
      name = "skylark_info_wrong_order",
      documented = false,
      parameters = {
        @Param(name = "one", type = String.class, named = true),
        @Param(name = "two", type = StarlarkInt.class, named = true),
        @Param(name = "three", type = String.class, named = true)
      },
      useStarlarkThread = true)
  public String threeArgMethod(StarlarkThread thread, String one, StarlarkInt two, String three) {
    return "bar";
  }
}
