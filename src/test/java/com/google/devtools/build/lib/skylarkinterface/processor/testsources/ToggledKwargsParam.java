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

package com.google.devtools.build.lib.skylarkinterface.processor.testsources;

import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;

/**
 * Test case for a SkylarkCallable method which has a "extraKeywords" parameter which has
 * enableOnlyWithFlag set. (This is unsupported.)
 */
public class ToggledKwargsParam {

  @SkylarkCallable(
      name = "toggled_kwargs_method",
      documented = false,
      parameters = {
        @Param(name = "one", type = String.class, named = true),
        @Param(name = "two", type = Integer.class, named = true),
      },
      extraPositionals = @Param(name = "args"),
      extraKeywords =
          @Param(
              name = "kwargs",
              enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_BUILD_SETTING_API),
      useEnvironment = true)
  public String toggledKwargsMethod(
      String one,
      Integer two,
      SkylarkList<?> args,
      SkylarkDict<?, ?> kwargs,
      Environment environment) {
    return "cat";
  }
}
