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

/**
 * Test case for a SkylarkCallable method which has a positional parameter specified after a
 * non positional parameter.
 */
public class PositionalParamAfterNonPositional {

  @SkylarkCallable(
      name = "positional_after_non_positional",
      documented = false,
      parameters = {
          @Param(name = "one",
              named = true,
              positional = false),
          @Param(name = "two",
              named = true,
              positional = true)
      })
  public Integer positionalAfterNonPositional(Integer one, Integer two) {
    return 42;
  }
}