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

package com.google.devtools.build.lib.analysis.skylarkinterface.processor.testsources;

import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;

/**
 * Test source file verifying various proper uses of SkylarkCallable.
 */
public class GoldenCase {

  @SkylarkCallable(
    name = "struct_field_method",
    doc = "",
    structField = true)
  public String structFieldMethod() {
    return "foo";
  }

  @SkylarkCallable(
    name = "zero_arg_method",
    doc = "")
  public Integer zeroArgMethod() {
    return 0;
  }

  @SkylarkCallable(
    name = "three_arg_method",
    doc = "")
  public String threeArgMethod(String one, Integer two, String three) {
    return "bar";
  }

  @SkylarkCallable(
    name = "three_arg_method_with_params",
    doc = "",
    parameters = {
      @Param(name = "one", type = String.class, named = true),
      @Param(name = "two", type = Integer.class, named = true),
      @Param(name = "three", type = String.class, named = true),
    })
  public String threeArgMethodWithParams(String one, Integer two, String three) {
    return "baz";
  }
}
