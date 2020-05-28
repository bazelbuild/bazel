// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkMethod;

/**
 * Test case for a StarlarkMethod method which has a parameter with an unsafely specified generic
 * type. (Parameters, if generic, may only have wildcards, as the types of these parameters must be
 * validated dynamically.)
 */
public class SpecifiedGenericType implements StarlarkValue {

  @StarlarkMethod(
      name = "specified_generic_type",
      documented = false,
      parameters = {
        @Param(name = "one", type = Sequence.class, named = true),
      })
  public String specifiedGenericType(Sequence<String> one) {
    return "bar";
  }
}
