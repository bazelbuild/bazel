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

package com.google.devtools.build.lib.starlarkbuildapi.test;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;

/**
 * This provider can be implemented by rules which need special environments to run in (especially
 * tests).
 */
@StarlarkBuiltin(
    name = "ExecutionInfo",
    doc = "Use this provider to specify special environment requirements needed to run tests.",
    documented = true,
    category = DocCategory.PROVIDER)
public interface ExecutionInfoApi extends StructApi {

  @StarlarkMethod(
      name = "requirements",
      doc = "A dict indicating special execution requirements, such as hardware platforms.",
      structField = true)
  ImmutableMap<String, String> getExecutionInfo();

  @StarlarkMethod(
      name = "exec_group",
      doc = "The name of the exec group that is used to execute the test.",
      structField = true)
  String getExecGroup();

  /** Provider for {@link ExecutionInfoApi}. */
  @StarlarkBuiltin(
      name = "Provider",
      // Documented by the outer ExecutionInfoApi class and its constructor doc.
      documented = false)
  interface ExecutionInfoApiProvider extends ProviderApi {

    @StarlarkMethod(
        name = "ExecutionInfo",
        doc = "Creates an instance.",
        documented = true,
        parameters = {
          @Param(
              name = "requirements",
              defaultValue = "{}",
              allowedTypes = {@ParamType(type = Dict.class)},
              named = true,
              positional = true,
              doc =
                  "A dict indicating special execution requirements, such as hardware platforms."),
          @Param(
              name = "exec_group",
              allowedTypes = {@ParamType(type = String.class)},
              defaultValue = "'test'",
              named = true,
              positional = true,
              doc = "The name of the exec group that is used to execute the test.")
        },
        selfCall = true)
    @StarlarkConstructor
    ExecutionInfoApi constructor(Dict<?, ?> requirements, String execGroup) throws EvalException;
  }
}
