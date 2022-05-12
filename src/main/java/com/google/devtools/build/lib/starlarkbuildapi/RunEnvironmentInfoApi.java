// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkbuildapi;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import java.util.List;
import java.util.Map;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;

/**
 * Provider containing any additional environment variables for use when running executables, either
 * in test actions or when executed via the run command.
 */
@StarlarkBuiltin(
    name = "RunEnvironmentInfo",
    category = DocCategory.PROVIDER,
    doc =
        "A provider that can be returned from executable rules to control the environment in"
            + " which their executable is executed.")
public interface RunEnvironmentInfoApi extends StructApi {

  @StarlarkMethod(
      name = "environment",
      doc =
          "A map of string keys and values that represent environment variables and their values."
              + " These will be made available when the target that returns this provider is"
              + " executed, either as a test or via the run command.",
      structField = true)
  Map<String, String> getEnvironment();

  @StarlarkMethod(
      name = "inherited_environment",
      doc =
          "A sequence of names of environment variables. These variables are made  available with"
              + " their current value taken from the shell environment when the target that returns"
              + " this provider is executed, either as a test or via the run command. If a variable"
              + " is contained in both <code>environment</code> and"
              + " <code>inherited_environment</code>, the value inherited from the shell"
              + " environment will take precedence if set.",
      structField = true)
  List<String> getInheritedEnvironment();

  /** Provider for {@link RunEnvironmentInfoApi}. */
  @StarlarkBuiltin(name = "Provider", category = DocCategory.PROVIDER, documented = false, doc = "")
  interface RunEnvironmentInfoApiProvider extends ProviderApi {

    @StarlarkMethod(
        name = "RunEnvironmentInfo",
        doc = "",
        documented = false,
        parameters = {
          @Param(
              name = "environment",
              defaultValue = "{}",
              named = true,
              positional = true,
              doc =
                  "A map of string keys and values that represent environment variables and their"
                      + " values. These will be made available when the target that returns this"
                      + " provider is executed, either as a test or via the run command."),
          @Param(
              name = "inherited_environment",
              allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
              defaultValue = "[]",
              named = true,
              positional = true,
              doc =
                  "A sequence of names of environment variables. These variables are made "
                      + " available with their current value taken from the shell environment"
                      + " when the target that returns this provider is executed, either as a"
                      + " test or via the run command. If a variable is contained in both <code>"
                      + "environment</code> and <code>inherited_environment</code>, the value"
                      + " inherited from the shell environment will take precedence if set.")
        },
        selfCall = true)
    @StarlarkConstructor
    RunEnvironmentInfoApi constructor(
        Dict<?, ?> environment, // <String, String> expected
        Sequence<?> inheritedEnvironment /* <String> expected */)
        throws EvalException;
  }
}
