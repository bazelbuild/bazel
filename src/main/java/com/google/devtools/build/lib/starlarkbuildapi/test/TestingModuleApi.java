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

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.RunEnvironmentInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkRuleFunctionsApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/** Helper module for accessing test infrastructure. */
@StarlarkBuiltin(
    name = "testing",
    category = DocCategory.TOP_LEVEL_MODULE,
    doc = "Helper methods for Starlark to access testing infrastructure.")
public interface TestingModuleApi extends StarlarkValue {

  @StarlarkMethod(
      name = "ExecutionInfo",
      doc =
          "<a href='../providers/ExecutionInfo.html'>testing.ExecutionInfo</a> provider"
              + " key/constructor",
      structField = true)
  ExecutionInfoApi.ExecutionInfoApiProvider executionInfo() throws EvalException;

  @StarlarkMethod(
      name = "TestEnvironment",
      doc =
          "<b>Deprecated: Use RunEnvironmentInfo instead.</b> Creates a new test environment "
              + "provider. Use this provider to specify extra environment variables to be made "
              + "available during test execution.",
      parameters = {
        @Param(
            name = "environment",
            named = true,
            positional = true,
            doc =
                "A map of string keys and values that represent environment variables and their"
                    + " values. These will be made available during the test execution."),
        @Param(
            name = "inherited_environment",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            defaultValue = "[]",
            named = true,
            positional = true,
            doc =
                "A sequence of names of environment variables. These variables are made available"
                    + " during the test execution with their current value taken from the shell"
                    + " environment. If a variable is contained in both <code>environment</code>"
                    + " and <code>inherited_environment</code>, the value inherited from the"
                    + " shell environment will take precedence if set.")
      })
  RunEnvironmentInfoApi testEnvironment(
      Dict<?, ?> environment, // <String, String> expected
      Sequence<?> inheritedEnvironment /* <String> expected */)
      throws EvalException;

  @StarlarkMethod(
      name = "analysis_test",
      doc =
          "Creates a new analysis test target. <p>The number of transitive dependencies of the test"
              + " are limited. The limit is controlled by"
              + " <code>--analysis_testing_deps_limit</code> flag.",
      parameters = {
        @Param(
            name = "name",
            named = true,
            doc =
                "Name of the target. It should be a Starlark identifier, matching pattern"
                    + " '[A-Za-z_][A-Za-z0-9_]*'."),
        @Param(
            name = "implementation",
            named = true,
            doc =
                "The Starlark function implementing this analysis test. It must have exactly one"
                    + " parameter: <a href=\"../builtins/ctx.html\">ctx</a>. The function is called"
                    + " during the analysis phase. It can access the attributes declared by"
                    + " <code>attrs</code> and populated via <code>attr_values</code>. The"
                    + " implementation function may not register actions. Instead, it must register"
                    + " a pass/fail result via providing <a"
                    + " href='../providers/AnalysisTestResultInfo.html'>AnalysisTestResultInfo</a>"
                    + "."),
        @Param(
            name = "attrs",
            allowedTypes = {
              @ParamType(type = Dict.class),
              @ParamType(type = NoneType.class),
            },
            named = true,
            defaultValue = "None",
            doc =
                "Dictionary declaring the attributes. See the <a"
                    + " href=\"../globals/bzl.html#rule\">rule</a> call. Attributes are allowed to"
                    + " use configuration transitions defined using <a "
                    + " href=\"../globals/bzl.html#analysis_test_transition\">analysis_test_transition</a>."),
        @Param(
            name = "fragments",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            named = true,
            defaultValue = "[]",
            doc =
                "List of configuration fragments that are available to the implementation of the"
                    + " analysis test."),
        @Param(
            name = StarlarkRuleFunctionsApi.TOOLCHAINS_PARAM,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = Object.class)},
            named = true,
            defaultValue = "[]",
            doc =
                "The set of toolchains the test requires. See the <a"
                    + " href=\"../globals/bzl.html#rule\">rule</a> call."),
        @Param(
            name = "attr_values",
            allowedTypes = {@ParamType(type = Dict.class, generic1 = String.class)},
            named = true,
            defaultValue = "{}",
            doc = "Dictionary of attribute values to pass to the implementation."),
      },
      useStarlarkThread = true,
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_ANALYSIS_TEST_CALL)
  void analysisTest(
      String name,
      StarlarkFunction implementation,
      Object attrs,
      Sequence<?> fragments,
      Sequence<?> toolchains,
      Object argsValue,
      StarlarkThread thread)
      throws EvalException, InterruptedException;
}
