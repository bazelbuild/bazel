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

package com.google.devtools.build.lib.skylarkbuildapi.test;

import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Helper module for accessing test infrastructure. */
@SkylarkModule(
    name = "testing",
    doc = "Helper methods for Starlark to access testing infrastructure.")
public interface TestingModuleApi extends StarlarkValue {

  // TODO(bazel-team): Change this function to be the actual ExecutionInfo.PROVIDER.
  @SkylarkCallable(
      name = "ExecutionInfo",
      doc =
          "Creates a new execution info provider. Use this provider to specify special"
              + "environments requirements needed to run tests.",
      parameters = {
        @Param(
            name = "requirements",
            type = Dict.class,
            named = false,
            positional = true,
            doc =
                "A map of string keys and values to indicate special execution requirements,"
                    + " such as hardware platforms, etc. These keys and values are passed to the"
                    + " executor of the test action as parameters to configure the execution"
                    + " environment.")
      })
  ExecutionInfoApi executionInfo(Dict<?, ?> requirements // <String, String> expected
      ) throws EvalException;

  // TODO(bazel-team): Change this function to be the actual TestEnvironmentInfo.PROVIDER.
  @SkylarkCallable(
      name = "TestEnvironment",
      doc =
          "Creates a new test environment provider. Use this provider to specify extra"
              + "environment variables to be made available during test execution.",
      parameters = {
        @Param(
            name = "environment",
            type = Dict.class,
            named = false,
            positional = true,
            doc =
                "A map of string keys and values that represent environment variables and their"
                    + " values. These will be made available during the test execution.")
      })
  TestEnvironmentInfoApi testEnvironment(Dict<?, ?> environment // <String, String> expected
      ) throws EvalException;
}
