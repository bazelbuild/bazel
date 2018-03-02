// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.test;

import com.google.devtools.build.lib.analysis.test.ExecutionInfo;
import com.google.devtools.build.lib.analysis.test.TestEnvironmentInfo;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.SkylarkDict;

/** A class that exposes testing infrastructure to skylark. */
@SkylarkModule(
  name = "testing",
  doc = "Helper methods for skylark to access testing infrastructure."
)
public class SkylarkTestingModule {

  // TODO(bazel-team): Change this BuiltinFunction to be the actual ExecutionInfo.PROVIDER.
  @SkylarkCallable(
    name = "ExecutionInfo",
    doc =
        "Creates a new execution info provider. Use this provider to specify special"
            + "environments requirements needed to run tests.",
    parameters = {
      @Param(
        name = "requirements",
        type = SkylarkDict.class,
        named = false,
        positional = true,
        doc =
            "A map of string keys and values to indicate special execution requirements, such as"
                + " hardware platforms, etc. These keys and values are passed to the executor of"
                + " the test action as parameters to configure the execution environment."
      )
    }
  )
  public ExecutionInfo executionInfo(SkylarkDict<String, String> requirements) {
    return new ExecutionInfo(requirements);
  }

  // TODO(bazel-team): Change this BuiltinFunction to be the actual TestEnvironmentInfo.PROVIDER.
  @SkylarkCallable(
    name = "TestEnvironment",
    doc =
        "Creates a new test environment provider. Use this provider to specify extra"
            + "environment variables to be made available during test execution.",
    parameters = {
      @Param(
        name = "environment",
        type = SkylarkDict.class,
        named = false,
        positional = true,
        doc =
            "A map of string keys and values that represent environment variables and their values."
                + " These will be made available during the test execution."
      )
    }
  )
  public TestEnvironmentInfo testEnvironment(SkylarkDict<String, String> environment) {
    return new TestEnvironmentInfo(environment);
  }
}
