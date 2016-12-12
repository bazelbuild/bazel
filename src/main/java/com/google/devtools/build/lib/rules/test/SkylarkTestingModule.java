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

import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkSignatureProcessor;

/** A class that exposes testing infrastructure to skylark. */
@SkylarkModule(
  name = "testing",
  doc = "Helper methods for skylark to access testing infrastructure."
)
public class SkylarkTestingModule {

  // TODO(bazel-team): Change this BuiltinFunction to be the actual
  // ExecutionInfoProvider.SKYLARK_CONSTRUCTOR.
  @SkylarkSignature(
    name = "ExecutionInfo",
    objectType = SkylarkTestingModule.class,
    returnType = ExecutionInfoProvider.class,
    doc =
        "Creates a new execution info provider. Use this provider to specify special"
            + "environments requirements needed to run tests.",
    parameters = {
      @Param(name = "self", type = SkylarkTestingModule.class, doc = "The 'testing' instance."),
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
  public static final BuiltinFunction NEW_EXECUTION_INFO_PROVIDER =
      new BuiltinFunction("ExecutionInfo") {
        @SuppressWarnings("unused")
        // This method is registered statically for skylark, and never called directly.
        public ExecutionInfoProvider invoke(SkylarkTestingModule self, SkylarkDict requirements) {
          return new ExecutionInfoProvider(requirements);
        }
      };

  // TODO(bazel-team): Change this BuiltinFunction to be the actual
  // TestEnvironmentProvider.SKYLARK_CONSTRUCTOR.
  @SkylarkSignature(
    name = "TestEnvironment",
    objectType = SkylarkTestingModule.class,
    returnType = TestEnvironmentProvider.class,
    doc =
        "Creates a new test environment provider. Use this provider to specify extra"
            + "environment variables to be made available during test execution.",
    parameters = {
      @Param(name = "self", type = SkylarkTestingModule.class, doc = "The 'testing' instance."),
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
  public static final BuiltinFunction NEW_TEST_ENVIRONMENT_PROVIDER =
      new BuiltinFunction("TestEnvironment") {
        @SuppressWarnings("unused")
        // This method is registered statically for skylark, and never called directly.
        public TestEnvironmentProvider invoke(SkylarkTestingModule self, SkylarkDict environment) {
          return new TestEnvironmentProvider(environment);
        }
      };

  static {
    SkylarkSignatureProcessor.configureSkylarkFunctions(SkylarkTestingModule.class);
  }
}
