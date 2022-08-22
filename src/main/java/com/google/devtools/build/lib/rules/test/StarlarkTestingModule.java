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

import com.google.devtools.build.lib.analysis.RunEnvironmentInfo;
import com.google.devtools.build.lib.analysis.test.ExecutionInfo;
import com.google.devtools.build.lib.starlarkbuildapi.test.TestingModuleApi;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;

/** A class that exposes testing infrastructure to Starlark. */
public class StarlarkTestingModule implements TestingModuleApi {

  @Override
  public ExecutionInfo executionInfo(
      Dict<?, ?> requirements /* <String, String> */, String execGroup) throws EvalException {
    return new ExecutionInfo(
        Dict.cast(requirements, String.class, String.class, "requirements"), execGroup);
  }

  @Override
  public RunEnvironmentInfo testEnvironment(
      Dict<?, ?> environment /* <String, String> */,
      Sequence<?> inheritedEnvironment /* <String> */)
      throws EvalException {
    return new RunEnvironmentInfo(
        Dict.cast(environment, String.class, String.class, "environment"),
        StarlarkList.immutableCopyOf(
            Sequence.cast(inheritedEnvironment, String.class, "inherited_environment")),
        /* shouldErrorOnNonExecutableRule */ false);
  }
}
