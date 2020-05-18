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

import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import java.util.Map;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;

/** Provider containing any additional environment variables for use in the test action. */
@StarlarkBuiltin(name = "TestEnvironmentInfo", doc = "", documented = false)
public interface TestEnvironmentInfoApi extends StructApi {

  @StarlarkMethod(
      name = "environment",
      doc = "A dict containing environment variables which should be set on the test action.",
      structField = true)
  Map<String, String> getEnvironment();
}
