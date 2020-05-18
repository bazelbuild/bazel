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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;

/**
 * This provider can be implemented by rules which need special environments to run in (especially
 * tests).
 */
@StarlarkBuiltin(name = "ExecutionInfo", doc = "", documented = false)
public interface ExecutionInfoApi extends StructApi {

  @StarlarkMethod(
      name = "requirements",
      doc = "A dict indicating special execution requirements, such as hardware platforms.",
      structField = true)
  ImmutableMap<String, String> getExecutionInfo();
}
