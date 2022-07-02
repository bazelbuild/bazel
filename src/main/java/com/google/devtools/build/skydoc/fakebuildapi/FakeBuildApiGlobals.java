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

package com.google.devtools.build.skydoc.fakebuildapi;

import com.google.devtools.build.lib.starlarkbuildapi.LateBoundDefaultApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkBuildApiGlobals;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;

/** Fake implementation of {@link StarlarkBuildApiGlobals}. */
public class FakeBuildApiGlobals implements StarlarkBuildApiGlobals {

  @Override
  public void visibility(Object value, StarlarkThread thread) throws EvalException {}

  @Override
  public LateBoundDefaultApi configurationField(String fragment, String name, StarlarkThread thread)
      throws EvalException {
    return new FakeLateBoundDefaultApi();
  }
}
