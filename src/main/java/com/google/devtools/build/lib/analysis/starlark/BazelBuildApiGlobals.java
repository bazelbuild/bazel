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

package com.google.devtools.build.lib.analysis.starlark;

import com.google.devtools.build.lib.packages.BazelStarlarkContext;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkBuildApiGlobals;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkThread;

/**
 * Bazel implementation of {@link StarlarkBuildApiGlobals}: a collection of global Starlark build
 * API functions that belong in the global namespace.
 */
public class BazelBuildApiGlobals implements StarlarkBuildApiGlobals {

  @Override
  public StarlarkLateBoundDefault<?> configurationField(
      String fragment, String name, StarlarkThread thread) throws EvalException {
    BazelStarlarkContext bazelContext = BazelStarlarkContext.from(thread);
    Class<?> fragmentClass = bazelContext.getFragmentNameToClass().get(fragment);
    if (fragmentClass == null) {
      throw Starlark.errorf("invalid configuration fragment name '%s'", fragment);
    }
    try {
      return StarlarkLateBoundDefault.forConfigurationField(
          fragmentClass, name, bazelContext.getToolsRepository());
    } catch (StarlarkLateBoundDefault.InvalidConfigurationFieldException exception) {
      throw new EvalException(exception);
    }
  }
}
