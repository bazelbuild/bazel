// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ActionsProvider;
import com.google.devtools.build.lib.analysis.DefaultInfo;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.packages.StarlarkLibrary;
import com.google.devtools.build.lib.packages.StructProvider;
import net.starlark.java.eval.Starlark;

/** A helper class for determining essential Build Language builtins. */
public final class StarlarkModules {

  private StarlarkModules() {}

  /**
   * Adds essential predeclared symbols for the Build Language.
   *
   * <p>This includes generic symbols like {@code rule()}, but not symbols specific to a rule
   * family, like {@code CcInfo}; those are registered on a RuleClassProvider instead. This also
   * does not include Starlark Universe symbols like {@code len()}.
   */
  public static void addPredeclared(ImmutableMap.Builder<String, Object> predeclared) {
    predeclared.putAll(StarlarkLibrary.COMMON); // e.g. select, depset
    Starlark.addMethods(predeclared, new BazelBuildApiGlobals()); // e.g. configuration_field
    Starlark.addMethods(predeclared, new StarlarkRuleClassFunctions()); // e.g. rule
    Starlark.addModule(predeclared, new StarlarkCommandLine()); // cmd_helper module
    Starlark.addModule(predeclared, new StarlarkAttrModule()); // attr module
    predeclared.put("struct", StructProvider.STRUCT);
    predeclared.put("OutputGroupInfo", OutputGroupInfo.STARLARK_CONSTRUCTOR);
    predeclared.put("Actions", ActionsProvider.INSTANCE);
    predeclared.put("DefaultInfo", DefaultInfo.PROVIDER);
  }
}
