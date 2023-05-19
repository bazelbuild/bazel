// Copyright 2023 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.analysis.RunEnvironmentInfo;
import com.google.devtools.build.lib.packages.SelectorList;
import com.google.devtools.build.lib.packages.StarlarkGlobals;
import com.google.devtools.build.lib.packages.StarlarkLibrary;
import com.google.devtools.build.lib.packages.StarlarkNativeModule;
import com.google.devtools.build.lib.packages.StructProvider;
import net.starlark.java.eval.Starlark;

/**
 * Sole implementation of {@link StarlarkGlobals}.
 *
 * <p>The reason for the class-interface split is to allow {@link BazelStarlarkEnvironment} to
 * retrieve symbols defined and aggregated in the lib/analysis/ dir, without creating a dependency
 * from lib/packages/ to lib/analysis.
 */
public final class StarlarkGlobalsImpl implements StarlarkGlobals {

  private StarlarkGlobalsImpl() {}

  public static final StarlarkGlobalsImpl INSTANCE = new StarlarkGlobalsImpl();

  @Override
  public ImmutableMap<String, Object> getFixedBuildFileToplevelsSharedWithNative() {
    return StarlarkNativeModule.BINDINGS_FOR_BUILD_FILES;
  }

  @Override
  public ImmutableMap<String, Object> getFixedBuildFileToplevelsNotInNative() {
    return StarlarkLibrary.BUILD; // e.g. select, depset
  }

  @Override
  public ImmutableMap<String, Object> getFixedBzlToplevels() {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    env.putAll(StarlarkLibrary.COMMON); // e.g. select, depset
    Starlark.addMethods(env, new BazelBuildApiGlobals()); // e.g. configuration_field
    Starlark.addMethods(env, new StarlarkRuleClassFunctions()); // e.g. rule
    Starlark.addMethods(env, SelectorList.SelectLibrary.INSTANCE);
    env.put("cmd_helper", new StarlarkCommandLine());
    env.put("attr", new StarlarkAttrModule());
    env.put("struct", StructProvider.STRUCT);
    env.put("OutputGroupInfo", OutputGroupInfo.STARLARK_CONSTRUCTOR);
    env.put("Actions", ActionsProvider.INSTANCE);
    env.put("DefaultInfo", DefaultInfo.PROVIDER);
    env.put("RunEnvironmentInfo", RunEnvironmentInfo.PROVIDER);
    return env.buildOrThrow();
  }
}
