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
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileGlobals;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.packages.BuildGlobals;
import com.google.devtools.build.lib.packages.Proto;
import com.google.devtools.build.lib.packages.RepoCallable;
import com.google.devtools.build.lib.packages.SelectorList;
import com.google.devtools.build.lib.packages.StarlarkGlobals;
import com.google.devtools.build.lib.packages.StarlarkNativeModule;
import com.google.devtools.build.lib.packages.StructProvider;
import net.starlark.java.eval.Starlark;
import net.starlark.java.lib.json.Json;

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

  private void addCommonUtilToplevels(ImmutableMap.Builder<String, Object> env) {
    // Maintainer's note: Changes to this method are relatively high impact since it's sourced for
    // BUILD, .bzl, and even cquery environments.
    Starlark.addMethods(env, Depset.DepsetLibrary.INSTANCE);
    env.put("json", Json.INSTANCE);
    env.put("proto", Proto.INSTANCE);
  }

  @Override
  public ImmutableMap<String, Object> getUtilToplevels() {
    // TODO(bazel-team): It's dubious that we include things like depset and select(), but not
    // struct() here.
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    addCommonUtilToplevels(env);
    Starlark.addMethods(env, SelectorList.SelectLibrary.INSTANCE);
    return env.buildOrThrow();
  }

  @Override
  public ImmutableMap<String, Object> getUtilToplevelsForCquery() {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    addCommonUtilToplevels(env);
    env.put("struct", StructProvider.STRUCT);
    return env.buildOrThrow();
  }

  @Override
  public ImmutableMap<String, Object> getFixedBuildFileToplevelsSharedWithNative() {
    return StarlarkNativeModule.BINDINGS_FOR_BUILD_FILES;
  }

  @Override
  public ImmutableMap<String, Object> getFixedBuildFileToplevelsNotInNative() {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();

    env.putAll(getUtilToplevels());
    Starlark.addMethods(env, BuildGlobals.INSTANCE);

    return env.buildOrThrow();
  }

  @Override
  public ImmutableMap<String, Object> getFixedBzlToplevels() {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();

    env.putAll(getUtilToplevels());

    Starlark.addMethods(env, new BazelBuildApiGlobals()); // e.g. configuration_field
    Starlark.addMethods(env, new StarlarkRuleClassFunctions()); // e.g. rule

    env.put("attr", new StarlarkAttrModule());
    env.put("struct", StructProvider.STRUCT);
    env.put("OutputGroupInfo", OutputGroupInfo.STARLARK_CONSTRUCTOR);
    env.put("Actions", ActionsProvider.INSTANCE);
    env.put("DefaultInfo", DefaultInfo.PROVIDER);
    env.put("RunEnvironmentInfo", RunEnvironmentInfo.PROVIDER);

    return env.buildOrThrow();
  }

  @Override
  public ImmutableMap<String, Object> getSclToplevels() {
    // TODO(bazel-team): We only want the visibility() symbol from BazelBuildApiGlobals, nothing
    // else, but Starlark#addMethods doesn't allow that kind of granularity, and the Starlark
    // interpreter doesn't provide any other way to turn a Java method definition into a
    // callable symbol. So we hack it by building the map of all symbols in that class and
    // retrieving just the one we want. The alternative of refactoring the class is more churn than
    // its worth, given the starlarkbuildapi/ split.
    ImmutableMap.Builder<String, Object> bazelBuildApiGlobalsSymbols = ImmutableMap.builder();
    Starlark.addMethods(bazelBuildApiGlobalsSymbols, new BazelBuildApiGlobals());
    Object visibilitySymbol = bazelBuildApiGlobalsSymbols.buildOrThrow().get("visibility");

    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    env.put("visibility", visibilitySymbol);
    env.put("struct", StructProvider.STRUCT);
    return env.buildOrThrow();
  }

  @Override
  public ImmutableMap<String, Object> getModuleToplevels() {
    var env = ImmutableMap.<String, Object>builder();
    Starlark.addMethods(env, new ModuleFileGlobals());
    return env.buildOrThrow();
  }

  @Override
  public ImmutableMap<String, Object> getRepoToplevels() {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    Starlark.addMethods(env, RepoCallable.INSTANCE);
    return env.buildOrThrow();
  }
}
