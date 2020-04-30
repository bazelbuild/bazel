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

package com.google.devtools.build.lib.analysis.skylark;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ActionsProvider;
import com.google.devtools.build.lib.analysis.DefaultInfo;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.packages.SkylarkNativeModule;
import com.google.devtools.build.lib.packages.StarlarkLibrary;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.skylarkbuildapi.TopLevelBootstrap;
import com.google.devtools.build.lib.syntax.Starlark;

/** The basis for a Starlark Environment with all build-related modules registered. */
public final class SkylarkModules {

  private SkylarkModules() { }

  /** A bootstrap for non-rules-specific globals of the build API. */
  private static TopLevelBootstrap topLevelBootstrap =
      new TopLevelBootstrap(
          new BazelBuildApiGlobals(),
          new StarlarkAttrModule(),
          new StarlarkCommandLine(),
          new SkylarkNativeModule(),
          new SkylarkRuleClassFunctions(),
          StructProvider.STRUCT,
          OutputGroupInfo.SKYLARK_CONSTRUCTOR,
          ActionsProvider.INSTANCE,
          DefaultInfo.PROVIDER);

  /**
   * Adds bindings for Starlark built-ins and non-rules-specific globals of the build API to the
   * given environment map builder.
   */
  public static void addSkylarkGlobalsToBuilder(ImmutableMap.Builder<String, Object> env) {
    env.putAll(Starlark.UNIVERSE);
    env.putAll(StarlarkLibrary.COMMON); // e.g. select, depset
    topLevelBootstrap.addBindingsToBuilder(env);
  }
}
