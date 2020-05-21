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
import com.google.devtools.build.lib.packages.StarlarkLibrary;
import com.google.devtools.build.lib.packages.StarlarkNativeModule;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.skylarkbuildapi.TopLevelBootstrap;

/** The basis for a Starlark Environment with all build-related modules registered. */
public final class StarlarkModules {

  private StarlarkModules() { }

  /** A bootstrap for non-rules-specific built-ins of the build API. */
  private static TopLevelBootstrap topLevelBootstrap =
      new TopLevelBootstrap(
          new BazelBuildApiGlobals(),
          new StarlarkAttrModule(),
          new StarlarkCommandLine(),
          new StarlarkNativeModule(),
          new StarlarkRuleClassFunctions(),
          StructProvider.STRUCT,
          OutputGroupInfo.STARLARK_CONSTRUCTOR,
          ActionsProvider.INSTANCE,
          DefaultInfo.PROVIDER);

  /** Adds predeclared Starlark bindings for the Bazel build language. */
  // TODO(adonovan): rename "globals" -> "builtins"
  public static void addStarlarkGlobalsToBuilder(ImmutableMap.Builder<String, Object> predeclared) {
    predeclared.putAll(StarlarkLibrary.COMMON); // e.g. select, depset
    topLevelBootstrap.addBindingsToBuilder(predeclared);
  }
}
