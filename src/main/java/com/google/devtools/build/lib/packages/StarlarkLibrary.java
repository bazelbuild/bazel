// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import net.starlark.java.eval.Starlark;
import net.starlark.java.lib.json.Json;

/**
 * This class is the source of truth for what symbols are present in various Bazel Starlark
 * environments, prior to customization by the rule class provider and by builtins injection.
 *
 * <p>The set of available symbols is constructed in layers. This class represents the bottommost
 * layer: a hardcoded aggregation of APIs that are unconditionally present. The next layer is the
 * {@link ConfiguredRuleClassProvider}, on which a set of top-level symbols are registered. These
 * symbols may differ based on the available {@link BlazeModule}s but are fixed for a given Bazel
 * runtime. The final layer is {@link BazelStarlarkEnvironment}, which takes into account builtins
 * injection.
 *
 * <p>The different class fields are for the different types of Starlark environments appearing in
 * Bazel. They do not contain symbols like, {@code int()} or {@code len()}, which are part of {@link
 * Starlark#UNIVERSE} and implicitly present in all environments.
 */
public final class StarlarkLibrary {

  private StarlarkLibrary() {} // uninstantiable

  /**
   * Symbols that are common to all Bazel Starlark environments except .scl. This includes BUILD,
   * .bzl, WORKSPACE, and cquery.
   */
  public static final ImmutableMap<String, Object> COMMON = initCommon();

  private static ImmutableMap<String, Object> initCommon() {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    Starlark.addMethods(env, Depset.DepsetLibrary.INSTANCE);
    env.put("json", Json.INSTANCE);
    env.put("proto", Proto.INSTANCE);
    return env.buildOrThrow();
  }

  /**
   * Symbols to add for BUILD files.
   *
   * <p>This is a superset of {@link #COMMON}. It excludes rules, which are registered on the rule
   * class provider. It also excludes functions that are also in the {@code native} object such as
   * {@code package()} and {@code glob()}.
   */
  public static final ImmutableMap<String, Object> BUILD = initBUILD();

  private static ImmutableMap<String, Object> initBUILD() {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    Starlark.addMethods(env, BuildGlobals.INSTANCE);
    Starlark.addMethods(env, SelectorList.SelectLibrary.INSTANCE);
    env.putAll(COMMON);
    return env.buildOrThrow();
  }
}
