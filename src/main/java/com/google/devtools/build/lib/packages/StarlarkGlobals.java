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

package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableMap;

/**
 * A library of the fixed Starlark environment for various contexts.
 *
 * <p>This is the source of truth for what symbols are available in what Starlark contexts (BUILD,
 * .bzl, etc.), before considering how symbols may be added by registering them on the rule class
 * provider, or how symbols may be substituted by builtins injection. In other words, this is the
 * starting point for defining the minimum Starlark environments that Bazel supports for BUILD
 * files, .bzl files, etc. See {@link BazelStarlarkEnvironment} for the final determination of the
 * environment after accounting for registered symbols and builtins injection.
 *
 * <p>This is split between an interface in the lib/packages/ directory and an implementation in the
 * lib/analysis/starlark/ directory, in order to avoid new dependency edges from lib/packages/ to
 * lib/analysis/.
 */
public interface StarlarkGlobals {

  /**
   * Returns a simple environment containing a few general utility modules, {@code depset}, and
   * {@code select()}.
   *
   * <p>In general, if you need a Bazel-y Starlark environment and don't know what to choose, prefer
   * to use this one for uniformity with as many other contexts as possible.
   */
  ImmutableMap<String, Object> getUtilToplevels();

  /**
   * Similar to {@link #getUtilToplevels} but without {@code select()} and with {@code struct}. Used
   * for cquery.
   */
  // TODO(bazel-team): Consider whether we should replace usage of this with getUtilTopLevels(), at
  // the cost of the cquery dialect changing slightly, for the sake of uniformity and fewer
  // kinds of environments.
  ImmutableMap<String, Object> getUtilToplevelsForCquery();

  /**
   * Returns the fixed top-levels for BUILD files that also happen to be fields of {@code native}.
   * This does not include any native rules.
   */
  ImmutableMap<String, Object> getFixedBuildFileToplevelsSharedWithNative();

  /** Returns the fixed top-levels for BUILD files that are *not* also fields of {@code native}. */
  ImmutableMap<String, Object> getFixedBuildFileToplevelsNotInNative();

  /** Returns the fixed top-levels for .bzl files, excluding the {@code native} object. */
  ImmutableMap<String, Object> getFixedBzlToplevels();

  /** Returns the top-levels for .scl files. */
  ImmutableMap<String, Object> getSclToplevels();

  /** Returns the top-levels for MODULE.bazel files and their imports. */
  ImmutableMap<String, Object> getModuleToplevels();

  /** Returns the top-levels for REPO.bazel files. */
  ImmutableMap<String, Object> getRepoToplevels();
}
