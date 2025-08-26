// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * A Skyframe value representing the result of evaluating the {@code @_builtins} pseudo-repository,
 * and in Bazel where applicable, applying autoloads.
 *
 * <p>To avoid unnecessary Skyframe edges, the {@code StarlarkSemantics} are included in this value,
 * so that a caller who obtains a StarlarkBuiltinsValue can also access the StarlarkSemantics
 * without an additional dependency.
 *
 * <p>These are parsed from {@code @_builtins//:exports.bzl}.
 */
public final class StarlarkBuiltinsValue implements SkyValue {

  /** Reports whether the given repository is the special builtins pseudo-repository. */
  public static boolean isBuiltinsRepo(RepositoryName repo) {
    // Use String.equals(), not RepositoryName.equals(), to force case sensitivity.
    return repo.getName().equals(RepositoryName.BUILTINS.getName()) && repo.isVisible();
  }

  // These are all (except transitiveDigest) deeply immutable since the Starlark values are already
  // frozen, so let's skip the accessors and mutators.

  /**
   * Top-level predeclared symbols for a .bzl file loaded on behalf of a BUILD file, after builtins
   * injection has been applied.
   */
  public final ImmutableMap<String, Object> predeclaredForBuildBzl;

  /**
   * Top-level predeclared symbols for a .bzl file loaded on behalf of a MODULE file after builtins
   * injection has been applied.
   */
  public final ImmutableMap<String, Object> predeclaredForModuleBzl;

  /**
   * Top-level predeclared symbols for a BUILD file, after builtins injection but before any prelude
   * file has been applied.
   */
  public final ImmutableMap<String, Object> predeclaredForBuild;

  /** Contents of the {@code exported_to_java} dict. */
  public final ImmutableMap<String, Object> exportedToJava;

  /** Transitive digest of all .bzl files in {@code @_builtins}. */
  public final byte[] transitiveDigest;

  /** The StarlarkSemantics used for {@code @_builtins} evaluation. */
  public final StarlarkSemantics starlarkSemantics;

  private StarlarkBuiltinsValue(
      ImmutableMap<String, Object> predeclaredForBuildBzl,
      ImmutableMap<String, Object> predeclaredForModuleBzl,
      ImmutableMap<String, Object> predeclaredForBuild,
      ImmutableMap<String, Object> exportedToJava,
      byte[] transitiveDigest,
      StarlarkSemantics starlarkSemantics) {
    this.predeclaredForBuildBzl = predeclaredForBuildBzl;
    this.predeclaredForModuleBzl = predeclaredForModuleBzl;
    this.predeclaredForBuild = predeclaredForBuild;
    this.exportedToJava = exportedToJava;
    this.transitiveDigest = transitiveDigest;
    this.starlarkSemantics = starlarkSemantics;
  }

  public static StarlarkBuiltinsValue create(
      ImmutableMap<String, Object> predeclaredForBuildBzl,
      ImmutableMap<String, Object> predeclaredForModuleBzl,
      ImmutableMap<String, Object> predeclaredForBuild,
      ImmutableMap<String, Object> exportedToJava,
      byte[] transitiveDigest,
      StarlarkSemantics starlarkSemantics) {
    return new StarlarkBuiltinsValue(
        predeclaredForBuildBzl,
        predeclaredForModuleBzl,
        predeclaredForBuild,
        exportedToJava,
        transitiveDigest,
        starlarkSemantics);
  }

  /**
   * Constructs a placeholder builtins value to be used when builtins injection is disabled, or for
   * use within builtins evaluation itself.
   *
   * <p>The placeholder simply wraps the StarlarkSemantics object. This lets code paths that don't
   * use injection still conveniently access the semantics without incurring a separate Skyframe
   * edge.
   */
  public static StarlarkBuiltinsValue createEmpty(StarlarkSemantics starlarkSemantics) {
    return new StarlarkBuiltinsValue(
        /* predeclaredForBuildBzl= */ ImmutableMap.of(),
        /* predeclaredForModuleBzl= */ ImmutableMap.of(),
        /* predeclaredForBuild= */ ImmutableMap.of(),
        /* exportedToJava= */ ImmutableMap.of(),
        /* transitiveDigest= */ new byte[] {},
        starlarkSemantics);
  }

  /** Returns the SkyKey for BuiltinsValue containing only additional builtin symbols and rules. */
  public static Key key() {
    return Key.INSTANCE;
  }

  /**
   * Returns the SkyKey for BuiltinsValue optionally amended with externally loaded symbols and
   * rules.
   */
  public static Key key(boolean withAutoloads) {
    return withAutoloads ? Key.INSTANCE_WITH_AUTOLOADS : Key.INSTANCE;
  }

  /**
   * Skyframe key for retrieving the {@code @_builtins} definitions.
   *
   * <p>This has no fields since there is only one {@code StarlarkBuiltinsValue} at a time.
   */
  static final class Key implements SkyKey {

    private final boolean withAutoloads;

    private static final Key INSTANCE = new Key(false);
    private static final Key INSTANCE_WITH_AUTOLOADS = new Key(true);

    private Key(boolean withAutoloads) {
      this.withAutoloads = withAutoloads;
    }

    public boolean isWithAutoloads() {
      return withAutoloads;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.STARLARK_BUILTINS;
    }

    @Override
    public String toString() {
      return "Starlark @_builtins";
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof Key key && this.withAutoloads == key.withAutoloads;
    }

    @Override
    public int hashCode() {
      return withAutoloads ? 7727 : 7277; // more or less xkcd/221
    }
  }
}
