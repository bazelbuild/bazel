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
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A Skyframe value representing the Starlark symbols defined by the {@code @builtins}
 * pseudo-repository.
 *
 * <p>These are parsed from {@code @builtins//:exports.bzl}.
 */
public final class StarlarkBuiltinsValue implements SkyValue {

  // These are all deeply immutable (the Starlark values are already frozen), so let's skip the
  // accessors and mutators.

  /** Top-level predeclared symbols for a .bzl file (loaded on behalf of a BUILD file). */
  // TODO(#11437): Corresponding predeclaredForBuild for BUILD files
  public final ImmutableMap<String, Object> predeclaredForBuildBzl;

  /** Contents of the {@code exported_to_java} dict. */
  public final ImmutableMap<String, Object> exportedToJava;

  /** Transitive digest of all .bzl files in {@code @builtins}. */
  public final byte[] transitiveDigest;

  public StarlarkBuiltinsValue(
      ImmutableMap<String, Object> predeclaredForBuildBzl,
      ImmutableMap<String, Object> exportedToJava,
      byte[] transitiveDigest) {
    this.predeclaredForBuildBzl = predeclaredForBuildBzl;
    this.exportedToJava = exportedToJava;
    this.transitiveDigest = transitiveDigest;
  }

  /** Returns the singleton SkyKey for this type of value. */
  public static Key key() {
    return Key.INSTANCE;
  }

  /**
   * Skyframe key for retrieving the {@code @builtins} definitions.
   *
   * <p>This has no fields since there is only one {@code StarlarkBuiltinsValue} at a time.
   */
  static final class Key implements SkyKey {

    private static final Key INSTANCE = new Key();

    private Key() {}

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.STARLARK_BUILTINS;
    }

    @Override
    public String toString() {
      return "Starlark @builtins";
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof Key;
    }

    @Override
    public int hashCode() {
      return 7727; // more or less xkcd/221
    }
  }
}
