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
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A Skyframe value representing the Starlark symbols defined by the {@code @_builtins}
 * pseudo-repository.
 *
 * <p>These are parsed from {@code @_builtins//:exports.bzl}.
 */
public final class StarlarkBuiltinsValue implements SkyValue {

  static final String BUILTINS_NAME = "@_builtins";

  /**
   * The builtins pseudo-repository.
   *
   * <p>It is illegal to declare a repository in WORKSPACE whose name collides with this one.
   * Whether a collision has in fact occurred is determined by {@link RepositoryName#equals}, and
   * hence depends on the OS path policy (i.e. the case sensitivity of the host system).
   *
   * <p>Regardless of path policy, all actual uses of the builtins pseudo-repo are case sensitive
   * and must match {@link #BUILTINS_NAME} exactly.
   */
  // TODO(#11437): Add actual enforcement that users cannot define a repo named "@_builtins".
  @SerializationConstant static final RepositoryName BUILTINS_REPO;

  /** Reports whether the given repository is the special builtins pseudo-repository. */
  static boolean isBuiltinsRepo(RepositoryName repo) {
    // Use String.equals(), not RepositoryName.equals(), to force case sensitivity.
    return repo.getName().equals(BUILTINS_NAME);
  }

  static {
    try {
      BUILTINS_REPO = RepositoryName.create(BUILTINS_NAME);
    } catch (LabelSyntaxException e) {
      throw new IllegalStateException(e);
    }
  }

  // These are all deeply immutable (the Starlark values are already frozen), so let's skip the
  // accessors and mutators.

  /** Top-level predeclared symbols for a .bzl file (loaded on behalf of a BUILD file). */
  // TODO(#11437): Corresponding predeclaredForBuild for BUILD files
  public final ImmutableMap<String, Object> predeclaredForBuildBzl;

  /** Contents of the {@code exported_to_java} dict. */
  public final ImmutableMap<String, Object> exportedToJava;

  /** Transitive digest of all .bzl files in {@code @_builtins}. */
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
   * Skyframe key for retrieving the {@code @_builtins} definitions.
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
      return "Starlark @_builtins";
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
