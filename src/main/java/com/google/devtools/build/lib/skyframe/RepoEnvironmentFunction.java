// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

/**
 * Skyframe function that provides the effective value for an environment variable in the context of
 * repository rules and module extensions. This will be the value from the repo environment as
 * provided by {@link com.google.devtools.build.lib.runtime.CommandEnvironment#getRepoEnv()}.
 *
 * <p>Operates analogously to {@link ClientEnvironmentFunction}: the effective environment is read
 * from an {@link AtomicReference} whose individual entries are injected into Skyframe per-variable
 * (see {@code SkyframeExecutor#handleRepositoryEnvironmentChanges}). This is crucial for
 * incremental efficiency: changing a single variable must not invalidate consumers of the others.
 * Reading the whole environment as one Skyframe node instead would result in widespread
 * invalidation of all dependents whenever any variable changes. Crucially, change pruning is not
 * fully effective here as it doesn't resurrect nodes that aren't part of the current evaluation,
 * but every evaluation marks all transitive dependents as dirty.
 */
public final class RepoEnvironmentFunction implements SkyFunction {
  private final AtomicReference<Map<String, String>> repoEnv;

  public RepoEnvironmentFunction(AtomicReference<Map<String, String>> repoEnv) {
    this.repoEnv = repoEnv;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) {
    return new EnvironmentVariableValue(repoEnv.get().get((String) skyKey.argument()));
  }

  /** Returns the SkyKey to invoke this function for the environment variable {@code variable}. */
  public static Key key(String variable) {
    return Key.create(variable);
  }

  @VisibleForSerialization
  @AutoCodec
  static class Key extends AbstractSkyKey<String> {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private Key(String arg) {
      super(arg);
    }

    private static Key create(String arg) {
      return interner.intern(new Key(arg));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static Key intern(Key key) {
      return interner.intern(key);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.REPOSITORY_ENVIRONMENT_VARIABLE;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }

  /**
   * Returns a map of environment variable key => values, getting them from Skyframe. Returns null
   * if and only if some dependencies from Skyframe still need to be resolved.
   */
  @Nullable
  public static ImmutableSortedMap<String, Optional<String>> getEnvironmentView(
      Environment env, Set<String> keys) throws InterruptedException {
    var skyKeys = Collections2.transform(keys, RepoEnvironmentFunction::key);
    SkyframeLookupResult values = env.getValuesAndExceptions(skyKeys);
    if (env.valuesMissing()) {
      return null;
    }

    var result = ImmutableSortedMap.<String, Optional<String>>naturalOrder();
    for (var key : skyKeys) {
      var value = (EnvironmentVariableValue) values.get(key);
      if (value == null) {
        return null;
      }
      result.put(key.argument().toString(), Optional.ofNullable(value.value()));
    }
    return result.buildOrThrow();
  }
}
