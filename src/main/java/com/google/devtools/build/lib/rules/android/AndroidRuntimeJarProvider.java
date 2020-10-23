// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import java.util.HashMap;

/**
 * Provider of Jar files transitively to be included into the runtime classpath of an Android app.
 */
@Immutable
public class AndroidRuntimeJarProvider implements TransitiveInfoProvider {

  /** Provider that doesn't provide any runtime Jars, typically used for neverlink targets. */
  public static final AndroidRuntimeJarProvider NEVERLINK =
      new AndroidRuntimeJarProvider(
          NestedSetBuilder.<ImmutableMap<Artifact, Artifact>>emptySet(STABLE_ORDER));

  /** Builder for {@link AndroidRuntimeJarProvider}. */
  public static class Builder {

    private final ImmutableMap.Builder<Artifact, Artifact> newlyDesugared = ImmutableMap.builder();
    private final NestedSetBuilder<ImmutableMap<Artifact, Artifact>> transitiveMappings =
        NestedSetBuilder.stableOrder();

    public Builder() {}

    /**
     * Copies all mappings from the given providers, which is useful to aggregate providers from
     * dependencies.
     */
    public Builder addTransitiveProviders(Iterable<AndroidRuntimeJarProvider> providers) {
      for (AndroidRuntimeJarProvider provider : providers) {
        transitiveMappings.addTransitive(provider.runtimeJars);
      }
      return this;
    }

    /** Adds a mapping from a Jar to its desugared version. */
    public Builder addDesugaredJar(Artifact jar, Artifact desugared) {
      newlyDesugared.put(checkNotNull(jar, "jar"), checkNotNull(desugared, "desugared"));
      return this;
    }

    /** Returns the finished {@link AndroidRuntimeJarProvider}. */
    public AndroidRuntimeJarProvider build() {
      return new AndroidRuntimeJarProvider(transitiveMappings.add(newlyDesugared.build()).build());
    }
  }

  /** Mappings from Jar artifacts to the corresponding dex archives. */
  private final NestedSet<ImmutableMap<Artifact, Artifact>> runtimeJars;

  private AndroidRuntimeJarProvider(NestedSet<ImmutableMap<Artifact, Artifact>> runtimeJars) {
    this.runtimeJars = runtimeJars;
  }

  /**
   * Returns function that maps Jars to desugaring results if available and returns the given Jar
   * otherwise.
   */
  public Function<Artifact, Artifact> collapseToFunction() {
    final HashMap<Artifact, Artifact> collapsed = new HashMap<>();
    for (ImmutableMap<Artifact, Artifact> partialMapping : runtimeJars.toList()) {
      collapsed.putAll(partialMapping);
    }
    return jar -> {
      Artifact result = collapsed.get(jar);
      return result != null ? result : jar; // return null iff input == null
    };
  }
}
