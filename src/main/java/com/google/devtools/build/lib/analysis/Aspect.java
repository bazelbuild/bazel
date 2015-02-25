// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.UnmodifiableIterator;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Extra information about a configured target computed on request of a dependent.
 *
 * <p>Analogous to {@link ConfiguredTarget}: contains a bunch of transitive info providers, which
 * are merged with the providers of the associated configured target before they are passed to
 * the configured target factories that depend on the configured target to which this aspect is
 * added.
 *
 * <p>Aspects are created alongside configured targets on request from dependents.
 */
@Immutable
public final class Aspect implements Iterable<TransitiveInfoProvider> {
  private final
      ImmutableMap<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> providers;

  private Aspect(
      ImmutableMap<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> providers) {
    this.providers = providers;
  }

  /**
   * Returns the providers created by the aspect.
   */
  public ImmutableMap<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider>
      getProviders() {
    return providers;
  }

  @Override
  public UnmodifiableIterator<TransitiveInfoProvider> iterator() {
    return providers.values().iterator();
  }

  /**
   * Builder for {@link Aspect}.
   */
  public static class Builder {
    private final Map<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider>
        providers = new LinkedHashMap<>();

    /**
     * Adds a provider to the aspect.
     */
    public Builder addProvider(
        Class<? extends TransitiveInfoProvider> key, TransitiveInfoProvider value) {
      Preconditions.checkNotNull(key);
      Preconditions.checkNotNull(value);
      AnalysisUtils.checkProvider(key);
      Preconditions.checkState(!providers.containsKey(key));
      providers.put(key, value);
      return this;
    }

    public Aspect build() {
      return new Aspect(ImmutableMap.copyOf(providers));
    }
  }
}