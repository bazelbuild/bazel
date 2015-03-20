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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * Provides information contained in a {@code objc_options} target.
 */
@Immutable
final class OptionsProvider
    extends Value<OptionsProvider>
    implements TransitiveInfoProvider {
  static final class Builder {
    private Iterable<String> copts = ImmutableList.of();
    private final NestedSetBuilder<Artifact> infoplists = NestedSetBuilder.stableOrder();

    /**
     * Adds copts to the end of the copts sequence.
     */
    public Builder addCopts(Iterable<String> copts) {
      this.copts = Iterables.concat(this.copts, copts);
      return this;
    }

    public Builder addInfoplists(Iterable<Artifact> infoplists) {
      this.infoplists.addAll(infoplists);
      return this;
    }

    /**
     * Adds infoplists and copts from the given provider, if present. copts are added to the end of
     * the sequence.
     */
    public Builder addTransitive(Optional<OptionsProvider> maybeProvider) {
      for (OptionsProvider provider : maybeProvider.asSet()) {
        this.copts = Iterables.concat(this.copts, provider.copts);
        this.infoplists.addTransitive(provider.infoplists);
      }
      return this;
    }

    public OptionsProvider build() {
      return new OptionsProvider(ImmutableList.copyOf(copts), infoplists.build());
    }
  }

  public static final OptionsProvider DEFAULT = new Builder().build();

  private final ImmutableList<String> copts;
  private final NestedSet<Artifact> infoplists;

  private OptionsProvider(ImmutableList<String> copts, NestedSet<Artifact> infoplists) {
    super(copts, infoplists);
    this.copts = Preconditions.checkNotNull(copts);
    this.infoplists = Preconditions.checkNotNull(infoplists);
  }

  public ImmutableList<String> getCopts() {
    return copts;
  }

  public NestedSet<Artifact> getInfoplists() {
    return infoplists;
  }
}
