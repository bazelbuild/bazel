// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * Configured targets implementing this provider can contribute Jack libraries to the compilation of
 * an Android APK using the Jack toolchain or another Jack library.
 *
 * @see <a href="http://tools.android.com/tech-docs/jackandjill">Jack documentation</a>
 * @see JackCompilationHelper
 */
@Immutable
public final class JackLibraryProvider implements TransitiveInfoProvider {
  public static final JackLibraryProvider EMPTY =
      new JackLibraryProvider(
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER));

  private final NestedSet<Artifact> transitiveJackLibrariesToLink;
  private final NestedSet<Artifact> transitiveJackClasspathLibraries;

  public JackLibraryProvider(
      NestedSet<Artifact> transitiveJackLibrariesToLink,
      NestedSet<Artifact> transitiveJackClasspathLibraries) {
    this.transitiveJackLibrariesToLink = Preconditions.checkNotNull(transitiveJackLibrariesToLink);
    this.transitiveJackClasspathLibraries =
        Preconditions.checkNotNull(transitiveJackClasspathLibraries);
  }

  /**
   * Gets the Jack libraries in the transitive closure which should be added to the final dex file.
   */
  public NestedSet<Artifact> getTransitiveJackLibrariesToLink() {
    return transitiveJackLibrariesToLink;
  }

  /**
   * Gets the Jack libraries which should be added to the classpath of any Jack action depending on
   * this provider.
   */
  public NestedSet<Artifact> getTransitiveJackClasspathLibraries() {
    return transitiveJackClasspathLibraries;
  }

  /**
   * Builder class to combine multiple JackLibraryProviders into a single one.
   */
  public static final class Builder {
    private final NestedSetBuilder<Artifact> transitiveJackLibrariesToLink =
        NestedSetBuilder.<Artifact>stableOrder();
    private final NestedSetBuilder<Artifact> transitiveJackClasspathLibraries =
        NestedSetBuilder.<Artifact>stableOrder();

    public Builder merge(JackLibraryProvider other) {
      transitiveJackLibrariesToLink.addTransitive(other.getTransitiveJackLibrariesToLink());
      transitiveJackClasspathLibraries.addTransitive(other.getTransitiveJackClasspathLibraries());
      return this;
    }

    public JackLibraryProvider build() {
      return new JackLibraryProvider(
          transitiveJackLibrariesToLink.build(), transitiveJackClasspathLibraries.build());
    }
  }
}
