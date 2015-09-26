// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * Configured targets implementing this provider can contribute Android IDL information to the
 * compilation.
 */
@Immutable
public final class AndroidIdlProvider implements TransitiveInfoProvider {

  public static final AndroidIdlProvider EMPTY = new AndroidIdlProvider(
      NestedSetBuilder.<String>emptySet(Order.STABLE_ORDER),
      NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
      NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER));

  private final NestedSet<String> transitiveIdlImportRoots;
  private final NestedSet<Artifact> transitiveIdlImports;
  private final NestedSet<Artifact> transitiveIdlJars;

  public AndroidIdlProvider(
      NestedSet<String> transitiveIdlImportRoots,
      NestedSet<Artifact> transitiveIdlImports,
      NestedSet<Artifact> transitiveIdlJars) {
    this.transitiveIdlImportRoots = transitiveIdlImportRoots;
    this.transitiveIdlImports = transitiveIdlImports;
    this.transitiveIdlJars = transitiveIdlJars;
  }

  /**
   * The set of IDL import roots need for compiling the IDL sources in the transitive closure.
   */
  public NestedSet<String> getTransitiveIdlImportRoots() {
    return transitiveIdlImportRoots;
  }

  /**
   * The IDL files in the transitive closure.
   */
  public NestedSet<Artifact> getTransitiveIdlImports() {
    return transitiveIdlImports;
  }

  /**
   * The IDL jars in the transitive closure, both class and source jars.
   */
  public NestedSet<Artifact> getTransitiveIdlJars() {
    return transitiveIdlJars;
  }
}
