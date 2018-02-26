// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.java;

import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.Collection;

/**
 * The collection of labels of exported targets and artifacts reached via "exports" attribute
 * transitively.
 */
@Immutable
@AutoCodec
public final class JavaExportsProvider implements TransitiveInfoProvider {

  private final NestedSet<Label> transitiveExports;

  JavaExportsProvider(NestedSet<Label> transitiveExports) {
    this.transitiveExports = transitiveExports;
  }

  public static JavaExportsProvider merge(Collection<JavaExportsProvider> providers) {
    NestedSetBuilder<Label> builder = NestedSetBuilder.stableOrder();

    providers.stream()
        .map(JavaExportsProvider::getTransitiveExports)
        .forEach(builder::addTransitive);

    return new JavaExportsProvider(builder.build());
  }

  /**
   * Returns the labels of exported targets and artifacts reached transitively through the "exports"
   * attribute.
   */
  public NestedSet<Label> getTransitiveExports() {
    return transitiveExports;
  }
}
