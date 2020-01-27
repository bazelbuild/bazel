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
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/**
 * A target that provides native libraries in the transitive closure of its deps that are needed for
 * executing Java code.
 */
@Immutable
@AutoCodec
public final class JavaNativeLibraryProvider implements TransitiveInfoProvider {

  private final NestedSet<LibraryToLink> transitiveJavaNativeLibraries;

  public JavaNativeLibraryProvider(NestedSet<LibraryToLink> transitiveJavaNativeLibraries) {
    this.transitiveJavaNativeLibraries = transitiveJavaNativeLibraries;
  }

  /**
   * Collects native libraries in the transitive closure of its deps that are needed for executing
   * Java code.
   */
  public NestedSet<LibraryToLink> getTransitiveJavaNativeLibraries() {
    return transitiveJavaNativeLibraries;
  }

  public static JavaNativeLibraryProvider merge(Iterable<JavaNativeLibraryProvider> deps) {
    NestedSetBuilder<LibraryToLink> transitiveSourceJars = NestedSetBuilder.stableOrder();

    for (JavaNativeLibraryProvider wrapper : deps) {
      transitiveSourceJars.addTransitive(wrapper.getTransitiveJavaNativeLibraries());
    }
    return new JavaNativeLibraryProvider(transitiveSourceJars.build());
  }
}
