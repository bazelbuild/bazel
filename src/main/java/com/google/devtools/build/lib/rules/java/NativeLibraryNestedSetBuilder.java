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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.cpp.CcLinkingOutputs;
import com.google.devtools.build.lib.rules.cpp.CcNativeLibraryProvider;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.util.FileType;

/** A builder that helps construct nested sets of native libraries. */
public final class NativeLibraryNestedSetBuilder {

  private final NestedSetBuilder<LibraryToLink> builder = NestedSetBuilder.linkOrder();

  /** Build a nested set of native libraries. */
  public NestedSet<LibraryToLink> build() {
    return builder.build();
  }

  /** Include specified artifacts as native libraries in the nested set. */
  public NativeLibraryNestedSetBuilder addAll(Iterable<LibraryToLink> deps) {
    builder.addAll(deps);
    return this;
  }

  /** Include native libraries of specified dependencies into the nested set. */
  public NativeLibraryNestedSetBuilder addJavaTargets(
      Iterable<? extends TransitiveInfoCollection> deps) {
    for (TransitiveInfoCollection dep : deps) {
      addJavaTarget(dep);
    }
    return this;
  }

  /** Include native Java libraries of a specified target into the nested set. */
  public NativeLibraryNestedSetBuilder addJavaTarget(TransitiveInfoCollection dep) {
    JavaNativeLibraryInfo javaProvider = dep.get(JavaNativeLibraryInfo.PROVIDER);
    if (javaProvider != null) {
      builder.addTransitive(javaProvider.getTransitiveJavaNativeLibraries());
      return this;
    }

    CcNativeLibraryProvider ccProvider = dep.get(CcNativeLibraryProvider.PROVIDER);
    if (ccProvider != null) {
      builder.addTransitive(ccProvider.getTransitiveCcNativeLibraries());
      return this;
    }

    addTarget(dep);

    return this;
  }

  /** Include native C/C++ libraries of specified dependencies into the nested set. */
  public NativeLibraryNestedSetBuilder addCcTargets(
      Iterable<? extends TransitiveInfoCollection> deps) {
    for (TransitiveInfoCollection dep : deps) {
      addCcTarget(dep);
    }
    return this;
  }

  /** Include native Java libraries of a specified target into the nested set. */
  private void addCcTarget(TransitiveInfoCollection dep) {
    CcNativeLibraryProvider provider = dep.get(CcNativeLibraryProvider.PROVIDER);
    if (provider != null) {
      builder.addTransitive(provider.getTransitiveCcNativeLibraries());
    } else {
      addTarget(dep);
    }
  }

  /** Include files and genrule artifacts. */
  private void addTarget(TransitiveInfoCollection dep) {
    for (Artifact artifact :
        FileType.filterList(
            dep.getProvider(FileProvider.class).getFilesToBuild().toList(),
            CppFileTypes.SHARED_LIBRARY)) {
      builder.add(
          LibraryToLink.builder()
              .setLibraryIdentifier(CcLinkingOutputs.libraryIdentifierOf(artifact))
              .setDynamicLibrary(artifact)
              .build());
    }
  }
}
