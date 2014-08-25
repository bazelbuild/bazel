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
package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.TransitiveInfoProvider;

import java.util.NavigableMap;

/**
 * Provides LIPO context information. It's implemented by the cc_binary ConfiguredTarget
 * that's specified at command-line as '--lipo_context=...', which is the same cc_binary
 * that generated the LIPO/FDO profile.
 */
@Immutable
public final class LipoContextProvider implements TransitiveInfoProvider {

  private final CppCompilationContext cppCompilationContext;

  private final ImmutableSortedMap<PathFragment, Label> pathsToLabels;

  public LipoContextProvider(CppCompilationContext cppCompilationContext,
      NavigableMap<PathFragment, Label> pathsToLabels) {
    this.cppCompilationContext = cppCompilationContext;
    this.pathsToLabels = ImmutableSortedMap.copyOf(pathsToLabels);
  }

  /**
   * Returns merged compilation context for the whole LIPO subtree.
   */
  public CppCompilationContext getLipoContext() {
    return cppCompilationContext;
  }

  /**
   * Returns a map of target directories to LipoInfos (targets)
   */
  public ImmutableSortedMap<PathFragment, Label> getPathsToLabels() {
    return pathsToLabels;
  }
}
