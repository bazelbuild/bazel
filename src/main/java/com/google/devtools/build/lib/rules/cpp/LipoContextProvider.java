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
package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Map;

/**
 * Provides LIPO context information to the LIPO-enabled target configuration.
 *
 * <p>This is a rollup of the data collected in the LIPO context collector configuration.
 * Each target in the LIPO context collector configuration has a {@link TransitiveLipoInfoProvider}
 * which is used to transitively collect the data, then the {@code cc_binary} that is referred to
 * in {@code --lipo_context} puts the collected data into {@link LipoContextProvider}, of which
 * there is only one in any given build.
 */
@Immutable
public final class LipoContextProvider implements TransitiveInfoProvider {

  private final CppCompilationContext cppCompilationContext;

  private final ImmutableMap<Artifact, IncludeScannable> includeScannables;
  private final ImmutableMap<PathFragment, Artifact> sourceArtifactMap;

  public LipoContextProvider(CppCompilationContext cppCompilationContext,
      Map<Artifact, IncludeScannable> scannables,
      Map<PathFragment, Artifact> sourceArtifactMap) {
    this.cppCompilationContext = cppCompilationContext;
    this.includeScannables = ImmutableMap.copyOf(scannables);
    this.sourceArtifactMap = ImmutableMap.copyOf(sourceArtifactMap);
  }

  /**
   * Returns merged compilation context for the whole LIPO subtree.
   */
  public CppCompilationContext getLipoContext() {
    return cppCompilationContext;
  }

  /**
   * Returns the map from source artifact to the include scannable object representing
   * the corresponding FDO source input file.
   */
  public ImmutableMap<Artifact, IncludeScannable> getIncludeScannables() {
    return includeScannables;
  }

  public ImmutableMap<PathFragment, Artifact> getSourceArtifactMap() {
    return sourceArtifactMap;
  }
}
