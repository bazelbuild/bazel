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

package com.google.devtools.build.lib.ideinfo;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.android.AndroidIdeInfoProvider;
import com.google.devtools.build.lib.rules.android.AndroidIdeInfoProvider.SourceDirectory;

/**
 * File provider for Android Studio ide build files.
 */
@Immutable
public final class AndroidStudioInfoFilesProvider implements TransitiveInfoProvider {
  private final NestedSet<Artifact> ideBuildFiles;
  private final NestedSet<Label> transitiveDependencies;
  private final NestedSet<Label> exportedDeps;
  private final NestedSet<AndroidIdeInfoProvider.SourceDirectory> transitiveResources;

  /**
   * Builder class for {@link AndroidStudioInfoFilesProvider}
   */
  public static class Builder {
    private final NestedSetBuilder<Artifact> ideBuildFilesBuilder;
    private final NestedSetBuilder<Label> transitiveDependenciesBuilder;
    private NestedSetBuilder<Label> exportedDepsBuilder;
    private NestedSetBuilder<AndroidIdeInfoProvider.SourceDirectory> transitiveResourcesBuilder;
    private NestedSet<AndroidIdeInfoProvider.SourceDirectory> transitiveResources;

    public Builder() {
      ideBuildFilesBuilder = NestedSetBuilder.stableOrder();
      transitiveDependenciesBuilder = NestedSetBuilder.stableOrder();
      exportedDepsBuilder = NestedSetBuilder.stableOrder();
      transitiveResourcesBuilder = NestedSetBuilder.stableOrder();
      transitiveResources = null;
    }

    public NestedSetBuilder<Artifact> ideBuildFilesBuilder() {
      return ideBuildFilesBuilder;
    }

    public NestedSetBuilder<Label> transitiveDependenciesBuilder() {
      return transitiveDependenciesBuilder;
    }

    public NestedSetBuilder<Label> exportedDepsBuilder() {
      return exportedDepsBuilder;
    }

    public NestedSetBuilder<SourceDirectory> transitiveResourcesBuilder() {
      return transitiveResourcesBuilder;
    }

    /**
     * Returns a set of transitive resources. {@link Builder#transitiveResourcesBuilder}
     * is unusable after this operation.
     */
    public NestedSet<AndroidIdeInfoProvider.SourceDirectory> getTransitiveResources() {
      if (transitiveResources != null) {
        return transitiveResources;
      }
      transitiveResources = transitiveResourcesBuilder.build();
      transitiveResourcesBuilder = null;
      return transitiveResources;
    }

    public AndroidStudioInfoFilesProvider build() {
      return new AndroidStudioInfoFilesProvider(
          ideBuildFilesBuilder.build(),
          transitiveDependenciesBuilder.build(),
          exportedDepsBuilder.build(),
          getTransitiveResources()
      );
    }
  }

  private AndroidStudioInfoFilesProvider(
      NestedSet<Artifact> ideBuildFiles,
      NestedSet<Label> transitiveDependencies,
      NestedSet<Label> exportedDeps,
      NestedSet<SourceDirectory> transitiveResources) {
    this.ideBuildFiles = ideBuildFiles;
    this.transitiveDependencies = transitiveDependencies;
    this.exportedDeps = exportedDeps;
    this.transitiveResources = transitiveResources;
  }

  public NestedSet<Artifact> getIdeBuildFiles() {
    return ideBuildFiles;
  }

  public NestedSet<Label> getTransitiveDependencies() {
    return transitiveDependencies;
  }

  public NestedSet<Label> getExportedDeps() {
    return exportedDeps;
  }

  public NestedSet<SourceDirectory> getTransitiveResources() {
    return transitiveResources;
  }
}
