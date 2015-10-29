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

/**
 * File provider for Android Studio ide build files.
 */
@Immutable
public final class AndroidStudioInfoFilesProvider implements TransitiveInfoProvider {
  private final NestedSet<Artifact> ideInfoFiles;
  private final NestedSet<Artifact> ideInfoTextFiles;
  private final NestedSet<Artifact> ideResolveFiles;
  private final NestedSet<Label> exportedDeps;

  /**
   * Builder class for {@link AndroidStudioInfoFilesProvider}
   */
  public static class Builder {
    private final NestedSetBuilder<Artifact> ideInfoFilesBuilder;
    private final NestedSetBuilder<Artifact> ideInfoTextFilesBuilder;
    private final NestedSetBuilder<Artifact> ideResolveFilesBuilder;
    private NestedSetBuilder<Label> exportedDepsBuilder;

    public Builder() {
      ideInfoFilesBuilder = NestedSetBuilder.stableOrder();
      ideInfoTextFilesBuilder = NestedSetBuilder.stableOrder();
      ideResolveFilesBuilder = NestedSetBuilder.stableOrder();
      exportedDepsBuilder = NestedSetBuilder.stableOrder();
    }

    public NestedSetBuilder<Artifact> ideInfoFilesBuilder() {
      return ideInfoFilesBuilder;
    }

    public NestedSetBuilder<Artifact> ideInfoTextFilesBuilder() {
      return ideInfoTextFilesBuilder;
    }

    public NestedSetBuilder<Artifact> ideResolveFilesBuilder() {
      return ideResolveFilesBuilder;
    }

    public NestedSetBuilder<Label> exportedDepsBuilder() {
      return exportedDepsBuilder;
    }

    public AndroidStudioInfoFilesProvider build() {
      return new AndroidStudioInfoFilesProvider(
          ideInfoFilesBuilder.build(),
          ideInfoTextFilesBuilder.build(),
          ideResolveFilesBuilder.build(),
          exportedDepsBuilder.build()
      );
    }
  }

  private AndroidStudioInfoFilesProvider(
      NestedSet<Artifact> ideInfoFiles,
      NestedSet<Artifact> ideInfoTextFiles,
      NestedSet<Artifact> ideResolveFiles,
      NestedSet<Label> exportedDeps) {
    this.ideInfoFiles = ideInfoFiles;
    this.ideInfoTextFiles = ideInfoTextFiles;
    this.ideResolveFiles = ideResolveFiles;
    this.exportedDeps = exportedDeps;
  }

  public NestedSet<Artifact> getIdeInfoFiles() {
    return ideInfoFiles;
  }

  public NestedSet<Artifact> getIdeInfoTextFiles() {
    return ideInfoTextFiles;
  }

  public NestedSet<Artifact> getIdeResolveFiles() {
    return ideResolveFiles;
  }

  public NestedSet<Label> getExportedDeps() {
    return exportedDeps;
  }
}
