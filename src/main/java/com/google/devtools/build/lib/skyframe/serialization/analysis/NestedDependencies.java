// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.base.MoreObjects.toStringHelper;
import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.annotations.VisibleForTesting;
import java.util.Arrays;
import java.util.Collection;

/**
 * A representation of a recursively composable set of {@link FileSystemDependencies}.
 *
 * <p>This corresponds to a previously serialized {@link
 * com.google.devtools.build.lib.skyframe.AbstractNestedFileOpNodes} instance, but this
 * implementation is mostly decoupled from Bazel code.
 */
abstract sealed class NestedDependencies
    implements FileSystemDependencies, FileDependencyDeserializer.NestedDependenciesOrFuture
    permits NestedDependencies.AvailableNestedDependencies,
        NestedDependencies.MissingNestedDependencies {
  // While formally possible, we don't anticipate analysisDependencies being empty often.
  // `sources` could be frequently empty.
  static final FileDependencies[] EMPTY_SOURCES = new FileDependencies[0];

  static NestedDependencies from(
      FileSystemDependencies[] analysisDependencies, FileDependencies[] sources) {
    for (FileSystemDependencies dep : analysisDependencies) {
      if (dep.isMissingData()) {
        return new MissingNestedDependencies();
      }
    }
    for (FileDependencies dep : sources) {
      if (dep.isMissingData()) {
        return new MissingNestedDependencies();
      }
    }
    return new AvailableNestedDependencies(analysisDependencies, sources);
  }

  @VisibleForTesting
  static NestedDependencies from(
      Collection<? extends FileSystemDependencies> analysisDependencies,
      Collection<FileDependencies> sources) {
    return from(
        analysisDependencies.toArray(FileSystemDependencies[]::new),
        sources.toArray(FileDependencies[]::new));
  }

  static NestedDependencies newMissingInstance() {
    return new MissingNestedDependencies();
  }

  static final class AvailableNestedDependencies extends NestedDependencies {
    private final FileSystemDependencies[] analysisDependencies;
    private final FileDependencies[] sources;

    AvailableNestedDependencies(
        FileSystemDependencies[] analysisDependencies, FileDependencies[] sources) {
      checkArgument(
          analysisDependencies.length >= 1 || sources.length >= 1,
          "analysisDependencies and sources both empty");
      this.analysisDependencies = analysisDependencies;
      this.sources = sources;
    }

    @Override
    public boolean isMissingData() {
      return false;
    }

    int analysisDependenciesCount() {
      return analysisDependencies.length;
    }

    FileSystemDependencies getAnalysisDependency(int index) {
      return analysisDependencies[index];
    }

    int sourcesCount() {
      return sources.length;
    }

    FileDependencies getSource(int index) {
      return sources[index];
    }

    @Override
    public String toString() {
      return toStringHelper(this)
          .add("analysisDependencies", Arrays.asList(analysisDependencies))
          .add("sources", Arrays.asList(sources))
          .toString();
    }
  }

  /**
   * Signals missing data in the nested set of dependencies.
   *
   * <p>This is deliberately not a singleton to avoid a memory leak in the weak-value caches in
   * {@link FileDependencyDeserializer}.
   */
  static final class MissingNestedDependencies extends NestedDependencies {
    private MissingNestedDependencies() {}

    @Override
    public boolean isMissingData() {
      return true;
    }
  }
}
