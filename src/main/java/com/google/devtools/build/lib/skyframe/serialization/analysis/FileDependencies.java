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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.ArrayList;

/**
 * Representation of a set of file names that could invalidate a given value.
 *
 * <p>Most values can be associated with some set of input files, represented in this nested way to
 * facilitate sharing between values. So given a set of changed files, invalidation is performed by
 * calling {@link #containsMatch} on an instance and all transitively reachable instances via {@link
 * #getDependencyCount} and {@link #getDependency}. If any matches are encountered, the associated
 * value is invalidated.
 */
abstract sealed class FileDependencies
    implements FileSystemDependencies, FileDependencyDeserializer.GetFileDependenciesResult
    permits FileDependencies.SingleResolvedPath,
        FileDependencies.SingleResolvedPathAndDependency,
        FileDependencies.MultiplePaths {

  abstract boolean containsMatch(ImmutableSet<String> paths);

  abstract int getDependencyCount();

  abstract FileDependencies getDependency(int index);

  /**
   * The real path associated with this node after resolution.
   *
   * <p>This is used by {@link FileDependencyDeserializer} to retrieve resolved parent paths but
   * isn't directly used by invalidation.
   */
  abstract String resolvedPath();

  /** Returns the resolved paths associated with the current node for testing. */
  @VisibleForTesting
  abstract ImmutableList<String> getAllResolvedPathsForTesting();

  static Builder builder(String firstResolvedPath) {
    return new Builder(firstResolvedPath);
  }

  static class Builder {
    private final ArrayList<String> paths = new ArrayList<>();
    private final ArrayList<FileDependencies> dependencies = new ArrayList<>();

    /**
     * At least one resolved path is required.
     *
     * <p>The last path added is treated as the overall {@link #resolvedPath} of the instance. The
     * {@code firstResolvedPath} argument is the {@link #resolvedPath} if it's the only path.
     */
    Builder(String firstResolvedPath) {
      paths.add(firstResolvedPath);
    }

    @CanIgnoreReturnValue
    Builder addPath(String path) {
      paths.add(path);
      return this;
    }

    @CanIgnoreReturnValue
    Builder addDependency(FileDependencies dependency) {
      dependencies.add(dependency);
      return this;
    }

    FileDependencies build() {
      if (paths.size() == 1) {
        int dependenciesSize = dependencies.size();
        if (dependenciesSize == 0) {
          return new SingleResolvedPath(paths.get(0));
        }
        if (dependenciesSize == 1) {
          return new SingleResolvedPathAndDependency(paths.get(0), dependencies.get(0));
        }
      }
      return new MultiplePaths(ImmutableList.copyOf(paths), ImmutableList.copyOf(dependencies));
    }
  }

  // The implementations here exist to reduce indirection and memory use.

  static final class SingleResolvedPath extends FileDependencies {
    private final String resolvedPath;

    private SingleResolvedPath(String resolvedPath) {
      this.resolvedPath = resolvedPath;
    }

    @Override
    boolean containsMatch(ImmutableSet<String> paths) {
      return paths.contains(resolvedPath);
    }

    @Override
    int getDependencyCount() {
      return 0;
    }

    @Override
    FileDependencies getDependency(int index) {
      throw new IndexOutOfBoundsException(this + " " + index);
    }

    @Override
    String resolvedPath() {
      return resolvedPath;
    }

    @Override
    ImmutableList<String> getAllResolvedPathsForTesting() {
      return ImmutableList.of(resolvedPath);
    }

    @Override
    public String toString() {
      return toStringHelper(this).add("resolvedPath", resolvedPath).toString();
    }
  }

  static final class SingleResolvedPathAndDependency extends FileDependencies {
    private final String resolvedPath;
    private final FileDependencies dependency;

    private SingleResolvedPathAndDependency(String resolvedPath, FileDependencies dependency) {
      this.resolvedPath = resolvedPath;
      this.dependency = dependency;
    }

    @Override
    boolean containsMatch(ImmutableSet<String> paths) {
      return paths.contains(resolvedPath);
    }

    @Override
    int getDependencyCount() {
      return 1;
    }

    @Override
    FileDependencies getDependency(int index) {
      if (index != 0) {
        throw new IndexOutOfBoundsException(this + " " + index);
      }
      return dependency;
    }

    @Override
    String resolvedPath() {
      return resolvedPath;
    }

    @Override
    ImmutableList<String> getAllResolvedPathsForTesting() {
      return ImmutableList.of(resolvedPath);
    }

    @Override
    public String toString() {
      return toStringHelper(this)
          .add("resolvedPath", resolvedPath)
          .add("dependency", dependency)
          .toString();
    }
  }

  static final class MultiplePaths extends FileDependencies {
    private final ImmutableList<String> resolvedPaths;
    private final ImmutableList<FileDependencies> dependencies;

    private MultiplePaths(
        ImmutableList<String> resolvedPaths, ImmutableList<FileDependencies> dependencies) {
      this.resolvedPaths = resolvedPaths;
      this.dependencies = dependencies;
    }

    @Override
    boolean containsMatch(ImmutableSet<String> paths) {
      for (int i = 0; i < resolvedPaths.size(); i++) {
        if (paths.contains(resolvedPaths.get(i))) {
          return true;
        }
      }
      return false;
    }

    @Override
    int getDependencyCount() {
      return dependencies.size();
    }

    @Override
    FileDependencies getDependency(int index) {
      return dependencies.get(index);
    }

    @Override
    String resolvedPath() {
      return Iterables.getLast(resolvedPaths);
    }

    @Override
    ImmutableList<String> getAllResolvedPathsForTesting() {
      return resolvedPaths;
    }

    @Override
    public String toString() {
      return toStringHelper(this)
          .add("resolvedPaths", resolvedPaths)
          .add("dependencies", dependencies)
          .toString();
    }
  }
}
