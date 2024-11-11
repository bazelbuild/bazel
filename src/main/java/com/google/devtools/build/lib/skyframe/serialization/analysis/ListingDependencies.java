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

import com.google.common.collect.ImmutableSet;

/** Type representing a directory listing operation. */
final class ListingDependencies
    implements FileDependencyDeserializer.GetListingDependenciesResult, FileSystemDependencies {
  private final FileDependencies realDirectory;

  ListingDependencies(FileDependencies realDirectory) {
    this.realDirectory = realDirectory;
  }

  /** True if this entry matches any directory name in {@code directoryPaths}. */
  boolean matchesAnyDirectory(ImmutableSet<String> directoryPaths) {
    return directoryPaths.contains(realDirectory.resolvedPath());
  }

  FileDependencies realDirectory() {
    return realDirectory;
  }

  @Override
  public String toString() {
    return toStringHelper(this).add("realDirectory", realDirectory).toString();
  }
}
