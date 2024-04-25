// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.config;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Set;
import java.util.function.Function;

/** Represents a single kind of convenience symlink ({@code bazel-bin}, etc.). */
public interface SymlinkDefinition {

  /**
   * Returns the name for this symlink in the workspace.
   *
   * <p>Note that this is independent of the target configuration(s) that may help determine the
   * symlink's destination.
   */
  String getLinkName(String symlinkPrefix, String workspaceBaseName);

  /**
   * Returns a set of candidate destination paths for the symlink.
   *
   * <p>The symlink should only be created if there is exactly one candidate. Zero candidates is a
   * no-op, and more than one candidate means a warning about ambiguous symlink destinations should
   * be emitted.
   *
   * @param packageOptions options that may control which symlinks get created and what they
   *     point to.
   * @param targetConfigs the configurations for which symlinks should be created. If these have
   *     conflicting requirements, multiple candidates are returned.
   * @param configGetter used to compute derived configurations, if needed. This is used for
   *     symlinks that link to the output directories of configs that are related to, but not
   *     included in, {@code targetConfigs}.
   * @param repositoryName the repository name.
   * @param outputPath the output path.
   * @param execRoot the exec root.
   */
  ImmutableSet<Path> getLinkPaths(
      PackageOptions packageOptions,
      Set<BuildConfigurationValue> targetConfigs,
      Function<BuildOptions, BuildConfigurationValue> configGetter,
      RepositoryName repositoryName,
      Path outputPath,
      Path execRoot);
}
