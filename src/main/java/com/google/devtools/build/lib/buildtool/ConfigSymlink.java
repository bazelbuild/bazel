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

package com.google.devtools.build.lib.buildtool;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.SymlinkDefinition;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Set;
import java.util.function.Function;

/** Base class for symlinks to output roots. Used only by {@link OutputDirectoryLinksUtils}. */
class ConfigSymlink implements SymlinkDefinition {
  @FunctionalInterface
  interface ConfigPathGetter {
    ArtifactRoot apply(BuildConfigurationValue configuration, RepositoryName repositoryName);
  }

  private final String suffix;
  private final ConfigPathGetter configToRoot;

  ConfigSymlink(String suffix, ConfigPathGetter configToRoot) {
    this.suffix = suffix;
    this.configToRoot = configToRoot;
  }

  @Override
  public String getLinkName(String symlinkPrefix, String workspaceBaseName) {
    return symlinkPrefix + suffix;
  }

  @Override
  public ImmutableSet<Path> getLinkPaths(
      PackageOptions packageOptions,
      Set<BuildConfigurationValue> targetConfigs,
      Function<BuildOptions, BuildConfigurationValue> configGetter,
      RepositoryName repositoryName,
      Path outputPath,
      Path execRoot) {
    return targetConfigs.stream()
        .map(config -> configToRoot.apply(config, repositoryName).getRoot().asPath())
        .distinct()
        .collect(toImmutableSet());
  }
}
