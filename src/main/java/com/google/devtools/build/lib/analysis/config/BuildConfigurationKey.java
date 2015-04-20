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
package com.google.devtools.build.lib.analysis.config;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.syntax.Label;

import java.util.Objects;
import java.util.Set;

/**
 * A key for the creation of {@link BuildConfigurationCollection} instances.
 */
public final class BuildConfigurationKey {

  private final BuildOptions buildOptions;
  private final BlazeDirectories directories;
  private final ImmutableSortedSet<String> multiCpu;

  /**
   * Creates a key for the creation of {@link BuildConfigurationCollection} instances.
   *
   * Note that the BuildConfiguration.Options instance must not contain unresolved relative paths.
   */
  public BuildConfigurationKey(
      BuildOptions buildOptions, BlazeDirectories directories, Set<String> multiCpu) {
    this.buildOptions = Preconditions.checkNotNull(buildOptions);
    this.directories = Preconditions.checkNotNull(directories);
    this.multiCpu = ImmutableSortedSet.copyOf(multiCpu);
  }

  public BuildConfigurationKey(BuildOptions buildOptions, BlazeDirectories directories) {
    this(buildOptions, directories, ImmutableSet.<String>of());
  }

  public BuildOptions getBuildOptions() {
    return buildOptions;
  }

  public BlazeDirectories getDirectories() {
    return directories;
  }

  public ImmutableSortedSet<String> getMultiCpu() {
    return multiCpu;
  }

  public ListMultimap<String, Label> getLabelsToLoadUnconditionally() {
    return buildOptions.getAllLabels();
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof BuildConfigurationKey)) {
      return false;
    }
    BuildConfigurationKey k = (BuildConfigurationKey) o;
    return buildOptions.equals(k.buildOptions)
        && directories.equals(k.directories)
        && multiCpu.equals(k.multiCpu);
  }

  @Override
  public int hashCode() {
    return Objects.hash(buildOptions, directories, multiCpu);
  }
}
