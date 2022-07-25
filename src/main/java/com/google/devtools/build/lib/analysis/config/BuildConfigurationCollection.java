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

package com.google.devtools.build.lib.analysis.config;

import com.google.common.base.MoreObjects;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

/**
 * Convenience container for top-level target and host configurations.
 *
 * <p>The target configuration is used for all targets specified on the command line.
 *
 * <p>The host configuration is used for tools that are executed during the build, e. g, compilers.
 */
@ThreadSafe
public final class BuildConfigurationCollection {
  private final BuildConfigurationValue targetConfiguration;
  private final BuildConfigurationValue hostConfiguration;

  public BuildConfigurationCollection(
      BuildConfigurationValue targetConfiguration, BuildConfigurationValue hostConfiguration)
      throws InvalidConfigurationException {
    this.targetConfiguration = targetConfiguration;
    this.hostConfiguration = hostConfiguration;
  }

  public BuildConfigurationValue getTargetConfiguration() {
    return targetConfiguration;
  }

  /**
   * Returns the host configuration for this collection.
   *
   * <p>Don't use this method. It's limited in that it assumes a single host configuration for the
   * entire collection. This may not be true in the future and more flexible interfaces will likely
   * supplant this interface anyway.
   */
  public BuildConfigurationValue getHostConfiguration() {
    return hostConfiguration;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof BuildConfigurationCollection)) {
      return false;
    }
    BuildConfigurationCollection that = (BuildConfigurationCollection) obj;
    return this.targetConfiguration.equals(that.targetConfiguration);
  }

  @Override
  public int hashCode() {
    return targetConfiguration.hashCode();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("targetConfiguration", targetConfiguration.checksum())
        .add("hostConfiguration", hostConfiguration.checksum())
        .toString();
  }
}
