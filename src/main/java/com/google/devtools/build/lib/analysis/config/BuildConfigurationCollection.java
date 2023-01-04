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
 * Convenience container for top-level target configuration.
 *
 * <p>The target configuration is used for all targets specified on the command line.
 */
// TODO(b/253313672): replace this class with just a BuildConfigurationValue (hostConfig is gone)
@ThreadSafe
public final class BuildConfigurationCollection {
  private final BuildConfigurationValue targetConfiguration;

  public BuildConfigurationCollection(BuildConfigurationValue targetConfiguration)
      throws InvalidConfigurationException {
    this.targetConfiguration = targetConfiguration;
  }

  public BuildConfigurationValue getTargetConfiguration() {
    return targetConfiguration;
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
        .toString();
  }
}
