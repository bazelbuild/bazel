// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import javax.annotation.Nullable;

/** A Bazel invocation requested a {@link BuildConfigurationValue}. */
public class ConfigRequestedEvent implements Postable {
  private final BuildConfigurationValue config;

  @Nullable private final String parentChecksum;

  /**
   * Requested configuration event.
   *
   * @param config the configuration
   * @param parentChecksum the configuration that produces this one from a transition. If null, the
   *     current configuration is top-level.
   */
  public ConfigRequestedEvent(BuildConfigurationValue config, @Nullable String parentChecksum) {
    this.config = config;
    this.parentChecksum = parentChecksum;
  }

  public BuildConfigurationValue getConfig() {
    return config;
  }

  @Nullable
  public String getParentChecksum() {
    return parentChecksum;
  }
}
