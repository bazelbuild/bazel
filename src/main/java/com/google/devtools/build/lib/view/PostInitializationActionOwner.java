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
package com.google.devtools.build.lib.view;

import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.config.BuildConfiguration;

/**
 * An {@link ActionOwner} that is created after the analysis of the corresponding configured
 * target is finished.
 */
public final class PostInitializationActionOwner implements ActionOwner {
  private final ConfiguredTarget realOwner;

  public PostInitializationActionOwner(ConfiguredTarget realOwner) {
    this.realOwner = realOwner;
  }

  @Override
  public Location getLocation() {
    return realOwner.getTarget().getLocation();
  }

  @Override
  public Label getLabel() {
    return realOwner.getLabel();
  }

  @Override
  public String getConfigurationName() {
    BuildConfiguration conf = realOwner.getConfiguration();
    return (conf == null ? "null" : conf.getShortName());
  }

  @Override
  public String getConfigurationMnemonic() {
    BuildConfiguration conf = realOwner.getConfiguration();
    return (conf == null ? "null" : conf.getMnemonic());
  }

  @Override
  public String getConfigurationShortCacheKey() {
    BuildConfiguration conf = realOwner.getConfiguration();
    return (conf == null ? "null" : conf.shortCacheKey());
  }

  @Override
  public String getTargetKind() {
    return realOwner.getTarget().getTargetKind();
  }

  @Override
  public String getAdditionalProgressInfo() {
    return realOwner.getConfiguration().isHostConfiguration() ? "for host" : null;
  }
}
