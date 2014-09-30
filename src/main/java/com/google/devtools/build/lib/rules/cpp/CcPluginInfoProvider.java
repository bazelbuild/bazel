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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.TransitiveInfoProvider;

/**
 * Provider of C plugin information.
 */
@Immutable
public final class CcPluginInfoProvider implements TransitiveInfoProvider {

  private final Label label;
  private final String targetType;
  private final Artifact dso;
  private final Artifact aggregatingRunfilesMiddleman;
  private final ImmutableList<String> userCopts;

  public CcPluginInfoProvider(Label label, String targetType, Artifact dso,
      Artifact aggregatingRunfilesMiddleman, ImmutableList<String> userCopts) {
    this.label = label;
    this.targetType = targetType;
    this.dso = dso;
    this.aggregatingRunfilesMiddleman = aggregatingRunfilesMiddleman;
    this.userCopts = userCopts;
  }

  /**
   * Returns the label that is associated with this piece of information.
   *
   * <p>This is usually the label of the target that provides the information.
   */
  public Label getLabel() {
    return label;
  }

  /**
   * Returns the type of the plugin target (e.g. "mao" or "gcc").
   */
  public String getTargetType() {
    return targetType;
  }

  /**
   * Returns the dynamic library artifact.
   */
  public Artifact getDso() {
    return dso;
  }

  /**
   * Returns an aggregating middleman with the runfiles required by this plugin.
   */
  public Artifact getAggregatingRunfilesMiddleman() {
    return aggregatingRunfilesMiddleman;
  }

  /**
   * Returns the tokenized copts for this plugin.
   */
  public ImmutableList<String> getUserCopts() {
    return userCopts;
  }
}
