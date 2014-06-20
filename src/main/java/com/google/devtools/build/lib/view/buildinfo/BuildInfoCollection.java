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
package com.google.devtools.build.lib.view.buildinfo;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;

import java.util.List;

/**
 * A collection of build-info files for both stamped and unstamped modes.
 */
public final class BuildInfoCollection {

  private final ImmutableList<Action> actions;
  private final List<Artifact> stampedBuildInfo;
  private final List<Artifact> redactedBuildInfo;

  public BuildInfoCollection(List<? extends Action> actions, List<Artifact> stampedBuildInfo,
      List<Artifact> redactedBuildInfo) {
    this.actions = ImmutableList.copyOf(actions);
    this.stampedBuildInfo = stampedBuildInfo;
    this.redactedBuildInfo = redactedBuildInfo;
  }

  public ImmutableList<Action> getActions() {
    return actions;
  }

  public List<Artifact> getStampedBuildInfo() {
    return stampedBuildInfo;
  }

  public List<Artifact> getRedactedBuildInfo() {
    return redactedBuildInfo;
  }
}
