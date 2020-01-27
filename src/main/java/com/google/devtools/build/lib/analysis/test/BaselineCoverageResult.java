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

package com.google.devtools.build.lib.analysis.test;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.events.ExtendedEventHandler;

/** This event is used to notify about a successfully built baseline coverage artifact. */
public class BaselineCoverageResult implements ExtendedEventHandler.ProgressLike {

  private final Artifact baselineCoverageData;
  private final String ownerString;

  BaselineCoverageResult(Artifact baselineCoverageData, String ownerString) {
    this.baselineCoverageData = Preconditions.checkNotNull(baselineCoverageData);
    this.ownerString = Preconditions.checkNotNull(ownerString);
  }

  public Artifact getArtifact() {
    return baselineCoverageData;
  }

  public String getOwnerString() {
    return ownerString;
  }
}
