// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.skyframe.WorkspaceInfoFromDiff;

/**
 * An event that tracks the cl difference between the current and previous evaluating version
 * provided by {@link WorkspaceInfoFromDiff#getEvaluatingVersion()}.
 */
public class BaselineClDiffEvent implements Postable {

  private final long baselineClDiff;

  public BaselineClDiffEvent(long baselineClDiff) {
    this.baselineClDiff = baselineClDiff;
  }

  public long getBaselineClDiff() {
    return baselineClDiff;
  }
}
