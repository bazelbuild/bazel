// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import java.time.Duration;

/**
 * This event is fired once the analysis cache has been cleared. Analysis cache clearing is
 * triggered at the beginning of execution phase if --discard_analysis_phase is set.
 */
public final class AnalysisCacheClearEvent {
  private final Duration clearTime;

  public AnalysisCacheClearEvent(Duration clearTime) {
    this.clearTime = clearTime;
  }

  public Duration getClearTime() {
    return clearTime;
  }
}
