// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.util.DetailedExitCode;

/**
 * This event is fired from BuildTool#stopRequest() just after {@link
 * com.google.devtools.build.lib.skyframe.SkyframeExecutor} calls notifyCommandComplete.
 */
public final class BuildToolFinalizingEvent {

  private final DetailedExitCode detailedExitCode;

  public BuildToolFinalizingEvent(DetailedExitCode detailedExitCode) {
    this.detailedExitCode = detailedExitCode;
  }

  // The DetailedExitCode from the build request.
  public DetailedExitCode getDetailedExitCode() {
    return detailedExitCode;
  }
}
