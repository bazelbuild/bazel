// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.buildtool.buildevent;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildCompletingEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.util.ExitCode;

/**
 * Event triggered after testing has completed.
 *
 * <p>This event is used by the BEP to construct the {@link
 * com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildFinished} event when
 * the test command is used.
 */
public class TestingCompleteEvent extends BuildCompletingEvent {
  /**
   * Creates a new {@link TestingCompleteEvent}.
   *
   * @param exitCode the overall exit code of "bazel test".
   * @param finishTimeMillis the finish time in milliseconds since the epoch.
   * @param wasSuspended was the build suspended at any point.
   */
  public TestingCompleteEvent(ExitCode exitCode, long finishTimeMillis, boolean wasSuspended) {
    super(
        exitCode,
        finishTimeMillis,
        ImmutableList.of(BuildEventIdUtil.buildToolLogs()),
        wasSuspended);
  }
}
