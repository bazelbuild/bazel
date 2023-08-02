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
package com.google.devtools.build.lib.buildtool.buildevent;

import com.google.devtools.build.lib.buildeventstream.BuildCompletingEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.util.ExitCode;
import java.util.Collection;

/**
 * Event triggered during a run command after the requested binary has been built but before it has
 * been run.
 *
 * <p>This event is used by the BEP to construct the {@link
 * com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildFinished} event when
 * the run command is used.
 */
public class RunBuildCompleteEvent extends BuildCompletingEvent {

  public RunBuildCompleteEvent(
      ExitCode exitCode, long finishTimeMillis, Collection<BuildEventId> children) {
    super(exitCode, finishTimeMillis, children);
  }

  public RunBuildCompleteEvent(ExitCode exitCode, long finishTimeMillis) {
    super(exitCode, finishTimeMillis);
  }
}
