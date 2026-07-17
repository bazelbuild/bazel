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

package com.google.devtools.build.lib.buildeventstream;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.protobuf.util.Timestamps;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * Class all events completing a build inherit from.
 *
 * <p>This class is abstract as for each particular event a specialized class should be used.
 * However, subclasses do not have to implement anything.
 */
public abstract class BuildCompletingEvent implements BuildEvent {
  @Nullable private final DetailedExitCode detailedExitCode;
  private final ExitCode exitCode;
  private final long finishTimeMillis;

  private final Collection<BuildEventId> children;

  public BuildCompletingEvent(
      ExitCode exitCode, long finishTimeMillis, Collection<BuildEventId> children) {
    this.detailedExitCode = null;
    this.exitCode = exitCode;
    this.finishTimeMillis = finishTimeMillis;
    this.children = children;
  }

  public BuildCompletingEvent(
      DetailedExitCode detailedExitCode, long finishTimeMillis, Collection<BuildEventId> children) {
    this.detailedExitCode = detailedExitCode;
    this.exitCode = detailedExitCode.getExitCode();
    this.finishTimeMillis = finishTimeMillis;
    this.children = children;
  }

  public BuildCompletingEvent(ExitCode exitCode, long finishTimeMillis) {
    this(exitCode, finishTimeMillis, ImmutableList.of());
  }

  public ExitCode getExitCode() {
    return exitCode;
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventIdUtil.buildFinished();
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return children;
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    BuildEventStreamProtos.BuildFinished.ExitCode protoExitCode =
        BuildEventStreamProtos.BuildFinished.ExitCode.newBuilder()
            .setName(exitCode.name())
            .setCode(exitCode.getNumericExitCode())
            .build();

    BuildEventStreamProtos.BuildFinished.Builder finished =
        BuildEventStreamProtos.BuildFinished.newBuilder()
            .setOverallSuccess(ExitCode.SUCCESS.equals(exitCode))
            .setExitCode(protoExitCode)
            .setFinishTime(Timestamps.fromMillis(finishTimeMillis))
            .setFinishTimeMillis(finishTimeMillis);

    if (detailedExitCode != null && detailedExitCode.getFailureDetail() != null) {
      finished.setFailureDetail(detailedExitCode.getFailureDetail());
    }

    return GenericBuildEvent.protoChaining(this).setFinished(finished.build()).build();
  }
}
