// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.buildeventstream.ProgressEvent;
import com.google.devtools.build.lib.util.ProcessUtils;
import java.util.ArrayList;
import java.util.Collection;

/** This event raised to indicate that no build will be happening for the given command. */
public final class NoBuildEvent implements BuildEvent {
  private final String id;
  private final String command;
  private final Long startTimeMillis;
  private final boolean separateFinishedEvent;
  private final boolean showProgress;
  private final ImmutableList<BuildEventId> additionalChildrenEvents;

  private NoBuildEvent(
      String command,
      Long startTimeMillis,
      boolean separateFinishedEvent,
      boolean showProgress,
      String id,
      ImmutableList<BuildEventId> additionalChildrenEvents) {
    this.command = command;
    this.startTimeMillis = startTimeMillis;
    this.separateFinishedEvent = separateFinishedEvent;
    this.showProgress = showProgress;
    this.id = id;
    this.additionalChildrenEvents = additionalChildrenEvents;
  }

  public static Builder newBuilder() {
    return new Builder();
  }

  /** Builder for {@link NoBuildEvent}. */
  public static class Builder {
    private String command = null;
    private String id = null;
    private boolean showProgress = false;
    private Long startTimeMillis = null;
    private boolean separateFinishedEvent = false;
    private ArrayList<BuildEventId> additionalChildrenEvents = new ArrayList<>();

    private Builder() {
    }

    public Builder setCommand(String command) {
      this.command = command;
      return this;
    }

    public Builder setId(String id) {
      this.id = id;
      return this;
    }

    public Builder setShowProgress(boolean showProgress) {
      this.showProgress = showProgress;
      return this;
    }

    public Builder setStartTimeMillis(long startTimeMillis) {
      this.startTimeMillis = startTimeMillis;
      return this;
    }

    public Builder setSeparateFinishedEvent(boolean separateFinishedEvent) {
      this.separateFinishedEvent = separateFinishedEvent;
      return this;
    }

    public Builder addAdditionalChildrenEvents(Iterable<BuildEventId> additionalChildrenEvents) {
      additionalChildrenEvents.forEach(this.additionalChildrenEvents::add);
      return this;
    }

    public NoBuildEvent build() {
      return new NoBuildEvent(
          command,
          startTimeMillis,
          separateFinishedEvent,
          showProgress,
          id,
          ImmutableList.copyOf(additionalChildrenEvents));
    }
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    ImmutableList.Builder allChildrenEventsBuilder = ImmutableList.builder();
    allChildrenEventsBuilder.add(ProgressEvent.INITIAL_PROGRESS_UPDATE);
    allChildrenEventsBuilder.addAll(additionalChildrenEvents);
    if (separateFinishedEvent) {
      allChildrenEventsBuilder.add(BuildEventId.buildFinished());
    }
    return allChildrenEventsBuilder.build();
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventId.buildStartedId();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    BuildEventStreamProtos.BuildStarted.Builder started =
        BuildEventStreamProtos.BuildStarted.newBuilder()
            .setBuildToolVersion(BlazeVersionInfo.instance().getVersion());
    if (command != null) {
      started.setCommand(command);
    }
    if (startTimeMillis != null) {
      started.setStartTimeMillis(startTimeMillis);
    }
    if (id != null) {
      started.setUuid(id);
    }
    started.setServerPid(ProcessUtils.getpid());
    return GenericBuildEvent.protoChaining(this).setStarted(started.build()).build();
  }

  public boolean separateFinishedEvent() {
    return separateFinishedEvent;
  }

  public boolean showProgress() {
    return showProgress;
  }
}
