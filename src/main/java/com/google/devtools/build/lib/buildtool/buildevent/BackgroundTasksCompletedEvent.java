package com.google.devtools.build.lib.buildtool.buildevent;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import java.util.Collection;

public record BackgroundTasksCompletedEvent() implements BuildEvent {
  @Override
  public BuildEventStreamProtos.BuildEventId getEventId() {
    return BuildEventIdUtil.backgroundTasksCompletedId();
  }

  @Override
  public Collection<BuildEventStreamProtos.BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext context)
      throws InterruptedException {
    return GenericBuildEvent.protoChaining(this).build();
  }
}
