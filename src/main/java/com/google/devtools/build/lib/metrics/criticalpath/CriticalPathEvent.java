package com.google.devtools.build.lib.metrics.criticalpath;

import build.bazel.bep.CriticalPath;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithOrderConstraint;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import java.time.Duration;
import java.util.Collection;

/** {@code Build event protocol} event for the {@code critical path} of a build. */
public final class CriticalPathEvent extends GenericBuildEvent
    implements BuildEventWithOrderConstraint {
  public static final BuildEventId BEP_ID =
      BuildEventId.newBuilder().setCriticalPath(CriticalPath.BepId.getDefaultInstance()).build();

  private final Duration totalTime;

  CriticalPathEvent(Duration totalTime) {
    super(BEP_ID, ImmutableList.of());

    this.totalTime = Preconditions.checkNotNull(totalTime);
  }

  @Override
  public Collection<BuildEventId> postedAfter() {
    return ImmutableList.of(BuildEventIdUtil.buildFinished());
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    return GenericBuildEvent.protoChaining(this)
        .setCriticalPath(
            CriticalPath.newBuilder()
                .setTotalTime(
                    com.google.protobuf.Duration.newBuilder()
                        .setSeconds(totalTime.getSeconds())
                        .setNanos(totalTime.getNano())
                        .build())
                .build())
        .build();
  }
}
