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
package com.google.devtools.build.lib.analysis.test;

import com.google.common.base.MoreObjects;
import com.google.common.base.Objects;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ConfigurationId;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import javax.annotation.Nullable;

/** This event may be raised while a test action is executing to report info about its execution. */
@Immutable
public final class TestProgress implements BuildEvent {
  /** The label of the target for the action. */
  private final String label;

  /** The configuration under which the action is running. */
  private final BuildEventId.ConfigurationId configId;

  /** The run number of the test action (e.g. for runs_per_test > 1). */
  private final int run;

  /** For sharded tests, the shard number of the test action. */
  private final int shard;

  /** The execution attempt number which may increase due to retries. */
  private final int attempt;

  /** A count which may be incremented to differentiate events. */
  private final int opaqueCount;

  /** Identifies a resource that can provide info about the active test run. */
  private final String uri;

  public TestProgress(
      String label,
      ConfigurationId configId,
      int run,
      int shard,
      int attempt,
      int opaqueCount,
      String uri) {
    this.label = label;
    this.configId = configId;
    this.run = run;
    this.shard = shard;
    this.attempt = attempt;
    this.opaqueCount = opaqueCount;
    this.uri = uri;
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventIdUtil.testProgressId(label, configId, run, shard, attempt, opaqueCount);
  }

  @Override
  public ImmutableList<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    return GenericBuildEvent.protoChaining(this).setTestProgress(asTestResult()).build();
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(label, configId, run, shard, attempt, opaqueCount, uri);
  }

  @Override
  public boolean equals(@Nullable Object object) {
    if (!(object instanceof TestProgress other)) {
      return false;
    }
    return label.equals(other.label)
        && configId.equals(other.configId)
        && run == other.run
        && shard == other.shard
        && attempt == other.attempt
        && opaqueCount == other.opaqueCount
        && uri.equals(other.uri);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("label", label)
        .add("configId", configId)
        .add("run", run)
        .add("shard", shard)
        .add("attempt", attempt)
        .add("opaqueCount", opaqueCount)
        .add("uri", uri)
        .toString();
  }

  private BuildEventStreamProtos.TestProgress asTestResult() {
    return BuildEventStreamProtos.TestProgress.newBuilder().setUri(uri).build();
  }
}
