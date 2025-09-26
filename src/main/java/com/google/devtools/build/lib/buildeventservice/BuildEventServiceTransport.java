// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.buildeventservice;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.eventbus.EventBus;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.buildeventservice.BuildEventServiceOptions.BesUploadMode;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient.CommandContext;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.util.JavaSleeper;
import com.google.devtools.build.lib.util.Sleeper;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.time.Duration;
import java.time.Instant;
import javax.annotation.Nullable;

/** A {@link BuildEventTransport} that streams {@link BuildEvent}s to BuildEventService. */
public class BuildEventServiceTransport implements BuildEventTransport {
  private final BuildEventServiceUploader besUploader;
  private final Duration besTimeout;
  private final BesUploadMode besUploadMode;

  private BuildEventServiceTransport(
      BuildEventServiceClient besClient,
      BuildEventArtifactUploader localFileUploader,
      BuildEventProtocolOptions bepOptions,
      Clock clock,
      boolean publishLifecycleEvents,
      ArtifactGroupNamer artifactGroupNamer,
      EventBus eventBus,
      Duration closeTimeout,
      Sleeper sleeper,
      CommandContext commandContext,
      Instant commandStartTime,
      BesUploadMode besUploadMode) {
    this.besTimeout = closeTimeout;
    this.besUploader =
        new BuildEventServiceUploader.Builder()
            .besClient(besClient)
            .localFileUploader(localFileUploader)
            .bepOptions(bepOptions)
            .clock(clock)
            .publishLifecycleEvents(publishLifecycleEvents)
            .sleeper(sleeper)
            .artifactGroupNamer(artifactGroupNamer)
            .eventBus(eventBus)
            .commandContext(commandContext)
            .commandStartTime(commandStartTime)
            .build();
    this.besUploadMode = besUploadMode;
  }

  @Override
  public ListenableFuture<Void> close() {
    return besUploader.close();
  }

  @Override
  public ListenableFuture<Void> getHalfCloseFuture() {
    return besUploader.getHalfCloseFuture();
  }

  @Override
  public BuildEventArtifactUploader getUploader() {
    return besUploader.getBuildEventUploader();
  }

  @Override
  public String name() {
    return "Build Event Service";
  }

  @Override
  public boolean mayBeSlow() {
    return true;
  }

  @Override
  public BesUploadMode getBesUploadMode() {
    return besUploadMode;
  }

  @Override
  public void sendBuildEvent(BuildEvent event) {
    besUploader.enqueueEvent(event);
  }

  @Override
  public Duration getTimeout() {
    return besTimeout;
  }

  /** A builder for {@link BuildEventServiceTransport}. */
  public static class Builder {
    private BuildEventServiceClient besClient;
    private BuildEventArtifactUploader localFileUploader;
    private BuildEventServiceOptions besOptions;
    private BuildEventProtocolOptions bepOptions;
    private Clock clock;
    private ArtifactGroupNamer artifactGroupNamer;
    private EventBus eventBus;
    @Nullable private Sleeper sleeper;
    private CommandContext commandContext;
    private Instant commandStartTime;

    @CanIgnoreReturnValue
    public Builder besClient(BuildEventServiceClient value) {
      this.besClient = value;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder localFileUploader(BuildEventArtifactUploader value) {
      this.localFileUploader = value;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder bepOptions(BuildEventProtocolOptions value) {
      this.bepOptions = value;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder besOptions(BuildEventServiceOptions value) {
      this.besOptions = value;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder clock(Clock value) {
      this.clock = value;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder artifactGroupNamer(ArtifactGroupNamer value) {
      this.artifactGroupNamer = value;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder eventBus(EventBus value) {
      this.eventBus = value;
      return this;
    }

    @CanIgnoreReturnValue
    @VisibleForTesting
    public Builder sleeper(Sleeper value) {
      this.sleeper = value;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder commandContext(CommandContext value) {
      this.commandContext = value;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder commandStartTime(Instant value) {
      this.commandStartTime = value;
      return this;
    }

    public BuildEventServiceTransport build() {
      checkNotNull(besOptions);
      return new BuildEventServiceTransport(
          checkNotNull(besClient),
          checkNotNull(localFileUploader),
          checkNotNull(bepOptions),
          checkNotNull(clock),
          besOptions.besLifecycleEvents,
          checkNotNull(artifactGroupNamer),
          checkNotNull(eventBus),
          (besOptions.besTimeout != null) ? besOptions.besTimeout : Duration.ZERO,
          sleeper != null ? sleeper : new JavaSleeper(),
          checkNotNull(commandContext),
          checkNotNull(commandStartTime),
          besOptions.besUploadMode);
    }
  }
}
