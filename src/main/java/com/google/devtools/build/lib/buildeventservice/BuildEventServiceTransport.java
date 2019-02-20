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


import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.eventbus.EventBus;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.JavaSleeper;
import com.google.devtools.build.lib.util.Sleeper;
import java.time.Duration;

/** A {@link BuildEventTransport} that streams {@link BuildEvent}s to BuildEventService. */
public class BuildEventServiceTransport implements BuildEventTransport {
  private final BuildEventServiceUploader besUploader;

  /** A builder for {@link BuildEventServiceTransport}. */
  public static class Builder {
    private boolean publishLifecycleEvents;
    private Duration closeTimeout;
    private Sleeper sleeper;
    private EventBus eventBus;

    /** Whether to publish lifecycle events. */
    public Builder publishLifecycleEvents(boolean publishLifecycleEvents) {
      this.publishLifecycleEvents = publishLifecycleEvents;
      return this;
    }

    /** The time to wait for the build event upload after the build has completed. */
    public Builder closeTimeout(Duration closeTimeout) {
      this.closeTimeout = closeTimeout;
      return this;
    }

    @VisibleForTesting
    public Builder sleeper(Sleeper sleeper) {
      this.sleeper = sleeper;
      return this;
    }

    public Builder setEventBus(EventBus eventBus) {
      this.eventBus = eventBus;
      return this;
    }

    public BuildEventServiceTransport build(
        BuildEventServiceClient besClient,
        BuildEventArtifactUploader localFileUploader,
        BuildEventProtocolOptions bepOptions,
        BuildEventServiceProtoUtil besProtoUtil,
        Clock clock,
        ExitFunction exitFunction,
        ArtifactGroupNamer namer) {
      Preconditions.checkNotNull(eventBus);
      return new BuildEventServiceTransport(
          besClient,
          localFileUploader,
          bepOptions,
          besProtoUtil,
          clock,
          exitFunction,
          publishLifecycleEvents,
          closeTimeout != null ? closeTimeout : Duration.ZERO,
          sleeper != null ? sleeper : new JavaSleeper(),
          namer,
          eventBus);
    }
  }

  private BuildEventServiceTransport(
      BuildEventServiceClient besClient,
      BuildEventArtifactUploader localFileUploader,
      BuildEventProtocolOptions bepOptions,
      BuildEventServiceProtoUtil besProtoUtil,
      Clock clock,
      ExitFunction exitFunc,
      boolean publishLifecycleEvents,
      Duration closeTimeout,
      Sleeper sleeper,
      ArtifactGroupNamer namer,
      EventBus eventBus) {
    this.besUploader =
        new BuildEventServiceUploader(
            besClient,
            localFileUploader,
            besProtoUtil,
            bepOptions,
            publishLifecycleEvents,
            closeTimeout,
            exitFunc,
            sleeper,
            clock,
            namer,
            eventBus);
  }

  @Override
  public ListenableFuture<Void> close() {
    // This future completes once the upload has finished. As
    // per API contract it is expected to never fail.
    SettableFuture<Void> closeFuture = SettableFuture.create();
    ListenableFuture<Void> uploaderCloseFuture = besUploader.close();
    uploaderCloseFuture.addListener(() -> closeFuture.set(null), MoreExecutors.directExecutor());
    return closeFuture;
  }

  @Override
  public BuildEventArtifactUploader getUploader() {
    return besUploader.getLocalFileUploader();
  }

  @Override
  public String name() {
    return "Build Event Service";
  }

  @Override
  public void sendBuildEvent(BuildEvent event) {
    besUploader.enqueueEvent(event);
  }

  @VisibleForTesting
  public BuildEventServiceUploader getBesUploader() {
    return besUploader;
  }

  /**
   * Called by the {@link BuildEventServiceUploader} in case of error to asynchronously notify Bazel
   * of an error.
   */
  @FunctionalInterface
  public interface ExitFunction {
    void accept(String message, Throwable cause, ExitCode code);
  }
}
