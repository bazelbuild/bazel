// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import static com.google.common.util.concurrent.Futures.immediateVoidFuture;

import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.buildeventservice.BuildEventServiceOptions.BesUploadMode;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.buildeventstream.transports.BuildEventStreamOptions;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.Options;
import java.util.ArrayList;
import java.util.List;
import org.junit.Before;

/** Base class for BuildEventStreamer tests, sharing common setup and helpers. */
public abstract class BuildEventStreamerTestBase extends FoundationTestCase {

  protected static final String OOM_MESSAGE = "Please build fewer targets.";

  protected final CountingArtifactGroupNamer artifactGroupNamer = new CountingArtifactGroupNamer();
  protected final RecordingBuildEventTransport transport =
      new RecordingBuildEventTransport(artifactGroupNamer);

  protected BuildEventStreamer streamer;

  @Before
  public void setUpStreamer() throws Exception {
    streamer =
        new BuildEventStreamer.Builder()
            .artifactGroupNamer(artifactGroupNamer)
            .buildEventTransports(ImmutableSet.of(transport))
            .besStreamOptions(Options.getDefaults(BuildEventStreamOptions.class))
            .oomMessage(OOM_MESSAGE)
            .build();
  }

  protected static BuildEventContext getTestBuildEventContext(
      ArtifactGroupNamer artifactGroupNamer) {
    return new BuildEventContext() {
      @Override
      public ArtifactGroupNamer artifactGroupNamer() {
        return artifactGroupNamer;
      }

      @Override
      public PathConverter pathConverter() {
        return Path::toString;
      }

      @Override
      public BuildEventProtocolOptions getOptions() {
        return Options.getDefaults(BuildEventProtocolOptions.class);
      }
    };
  }

  protected static final class RecordingBuildEventTransport implements BuildEventTransport {
    private final List<BuildEvent> events = new ArrayList<>();
    private final List<BuildEventStreamProtos.BuildEvent> eventsAsProtos = new ArrayList<>();
    private final ArtifactGroupNamer artifactGroupNamer;

    RecordingBuildEventTransport(ArtifactGroupNamer namer) {
      this.artifactGroupNamer = namer;
    }

    @Override
    public String name() {
      return this.getClass().getSimpleName();
    }

    @Override
    public boolean mayBeSlow() {
      return false;
    }

    @Override
    public BesUploadMode getBesUploadMode() {
      return BesUploadMode.WAIT_FOR_UPLOAD_COMPLETE;
    }

    @Override
    public synchronized void sendBuildEvent(BuildEvent event) {
      events.add(event);
      try {
        eventsAsProtos.add(event.asStreamProto(getTestBuildEventContext(this.artifactGroupNamer)));
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IllegalStateException("interrupts not supported in test instance");
      }
    }

    @Override
    public ListenableFuture<Void> close() {
      return immediateVoidFuture();
    }

    @Override
    public BuildEventArtifactUploader getUploader() {
      throw new IllegalStateException();
    }

    List<BuildEvent> getEvents() {
      return events;
    }

    List<BuildEventStreamProtos.BuildEvent> getEventProtos() {
      return eventsAsProtos;
    }
  }
}
