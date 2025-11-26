// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildeventservice;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.v1.BuildEvent.EventCase.BAZEL_EVENT;
import static com.google.devtools.build.v1.BuildEvent.EventCase.BUILD_ENQUEUED;
import static com.google.devtools.build.v1.BuildEvent.EventCase.INVOCATION_ATTEMPT_STARTED;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;
import static org.mockito.ArgumentMatchers.anyMap;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.hash.HashCode;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient.CommandContext;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient.InvocationStatus;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceProtoUtil;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildCompletingEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.File;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.NamedSetOfFiles;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.buildeventstream.LocalFilesArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.buildeventstream.PathConverter.FileUriPathConverter;
import com.google.devtools.build.lib.buildeventstream.ProgressEvent;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.server.FailureDetails.BuildProgress;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.v1.PublishBuildToolEventStreamRequest;
import com.google.devtools.common.options.Options;
import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import io.grpc.Status;
import io.grpc.StatusException;
import io.netty.util.AbstractReferenceCounted;
import io.netty.util.ReferenceCounted;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Integration tests for {@link BuildEventServiceTransport} */
@RunWith(JUnit4.class)
public abstract class AbstractBuildEventServiceTransportTest extends FoundationTestCase {

  private static final long TIMEOUT_MILLIS = 20000;

  private static final String BUILD_REQUEST_ID = "feedbeef-dead-4321-beef-deaddeaddead";
  private static final String BUILD_INVOCATION_ID = "feedbeef-dead-4444-beef-deaddeaddead";
  private static final String COMMAND_NAME = "test";
  private static final ImmutableSet<String> KEYWORDS = ImmutableSet.of("foo=bar", "spam=eggs");
  private static final Instant COMMAND_START_TIME = Instant.ofEpochMilli(500L);
  private static final CommandContext COMMAND_CONTEXT =
      CommandContext.builder()
          .setBuildId(BUILD_REQUEST_ID)
          .setInvocationId(BUILD_INVOCATION_ID)
          .setAttemptNumber(1)
          .setKeywords(KEYWORDS)
          .setProjectId(null)
          .setCheckPrecedingLifecycleEvents(false)
          .build();

  private final ArtifactGroupNamer artifactGroupNamer = mock(ArtifactGroupNamer.class);
  private final BuildRequest buildRequest = mock(BuildRequest.class);
  private final BuildEventContext buildEventContext = mock(BuildEventContext.class);

  private final ManualClock clock = new ManualClock();

  private final AbstractBuildEventRecorder fakeBesServer = createBesServer();

  private final BuildEvent started =
      BuildStartingEvent.create(
          "OutputFileSystemType",
          /*usesInMemoryFileSystem=*/ false,
          buildRequest,
          /*workspace=*/ null,
          "/pwd");
  private final BuildEvent progress = ProgressEvent.progressUpdate(1);
  private final BuildEvent success =
      new BuildCompletingEvent(ExitCode.SUCCESS, System.currentTimeMillis()) {};
  private final BuildEvent failed =
      new BuildCompletingEvent(ExitCode.BUILD_FAILURE, System.currentTimeMillis()) {};

  @Before
  public void setUp() {
    when(buildRequest.getId()).thenReturn(UUID.fromString(BUILD_REQUEST_ID));
    when(buildRequest.getCommandName()).thenReturn(COMMAND_NAME);
    when(buildRequest.getOptionsDescription()).thenReturn("");

    fakeBesServer.startRpcServer();
  }

  @After
  public void tearDown() {
    Mockito.validateMockitoUsage();
    fakeBesServer.stopRpcServer();
  }

  @Test(timeout = TIMEOUT_MILLIS)
  public void testPublishLifecyleEvents_commandSucceeded() throws Exception {
    testPublishLifecycleEvents(InvocationStatus.SUCCEEDED, success);
  }

  @Test(timeout = TIMEOUT_MILLIS)
  public void testPublishLifecycleEvents_commandFailed() throws Exception {
    testPublishLifecycleEvents(InvocationStatus.FAILED, failed);
  }

  @Test(timeout = TIMEOUT_MILLIS)
  public void testPublishLifecycleEvents_statusUnknown() throws Exception {
    testPublishLifecycleEvents(InvocationStatus.UNKNOWN, progress);
  }

  private void testPublishLifecycleEvents(InvocationStatus expectedStatus, BuildEvent lastEvent)
      throws Exception {
    clock.advanceMillis(750L);
    Instant invocationStartedTimestamp = clock.now();
    BuildEventServiceTransport transport =
        newBuildEventServiceTransport(/*publishLifecycleEvents=*/ true);
    clock.advanceMillis(250L);
    Instant timestamp = clock.now();
    transport.sendBuildEvent(started);
    transport.sendBuildEvent(progress);
    transport.sendBuildEvent(lastEvent);
    transport.close().get();

    // build lifecycle events
    assertThat(
            fakeBesServer.getLifecycleEvents(
                BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, BUILD_ENQUEUED)))
        .containsExactly(
            BuildEventServiceProtoUtil.buildEnqueued(COMMAND_CONTEXT, COMMAND_START_TIME),
            BuildEventServiceProtoUtil.buildFinished(COMMAND_CONTEXT, timestamp, expectedStatus));

    // invocation lifecycle events
    assertThat(
            fakeBesServer.getLifecycleEvents(
                BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, INVOCATION_ATTEMPT_STARTED)))
        .containsExactly(
            BuildEventServiceProtoUtil.invocationStarted(
                COMMAND_CONTEXT, invocationStartedTimestamp),
            BuildEventServiceProtoUtil.invocationFinished(
                COMMAND_CONTEXT, timestamp, expectedStatus));

    // bazel stream events
    assertThat(
            fakeBesServer.getStreamEvents(
                BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, BAZEL_EVENT)))
        .containsExactly(
            BuildEventServiceProtoUtil.bazelEvent(
                COMMAND_CONTEXT,
                timestamp,
                1,
                started.asStreamProto(buildEventContext).toByteString()),
            BuildEventServiceProtoUtil.bazelEvent(
                COMMAND_CONTEXT,
                timestamp,
                2,
                progress.asStreamProto(buildEventContext).toByteString()),
            BuildEventServiceProtoUtil.bazelEvent(
                COMMAND_CONTEXT,
                timestamp,
                3,
                lastEvent.asStreamProto(buildEventContext).toByteString()),
            BuildEventServiceProtoUtil.streamFinished(COMMAND_CONTEXT, timestamp, 4))
        .inOrder();
  }

  @Test(timeout = TIMEOUT_MILLIS)
  public void disablingLifecycleEventsWorks() throws Exception {
    clock.advanceMillis(1000L);
    Instant timestamp = clock.now();
    BuildEventServiceTransport transport =
        newBuildEventServiceTransport(/*publishLifecycleEvents=*/ false);
    transport.sendBuildEvent(started);
    transport.sendBuildEvent(progress);
    transport.sendBuildEvent(success);
    transport.close().get();

    // bazel stream events
    assertThat(
            fakeBesServer.getStreamEvents(
                BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, BAZEL_EVENT)))
        .containsExactly(
            BuildEventServiceProtoUtil.bazelEvent(
                COMMAND_CONTEXT,
                timestamp,
                1,
                started.asStreamProto(buildEventContext).toByteString()),
            BuildEventServiceProtoUtil.bazelEvent(
                COMMAND_CONTEXT,
                timestamp,
                2,
                progress.asStreamProto(buildEventContext).toByteString()),
            BuildEventServiceProtoUtil.bazelEvent(
                COMMAND_CONTEXT,
                timestamp,
                3,
                success.asStreamProto(buildEventContext).toByteString()),
            BuildEventServiceProtoUtil.streamFinished(COMMAND_CONTEXT, timestamp, 4))
        .inOrder();
  }

  @Test(timeout = TIMEOUT_MILLIS)
  public void sendEventsInLockStep() throws Exception {
    // A test that only sends the next build event after the previous build event has been
    // ACKed by the server.
    BuildEventServiceTransport transport =
        newBuildEventServiceTransport(/*publishLifecycleEvents=*/ false);

    List<BuildEvent> toSend = Arrays.asList(started, progress, success);
    for (int i = 0; i < toSend.size(); i++) {
      transport.sendBuildEvent(toSend.get(i));
      while (fakeBesServer
              .getSuccessfulStreamEvents(
                  BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, BAZEL_EVENT))
              .size()
          != i + 1) {
        Thread.sleep(10);
      }
    }

    transport.close().get();
  }

  @Test(timeout = TIMEOUT_MILLIS)
  public void testAcksInBatchMode() throws Exception {
    clock.advanceMillis(1000L);
    Instant timestamp = clock.now();
    // Send the first ACK only after the last event has been received.
    fakeBesServer.setSendResponsesOnRequestPredicate(
        (req) ->
            Objects.equals(
                req, BuildEventServiceProtoUtil.streamFinished(COMMAND_CONTEXT, timestamp, 4)));
    BuildEventServiceTransport transport =
        newBuildEventServiceTransport(/*publishLifecycleEvents=*/ false);
    transport.sendBuildEvent(started);
    transport.sendBuildEvent(progress);
    transport.sendBuildEvent(success);
    transport.close().get();
  }

  @Test(timeout = TIMEOUT_MILLIS)
  public void retriesForLastEventShouldWork() throws Exception {
    clock.advanceMillis(1000L);
    Instant timestamp = clock.now();
    // Send UNAVAILABLE on streamFinished event
    fakeBesServer.setStreamEventPredicateAndResponseStatus(
        (req) ->
            Objects.equals(
                req, BuildEventServiceProtoUtil.streamFinished(COMMAND_CONTEXT, timestamp, 4)),
        Status.UNAVAILABLE);

    BuildEventServiceTransport transport =
        newBuildEventServiceTransport(/*publishLifecycleEvents=*/ false);
    transport.sendBuildEvent(started);
    transport.sendBuildEvent(progress);
    transport.sendBuildEvent(success);

    ExecutionException exception =
        assertThrows(ExecutionException.class, () -> transport.close().get());
    assertTransientError(exception, BuildProgress.Code.BES_UPLOAD_RETRY_LIMIT_EXCEEDED_FAILURE);
    assertThat(exception).hasCauseThat().hasCauseThat().isInstanceOf(StatusException.class);
    assertThat(((StatusException) exception.getCause().getCause()).getStatus().getCode())
        .isEqualTo(Status.UNAVAILABLE.getCode());

    assertThat(
            fakeBesServer.getStreamEvents(
                BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, BAZEL_EVENT)))
        .containsAtLeast(
            BuildEventServiceProtoUtil.bazelEvent(
                COMMAND_CONTEXT,
                timestamp,
                1,
                started.asStreamProto(buildEventContext).toByteString()),
            BuildEventServiceProtoUtil.bazelEvent(
                COMMAND_CONTEXT,
                timestamp,
                2,
                progress.asStreamProto(buildEventContext).toByteString()),
            BuildEventServiceProtoUtil.bazelEvent(
                COMMAND_CONTEXT,
                timestamp,
                3,
                success.asStreamProto(buildEventContext).toByteString()),
            BuildEventServiceProtoUtil.streamFinished(COMMAND_CONTEXT, timestamp, 4),
            // Verify retry on streamFinished message
            BuildEventServiceProtoUtil.streamFinished(COMMAND_CONTEXT, timestamp, 4))
        .inOrder();
  }

  @Test(timeout = TIMEOUT_MILLIS)
  public void retriesForInvocationStartedEventShouldWork() throws Exception {
    clock.advanceMillis(750L);
    Instant invocationStartedTimestamp = clock.now();
    // Respond with UNAVAILABLE to invocation started lifecycle event
    fakeBesServer.setLifecycleEventPredicateAndResponseStatus(
        (req) ->
            Objects.equals(
                req,
                BuildEventServiceProtoUtil.invocationStarted(
                    COMMAND_CONTEXT, invocationStartedTimestamp)),
        Status.UNAVAILABLE);

    BuildEventServiceTransport transport =
        newBuildEventServiceTransport(/*publishLifecycleEvents=*/ true);

    ExecutionException exception =
        assertThrows(ExecutionException.class, () -> transport.close().get());
    assertTransientError(exception, BuildProgress.Code.BES_UPLOAD_RETRY_LIMIT_EXCEEDED_FAILURE);
    assertThat(exception).hasCauseThat().hasCauseThat().isInstanceOf(StatusException.class);
    assertThat(((StatusException) exception.getCause().getCause()).getStatus().getCode())
        .isEqualTo(Status.UNAVAILABLE.getCode());

    // should not proceed as lifecycle event failed
    assertThat(
            fakeBesServer.getLifecycleEvents(
                BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, BUILD_ENQUEUED)))
        .containsExactly(
            BuildEventServiceProtoUtil.buildEnqueued(COMMAND_CONTEXT, COMMAND_START_TIME));

    // should retry only the rpc that failed
    assertThat(
            fakeBesServer.getLifecycleEvents(
                BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, INVOCATION_ATTEMPT_STARTED)))
        .containsExactly(
            BuildEventServiceProtoUtil.invocationStarted(
                COMMAND_CONTEXT, invocationStartedTimestamp),
            BuildEventServiceProtoUtil.invocationStarted(
                COMMAND_CONTEXT, invocationStartedTimestamp),
            BuildEventServiceProtoUtil.invocationStarted(
                COMMAND_CONTEXT, invocationStartedTimestamp),
            BuildEventServiceProtoUtil.invocationStarted(
                COMMAND_CONTEXT, invocationStartedTimestamp),
            BuildEventServiceProtoUtil.invocationStarted(
                COMMAND_CONTEXT, invocationStartedTimestamp));
  }

  @Test(timeout = TIMEOUT_MILLIS)
  public void testRetriesForBuildEvents_oneEventFailsAlways() throws Exception {
    clock.advanceMillis(1000L);
    Instant timestamp = clock.now();

    ByteString expectedPayload = progress.asStreamProto(buildEventContext).toByteString();
    fakeBesServer.setStreamEventPredicateAndResponseStatus(
        (req) ->
            Objects.equals(
                req,
                BuildEventServiceProtoUtil.bazelEvent(
                    COMMAND_CONTEXT, timestamp, 2, expectedPayload)),
        Status.CANCELLED);

    BuildEventServiceTransport transport =
        newBuildEventServiceTransport(/*publishLifecycleEvents=*/ false);
    transport.sendBuildEvent(started);
    transport.sendBuildEvent(progress);
    transport.sendBuildEvent(success);

    ExecutionException exception =
        assertThrows(ExecutionException.class, () -> transport.close().get());
    assertTransientError(exception, BuildProgress.Code.BES_UPLOAD_RETRY_LIMIT_EXCEEDED_FAILURE);
    assertThat(exception).hasCauseThat().hasCauseThat().isInstanceOf(StatusException.class);
    assertThat(((StatusException) exception.getCause().getCause()).getStatus().getCode())
        .isEqualTo(Status.CANCELLED.getCode());

    assertThat(
            fakeBesServer.getSuccessfulStreamEvents(
                BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, BAZEL_EVENT)))
        .contains(
            BuildEventServiceProtoUtil.bazelEvent(
                COMMAND_CONTEXT,
                timestamp,
                1,
                started.asStreamProto(buildEventContext).toByteString()));

    assertThat(
            fakeBesServer.getStreamEvents(
                BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, BAZEL_EVENT)))
        .containsAtLeast(
            BuildEventServiceProtoUtil.bazelEvent(COMMAND_CONTEXT, timestamp, 2, expectedPayload),
            BuildEventServiceProtoUtil.bazelEvent(COMMAND_CONTEXT, timestamp, 2, expectedPayload),
            BuildEventServiceProtoUtil.bazelEvent(COMMAND_CONTEXT, timestamp, 2, expectedPayload),
            BuildEventServiceProtoUtil.bazelEvent(COMMAND_CONTEXT, timestamp, 2, expectedPayload),
            BuildEventServiceProtoUtil.bazelEvent(COMMAND_CONTEXT, timestamp, 2, expectedPayload));
  }

  @Test(timeout = TIMEOUT_MILLIS)
  public void testRetriesForBuildEvents_everyEventFailsOnce() throws Exception {
    clock.advanceMillis(1000L);
    Instant timestamp = clock.now();
    fakeBesServer.setStreamEventPredicateAndResponseStatus(
        everyEventFailsOnce(), Status.UNAVAILABLE);

    BuildEventServiceTransport transport =
        newBuildEventServiceTransport(/*publishLifecycleEvents=*/ false);
    transport.sendBuildEvent(started);
    transport.sendBuildEvent(success);
    transport.close().get();

    assertThat(
            fakeBesServer.getSuccessfulStreamEvents(
                BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, BAZEL_EVENT)))
        .containsAtLeast(
            BuildEventServiceProtoUtil.bazelEvent(
                COMMAND_CONTEXT,
                timestamp,
                1,
                started.asStreamProto(buildEventContext).toByteString()),
            BuildEventServiceProtoUtil.bazelEvent(
                COMMAND_CONTEXT,
                timestamp,
                2,
                success.asStreamProto(buildEventContext).toByteString()),
            BuildEventServiceProtoUtil.streamFinished(COMMAND_CONTEXT, timestamp, 3));

    assertThat(
            fakeBesServer.getStreamEvents(
                BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, BAZEL_EVENT)))
        .containsAtLeast(
            BuildEventServiceProtoUtil.bazelEvent(
                COMMAND_CONTEXT,
                timestamp,
                1,
                started.asStreamProto(buildEventContext).toByteString()),
            BuildEventServiceProtoUtil.bazelEvent(
                COMMAND_CONTEXT,
                timestamp,
                1,
                started.asStreamProto(buildEventContext).toByteString()),
            BuildEventServiceProtoUtil.bazelEvent(
                COMMAND_CONTEXT,
                timestamp,
                2,
                success.asStreamProto(buildEventContext).toByteString()),
            BuildEventServiceProtoUtil.bazelEvent(
                COMMAND_CONTEXT,
                timestamp,
                2,
                success.asStreamProto(buildEventContext).toByteString()),
            BuildEventServiceProtoUtil.streamFinished(COMMAND_CONTEXT, timestamp, 3),
            BuildEventServiceProtoUtil.streamFinished(COMMAND_CONTEXT, timestamp, 3));
  }

  /** Tests that a successfully transmitted build event resets the retry counter. */
  @Test(timeout = TIMEOUT_MILLIS)
  public void testRetriesForBuildEvents_acksResetsAttempts() throws Exception {
    Set<Long> failedSeqNumbers = Collections.synchronizedSet(new LinkedHashSet<>());
    // Fail once with UNAVAILABLE (transient error) for every build event.
    fakeBesServer.setStreamEventPredicateAndResponseStatus(
        req -> {
          long seqNumber = req.getOrderedBuildEvent().getSequenceNumber();
          return failedSeqNumbers.add(seqNumber);
        },
        Status.UNAVAILABLE);

    BuildEventServiceTransport transport =
        newBuildEventServiceTransport(/*publishLifecycleEvents=*/ false);

    transport.sendBuildEvent(started);
    for (int i = 0; i < 10; i++) {
      transport.sendBuildEvent(progress);
    }
    transport.sendBuildEvent(success);

    transport.close().get();

    Set<Long> successfulSequenceNumbers =
        fakeBesServer
            .getSuccessfulStreamEvents(
                BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, BAZEL_EVENT))
            .stream()
            .map((e) -> e.getOrderedBuildEvent().getSequenceNumber())
            .collect(Collectors.toSet());

    assertThat(successfulSequenceNumbers).containsExactlyElementsIn(failedSeqNumbers);
    assertThat(successfulSequenceNumbers).hasSize(13);
  }

  @Test(timeout = TIMEOUT_MILLIS)
  public void persistentErrorsShouldNotBeRetried_eventStream_invalidArgument() throws Exception {
    testPermanentErrorsCauseBlazeExit(
        Status.INVALID_ARGUMENT,
        ExitCode.PERSISTENT_BUILD_EVENT_SERVICE_UPLOAD_ERROR,
        BuildProgress.Code.BES_STREAM_NOT_RETRYING_FAILURE);
  }

  @Test(timeout = TIMEOUT_MILLIS)
  public void persistentErrorsShouldNotBeRetried_eventStream_failedPrecondition() throws Exception {
    testPermanentErrorsCauseBlazeExit(
        Status.FAILED_PRECONDITION,
        ExitCode.TRANSIENT_BUILD_EVENT_SERVICE_UPLOAD_ERROR,
        BuildProgress.Code.BES_UPLOAD_TIMEOUT_ERROR);
  }

  private void testPermanentErrorsCauseBlazeExit(
      Status status, ExitCode exitCode, BuildProgress.Code buildProgressCode) throws Exception {
    clock.advanceMillis(1000L);
    Instant timestamp = clock.now();
    fakeBesServer.setStreamEventPredicateAndResponseStatus((req) -> true, status);

    BuildEventServiceTransport transport =
        newBuildEventServiceTransport(/* publishLifecycleEvents= */ false);
    transport.sendBuildEvent(started);

    ExecutionException exception =
        assertThrows(ExecutionException.class, () -> transport.close().get());
    assertExecutionException(exception, exitCode, buildProgressCode);
    assertThat(exception).hasCauseThat().hasCauseThat().isInstanceOf(StatusException.class);
    assertThat(((StatusException) exception.getCause().getCause()).getStatus().getCode())
        .isEqualTo(status.getCode());

    assertThat(
            fakeBesServer.getStreamEvents(
                BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, BAZEL_EVENT)))
        .contains(
            BuildEventServiceProtoUtil.bazelEvent(
                COMMAND_CONTEXT,
                timestamp,
                1,
                started.asStreamProto(buildEventContext).toByteString()));

    assertThat(
            fakeBesServer.getSuccessfulStreamEvents(
                BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, BAZEL_EVENT)))
        .isEmpty();
  }

  @Test(timeout = TIMEOUT_MILLIS)
  public void persistentErrorsShouldNotBeRetried_lifecycleEvents() throws Exception {
    fakeBesServer.setLifecycleEventPredicateAndResponseStatus(
        (req) -> true, Status.FAILED_PRECONDITION);

    BuildEventServiceTransport transport =
        newBuildEventServiceTransport(/*publishLifecycleEvents=*/ true);
    transport.sendBuildEvent(started);

    ExecutionException exception =
        assertThrows(ExecutionException.class, () -> transport.close().get());
    assertPersistentError(exception, BuildProgress.Code.BES_STREAM_NOT_RETRYING_FAILURE);
    assertThat(exception).hasCauseThat().hasCauseThat().isInstanceOf(StatusException.class);
    assertThat(((StatusException) exception.getCause().getCause()).getStatus().getCode())
        .isEqualTo(Status.FAILED_PRECONDITION.getCode());

    assertThat(
            fakeBesServer.getLifecycleEvents(
                BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, BUILD_ENQUEUED)))
        .containsExactly(
            BuildEventServiceProtoUtil.buildEnqueued(COMMAND_CONTEXT, COMMAND_START_TIME));
  }

  @Test(timeout = TIMEOUT_MILLIS)
  public void lifecycleEventsAreRetried() throws Exception {
    clock.advanceMillis(750L);
    Instant invocationStartedTimestamp = clock.now();
    fakeBesServer.setLifecycleEventPredicateAndResponseStatus(
        everyEventFailsOnce(), Status.UNAVAILABLE);

    BuildEventServiceTransport transport =
        newBuildEventServiceTransport(/*publishLifecycleEvents=*/ true);
    clock.advanceMillis(250L);
    Instant timestamp = clock.now();
    transport.close().get();

    // all  build lifecycle events
    assertThat(
            fakeBesServer.getLifecycleEvents(
                BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, BUILD_ENQUEUED)))
        .containsExactly(
            BuildEventServiceProtoUtil.buildEnqueued(COMMAND_CONTEXT, COMMAND_START_TIME),
            BuildEventServiceProtoUtil.buildEnqueued(COMMAND_CONTEXT, COMMAND_START_TIME),
            BuildEventServiceProtoUtil.buildFinished(
                COMMAND_CONTEXT, timestamp, InvocationStatus.UNKNOWN),
            BuildEventServiceProtoUtil.buildFinished(
                COMMAND_CONTEXT, timestamp, InvocationStatus.UNKNOWN))
        .inOrder();

    // all invocation lifecycle events
    assertThat(
            fakeBesServer.getLifecycleEvents(
                BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, INVOCATION_ATTEMPT_STARTED)))
        .containsExactly(
            BuildEventServiceProtoUtil.invocationStarted(
                COMMAND_CONTEXT, invocationStartedTimestamp),
            BuildEventServiceProtoUtil.invocationStarted(
                COMMAND_CONTEXT, invocationStartedTimestamp),
            BuildEventServiceProtoUtil.invocationFinished(
                COMMAND_CONTEXT, timestamp, InvocationStatus.UNKNOWN),
            BuildEventServiceProtoUtil.invocationFinished(
                COMMAND_CONTEXT, timestamp, InvocationStatus.UNKNOWN))
        .inOrder();

    // All event stream.
    assertThat(
            fakeBesServer.getStreamEvents(
                BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, BAZEL_EVENT)))
        .containsExactly(BuildEventServiceProtoUtil.streamFinished(COMMAND_CONTEXT, timestamp, 1));
  }

  /**
   * Sending a response status OK with ACKs outstanding is a protocol error and should fail the
   * stream without retries.
   */
  @Test(timeout = TIMEOUT_MILLIS)
  public void responseStatusOkWithAcksMissing() throws Exception {
    fakeBesServer.setStreamEventPredicateAndResponseStatus(everyEventFailsOnce(), Status.OK);

    BuildEventServiceTransport transport =
        newBuildEventServiceTransport(/*publishLifecycleEvents=*/ false);
    transport.sendBuildEvent(started);
    transport.sendBuildEvent(progress);
    transport.sendBuildEvent(success);

    ExecutionException exception =
        assertThrows(ExecutionException.class, () -> transport.close().get());
    assertPersistentError(
        exception, BuildProgress.Code.BES_STREAM_COMPLETED_WITH_UNACK_EVENTS_ERROR);
    assertThat(exception).hasCauseThat().hasCauseThat().isInstanceOf(StatusException.class);
    assertThat(((StatusException) exception.getCause().getCause()).getStatus().getCode())
        .isEqualTo(Status.FAILED_PRECONDITION.getCode());
  }

  /** Tests that uploading files referenced by a build event works. */
  @Test(timeout = TIMEOUT_MILLIS)
  public void testFileUpload() throws Exception {
    InMemoryFileSystem inMemoryFs = new InMemoryFileSystem(makeVfsHashFunction());
    Path file1 = inMemoryFs.getPath("/file1");
    Path file2 = inMemoryFs.getPath("/file2");
    FileSystemUtils.writeContentAsLatin1(file1, "file1");
    FileSystemUtils.writeContentAsLatin1(file2, "file2");
    BuildEvent withFiles =
        new BuildEventWithFiles(
            ImmutableList.of(
                new LocalFile(file1, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null),
                new LocalFile(file2, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null)));

    BuildEventArtifactUploader uploader =
        new BuildEventArtifactUploaderWithRefCounting() {
          @Override
          public ListenableFuture<PathConverter> upload(Map<Path, LocalFile> files) {
            Map<Path, String> conversion = new HashMap<>();
            for (Path file : files.keySet()) {
              try {
                conversion.put(file, "cas://" + HashCode.fromBytes(file.getDigest()));
              } catch (IOException e) {
                return Futures.immediateFailedFuture(e);
              }
            }
            return Futures.immediateFuture(conversion::get);
          }

          @Override
          public boolean mayBeSlow() {
            return false;
          }
        };
    uploader = Mockito.spy(uploader);

    BuildEventServiceTransport transport =
        newBuildEventServiceTransport(createBesClient(), true, Duration.ZERO, uploader);
    transport.sendBuildEvent(started);
    transport.sendBuildEvent(progress);
    transport.sendBuildEvent(withFiles);
    transport.sendBuildEvent(success);

    transport.close().get();

    verify(uploader)
        .upload(
            eq(
                ImmutableMap.of(
                    file1,
                    new LocalFile(file1, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null),
                    file2,
                    new LocalFile(
                        file2, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null))));

    List<PublishBuildToolEventStreamRequest> events =
        fakeBesServer.getStreamEvents(
            BuildEventServiceProtoUtil.streamId(COMMAND_CONTEXT, BAZEL_EVENT));

    Any anyEvent = events.get(2).getOrderedBuildEvent().getEvent().getBazelEvent();
    BuildEventStreamProtos.BuildEvent buildEvent =
        anyEvent.unpack(BuildEventStreamProtos.BuildEvent.class);
    assertThat(buildEvent).isNotNull();
    assertThat(buildEvent.hasNamedSetOfFiles()).isTrue();
    assertThat(buildEvent.getNamedSetOfFiles().getFilesCount()).isEqualTo(2);

    Set<String> referencedFiles =
        buildEvent.getNamedSetOfFiles().getFilesList().stream()
            .map(File::getUri)
            .collect(Collectors.toSet());
    String file1Hash =
        makeVfsHashFunction().getHashFunction().hashString("file1", UTF_8).toString();
    String file2Hash =
        makeVfsHashFunction().getHashFunction().hashString("file2", UTF_8).toString();
    assertThat(referencedFiles).containsExactly("cas://" + file1Hash, "cas://" + file2Hash);
  }

  /** Regression test for b/112189077. */
  @Test(timeout = TIMEOUT_MILLIS)
  public void testFileUploadWithDuplicatePaths() throws Exception {
    InMemoryFileSystem inMemoryFs = new InMemoryFileSystem(new JavaClock(), makeVfsHashFunction());
    Path file1 = inMemoryFs.getPath("/file1");
    FileSystemUtils.writeContentAsLatin1(file1, "file1");
    BuildEvent withFiles =
        new BuildEventWithFiles(
            ImmutableList.of(
                new LocalFile(file1, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null),
                new LocalFile(file1, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null)));

    BuildEventArtifactUploader uploader =
        new BuildEventArtifactUploaderWithRefCounting() {
          @Override
          public ListenableFuture<PathConverter> upload(Map<Path, LocalFile> files) {
            return Futures.immediateFuture(new FileUriPathConverter());
          }

          @Override
          public boolean mayBeSlow() {
            return false;
          }
        };
    uploader = Mockito.spy(uploader);

    BuildEventServiceTransport transport =
        newBuildEventServiceTransport(createBesClient(), true, Duration.ZERO, uploader);
    transport.sendBuildEvent(withFiles);
    transport.close().get();

    // Check to make sure the code path was exercised
    verify(uploader)
        .upload(
            eq(
                ImmutableMap.of(
                    file1,
                    new LocalFile(
                        file1, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null))));
  }

  /** Regression test for b/111389420. */
  @Test(timeout = TIMEOUT_MILLIS)
  public void testFileUploadFails() throws Exception {
    // Test that a failed file upload is not retried and fails the whole upload.
    Exception uploadFailed = new IOException("File upload failed.");
    BuildEventArtifactUploader uploader =
        new BuildEventArtifactUploaderWithRefCounting() {
          private int callCount;

          @Override
          public ListenableFuture<PathConverter> upload(Map<Path, LocalFile> files) {
            callCount++;
            // Apparently, Stubby behaves like this:
            // When we create a connection but immediately abort it, it seems like the server is not
            // notified at all, so we need to post at least one event before we abort.
            if (callCount == 1) {
              return Futures.immediateFuture(PathConverter.NO_CONVERSION);
            } else if (callCount == 2) {
              return Futures.immediateFailedFuture(uploadFailed);
            } else {
              fail("Expected exactly two calls to upload.");
              return null;
            }
          }

          @Override
          public boolean mayBeSlow() {
            return false;
          }
        };
    uploader = Mockito.spy(uploader);

    BuildEventServiceTransport transport =
        newBuildEventServiceTransport(createBesClient(), true, Duration.ZERO, uploader);
    transport.sendBuildEvent(started);

    // Wait for lifecycle events to be sent.
    while (!fakeBesServer.publishBuildToolEventStreamAccepted()) {
      Thread.sleep(10);
    }

    // This event will trigger a upload that fails.
    transport.sendBuildEvent(success);

    // Wait until the server error is found _before_ we shut down the transport. Otherwise the close
    // might race with the error.
    while (fakeBesServer.eventStreamError() == null) {
      Thread.sleep(10);
    }

    ExecutionException exception =
        assertThrows(ExecutionException.class, () -> transport.close().get());
    assertTransientError(exception, BuildProgress.Code.BES_UPLOAD_LOCAL_FILE_ERROR);
    assertThat(exception).hasCauseThat().hasCauseThat().isEqualTo(uploadFailed);

    assertThat(fakeBesServer.eventStreamError().getCode())
        .isAnyOf(Status.CANCELLED.getCode(), Status.INTERNAL.getCode());

    verify(uploader, Mockito.times(2)).upload(anyMap());
  }

  /**
   * Tests that sending ACKS out of order or for non-existing events fails the upload without
   * retries, as this signals a bug in the server code.
   *
   * <p>Note that we do not retry within the invocation, but we return a <em>transient</em> exit
   * code. The {@code FAILED_PRECONDITION} error indicates the protocol has broken; retrying the
   * entire Blaze invocation would construct a new instance of the protocol and might work.
   */
  @Test(timeout = TIMEOUT_MILLIS)
  public void testWrongAckShouldFailTheUpload() throws Exception {
    fakeBesServer.sendOutOfOrderAcknowledgments();

    BuildEventServiceTransport transport =
        newBuildEventServiceTransport(/*publishLifecycleEvents=*/ true);
    transport.sendBuildEvent(started);

    ExecutionException exception =
        assertThrows(ExecutionException.class, () -> transport.close().get());
    assertTransientError(exception, BuildProgress.Code.BES_UPLOAD_TIMEOUT_ERROR);
    assertThat(exception).hasCauseThat().hasCauseThat().isInstanceOf(StatusException.class);
    assertThat(((StatusException) exception.getCause().getCause()).getStatus().getCode())
        .isEqualTo(Status.FAILED_PRECONDITION.getCode());
  }

  /**
   * Don't ACK build events, and never half-close the stream from the server side thus forcing a
   * timeout on the client.
   */
  @Test(timeout = TIMEOUT_MILLIS)
  public void testCloseTimeout() throws Exception {
    fakeBesServer.setStreamEventPredicateAndResponseStatus((req) -> true, null);

    // Timeout 1 second after calling close()
    BuildEventServiceTransport transport =
        newBuildEventServiceTransport(
            createBesClient(),
            /*publishLifecycleEvents=*/ true,
            Duration.ofSeconds(1),
            new LocalFilesArtifactUploader());
    transport.sendBuildEvent(started);

    assertThrows(
        TimeoutException.class,
        () -> transport.close().get(transport.getTimeout().toMillis(), TimeUnit.MILLISECONDS));
  }

  private static void assertTransientError(Exception e, BuildProgress.Code bpCode) {
    assertExecutionException(e, ExitCode.TRANSIENT_BUILD_EVENT_SERVICE_UPLOAD_ERROR, bpCode);
  }

  private static void assertPersistentError(Exception e, BuildProgress.Code bpCode) {
    assertExecutionException(e, ExitCode.PERSISTENT_BUILD_EVENT_SERVICE_UPLOAD_ERROR, bpCode);
  }

  private static void assertExecutionException(
      Exception e, ExitCode exitCode, BuildProgress.Code bpCode) {
    assertThat(e).hasCauseThat().isInstanceOf(AbruptExitException.class);
    DetailedExitCode detailedExitCode = ((AbruptExitException) e.getCause()).getDetailedExitCode();
    FailureDetail failureDetail = detailedExitCode.getFailureDetail();
    assertThat(detailedExitCode.getExitCode()).isEqualTo(exitCode);
    assertThat(failureDetail.getBuildProgress().getCode()).isEqualTo(bpCode);
  }

  protected abstract AbstractBuildEventRecorder createBesServer();

  protected abstract BuildEventServiceClient createBesClient() throws IOException;

  protected abstract BuildEventServiceClient createBesClient(int serverPort) throws IOException;

  protected abstract DigestHashFunction makeVfsHashFunction();

  private BuildEventServiceTransport newBuildEventServiceTransport(boolean publishLifecycleEvents)
      throws IOException {
    return newBuildEventServiceTransport(
        createBesClient(), publishLifecycleEvents, Duration.ZERO, new LocalFilesArtifactUploader());
  }

  private BuildEventServiceTransport newBuildEventServiceTransport(
      BuildEventServiceClient client,
      boolean publishLifecycleEvents,
      Duration closeTimeout,
      @Nullable BuildEventArtifactUploader artifactUploader) {

    BuildEventServiceOptions besOptions = Options.getDefaults(BuildEventServiceOptions.class);
    besOptions.besTimeout = closeTimeout;
    besOptions.besLifecycleEvents = publishLifecycleEvents;

    return new BuildEventServiceTransport.Builder()
        .besOptions(besOptions)
        // Reduce exponential backoff sleep times to speed up testing
        .sleeper(
            (sleepMillis) ->
                TimeUnit.MILLISECONDS.sleep(sleepMillis > 10 ? sleepMillis / 10 : sleepMillis))
        .eventBus(eventBus)
        .besClient(client)
        .artifactGroupNamer(artifactGroupNamer)
        .localFileUploader(
            artifactUploader != null ? artifactUploader : new LocalFilesArtifactUploader())
        .bepOptions(Options.getDefaults(BuildEventProtocolOptions.class))
        .clock(clock)
        .commandContext(COMMAND_CONTEXT)
        .commandStartTime(COMMAND_START_TIME)
        .build();
  }

  private static final class BuildEventWithFiles implements BuildEvent {
    private final Collection<LocalFile> files;

    BuildEventWithFiles(Collection<LocalFile> files) {
      this.files = files;
    }

    @Override
    public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
      NamedSetOfFiles.Builder builder = NamedSetOfFiles.newBuilder();
      for (LocalFile file : files) {
        String uri = converters.pathConverter().apply(file.path);
        if (uri != null) {
          builder.addFiles(File.newBuilder().setName(file.path.getBaseName()).setUri(uri));
        }
      }
      return GenericBuildEvent.protoChaining(this).setNamedSetOfFiles(builder.build()).build();
    }

    @Override
    public BuildEventId getEventId() {
      return BuildEventIdUtil.fromArtifactGroupName("list-of-files");
    }

    @Override
    public Collection<LocalFile> referencedLocalFiles() {
      return files;
    }

    @Override
    public Collection<BuildEventId> getChildrenEvents() {
      return ImmutableSet.of();
    }
  }

  /** Utility method that produces a stateful predicate that matches a parameter only once. */
  private static <T> Predicate<T> everyEventFailsOnce() {
    return new Predicate<T>() {

      private final Set<T> alreadyMatched = new HashSet<>();

      @Override
      public boolean test(@Nullable T o) {
        return alreadyMatched.add(o);
      }
    };
  }

  private abstract static class BuildEventArtifactUploaderWithRefCounting
      extends AbstractReferenceCounted implements BuildEventArtifactUploader {

    @Override
    protected void deallocate() {}

    @Override
    public ReferenceCounted touch(Object o) {
      return this;
    }
  }
}
