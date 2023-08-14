// Copyright 2021 The Bazel Authors. All rights reserved.
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
import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;
import static com.google.devtools.build.lib.buildeventservice.BuildEventServiceModule.RUNS_PER_TEST_LIMIT;
import static org.junit.Assert.assertThrows;
import static org.junit.Assume.assumeFalse;

import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialModule;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.bugreport.Crash;
import com.google.devtools.build.lib.bugreport.CrashContext;
import com.google.devtools.build.lib.buildeventservice.BazelBuildEventServiceModule.BackendConfig;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.Aborted;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.Aborted.AbortReason;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.BuildFinishedId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ConfigurationId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.NamedSetOfFilesId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.TargetCompletedId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.NamedSetOfFiles;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.OutputGroup;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.transports.BinaryFormatFileTransport;
import com.google.devtools.build.lib.buildeventstream.transports.JsonFormatFileTransport;
import com.google.devtools.build.lib.buildeventstream.transports.TextFormatFileTransport;
import com.google.devtools.build.lib.buildtool.util.BlazeRuntimeWrapper;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.network.ConnectivityStatus;
import com.google.devtools.build.lib.network.ConnectivityStatusProvider;
import com.google.devtools.build.lib.network.NoOpConnectivityModule;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.NoSpawnCacheModule;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.skyframe.NotifyingHelper;
import com.google.devtools.build.v1.BuildEvent.BuildComponentStreamFinished.FinishType;
import com.google.devtools.build.v1.PublishBuildEventGrpc.PublishBuildEventImplBase;
import com.google.devtools.build.v1.PublishBuildToolEventStreamRequest;
import com.google.devtools.build.v1.PublishBuildToolEventStreamResponse;
import com.google.devtools.build.v1.PublishLifecycleEventRequest;
import com.google.devtools.build.v1.StreamId;
import com.google.protobuf.Empty;
import io.grpc.ManagedChannel;
import io.grpc.Metadata;
import io.grpc.Server;
import io.grpc.ServerInterceptors;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import io.grpc.util.MutableHandlerRegistry;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.Thread.UncaughtExceptionHandler;
import java.time.Duration;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedSet;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BazelBuildEventServiceModule}. */
@RunWith(JUnit4.class)
public final class BazelBuildEventServiceModuleTest extends BuildIntegrationTestCase {

  private static final Duration WAIT_FOR_LAST_INVOCATION_TIMEOUT = Duration.ofSeconds(2);

  private final String fakeServerName = "fake server for " + getClass();
  private final DelayingPublishBuildEventService buildEventService =
      new DelayingPublishBuildEventService();
  private final MutableHandlerRegistry serviceRegistry = new MutableHandlerRegistry();
  private Server fakeServer;

  private BazelBuildEventServiceModule besModule;
  private BlazeModule connectivityModule = new NoOpConnectivityModule();

  @Rule public TemporaryFolder tmpFolder = new TemporaryFolder();

  @Override
  protected BlazeModule getConnectivityModule() {
    return connectivityModule;
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(
            new BlazeModule() {
              @Override
              public void beforeCommand(CommandEnvironment env) {
                BazelBuildEventServiceModuleTest.this.events.initExternal(env.getReporter());
              }
            })
        .addBlazeModule(new NoSpawnCacheModule())
        .addBlazeModule(new CredentialModule())
        .addBlazeModule(
            new BazelBuildEventServiceModule() {
              @Override
              protected ManagedChannel newGrpcChannel(BackendConfig config) throws IOException {
                if (config.besBackend().equals("inprocess")) {
                  return InProcessChannelBuilder.forName(fakeServerName).build();
                }
                return super.newGrpcChannel(config);
              }

              @Override
              protected Duration getMaxWaitForPreviousInvocation() {
                return WAIT_FOR_LAST_INVOCATION_TIMEOUT;
              }
            });
  }

  private void runBuildWithOptions(String... options) throws Exception {
    addOptions(options);
    besModule = runtimeWrapper.getRuntime().getBlazeModule(BazelBuildEventServiceModule.class);
    runtimeWrapper.newCommand();
    buildTarget();
  }

  private void afterBuildCommand() throws Exception {
    runtimeWrapper.newCommand();
  }

  @Override
  @Nullable
  protected UncaughtExceptionHandler createUncaughtExceptionHandler() {
    // Disable the crash handler since this test leaves runaway threads e.g. accessing shut down
    // fakeServer.
    return null;
  }

  @Before
  public void setUp() throws Exception {
    serviceRegistry.addService(
        ServerInterceptors.intercept(
            buildEventService, new TracingMetadataUtils.ServerHeadersInterceptor()));
    fakeServer =
        InProcessServerBuilder.forName(fakeServerName)
            .fallbackHandlerRegistry(serviceRegistry)
            .directExecutor()
            .build()
            .start();
  }

  @After
  public void tearDown() throws Exception {
    fakeServer.shutdownNow();
    fakeServer.awaitTermination();
  }

  @Test
  public void testCreatesStreamerForTextFormatFileTransport() throws Exception {
    runBuildWithOptions("--build_event_text_file=" + tmpFolder.newFile().getAbsolutePath());
    assertThat(besModule.getBepTransports()).hasSize(1);
    assertThat(besModule.getBepTransports().asList().get(0))
        .isInstanceOf(TextFormatFileTransport.class);
  }

  @Test
  public void testCreatesStreamerForBinaryFormatFileTransport() throws Exception {
    runBuildWithOptions("--build_event_binary_file=" + tmpFolder.newFile().getAbsolutePath());
    assertThat(besModule.getBepTransports()).hasSize(1);
    assertThat(besModule.getBepTransports().asList().get(0))
        .isInstanceOf(BinaryFormatFileTransport.class);
  }

  @Test
  public void testCreatesStreamerForJsonFormatFileTransport() throws Exception {
    runBuildWithOptions("--build_event_json_file=" + tmpFolder.newFile().getAbsolutePath());
    assertThat(besModule.getBepTransports()).hasSize(1);
    assertThat(besModule.getBepTransports().asList().get(0))
        .isInstanceOf(JsonFormatFileTransport.class);
  }

  @Test
  public void testCreatesStreamerForBesTransport() throws Exception {
    runBuildWithOptions("--bes_backend=does.not.exist:1234");
    assertThat(besModule.getBepTransports()).hasSize(1);
    assertThat(besModule.getBepTransports().asList().get(0))
        .isInstanceOf(BuildEventServiceTransport.class);
  }

  @Test
  public void testRetryCount() throws Exception {
    runBuildWithOptions(
        "--bes_backend=does.not.exist:1234", "--experimental_build_event_upload_max_retries=3");
    afterBuildCommand();

    events.assertContainsError(
        "The Build Event Protocol upload failed: All 3 retry attempts failed");
  }

  @Test
  public void testConnectivityFailureDisablesBesStreaming() throws Exception {
    class FailingConnectivityStatusProvider extends BlazeModule
        implements ConnectivityStatusProvider {
      @Override
      public ConnectivityStatus getStatus(String service) {
        return new ConnectivityStatus(
            ConnectivityStatus.Status.NO_CREDENTIALS, "forced connectivity failure");
      }
    }

    connectivityModule = new FailingConnectivityStatusProvider();

    BlazeRuntimeWrapper runtimeWrapper =
        new BlazeRuntimeWrapper(
            events, serverDirectories, directories, binTools, getRuntimeBuilder());

    BazelBuildEventServiceModule besModule =
        runtimeWrapper.getRuntime().getBlazeModule(BazelBuildEventServiceModule.class);
    runtimeWrapper.addOptions("--bes_backend=does.not.exist:1234");
    runtimeWrapper.addOptions("--spawn_strategy=standalone");
    runtimeWrapper.executeBuild(ImmutableList.of());
    assertThat(besModule.getBepTransports()).isEmpty();
  }

  @Test
  public void testCreatesStreamerForGrpcBesResultsUrl() throws Exception {
    runBuildWithOptions(
        "--bes_backend=inprocess",
        "--bes_upload_mode=FULLY_ASYNC",
        "--bes_results_url=http://results-ui/");

    assertThat(besModule.getBepTransports()).hasSize(1);
    assertThat(besModule.getBepTransports().asList().get(0))
        .isInstanceOf(BuildEventServiceTransport.class);
  }

  @Test
  public void testCreatesStreamerForGrpcRunsPerTestTooHighDisablesStreaming() {
    AbruptExitException expected =
        assertThrows(
            AbruptExitException.class,
            () ->
                runBuildWithOptions(
                    "--bes_backend=inprocess", "--runs_per_test=" + (RUNS_PER_TEST_LIMIT + 1)));
    assertThat(expected.getExitCode()).isEqualTo(ExitCode.COMMAND_LINE_ERROR);
    assertThat(besModule.getBepTransports()).isEmpty();
    assertContainsError("The value of --runs_per_test");
  }

  @Test
  public void testBeforeCommandGrpcReportsBesResultsUrl() throws Exception {
    runBuildWithOptions(
        "--bes_backend=inprocess",
        "--bes_upload_mode=FULLY_ASYNC",
        "--bes_results_url=http://results-ui/");
    events.assertContainsEventsInOrder(
        "Streaming build results to: http://results-ui/", "Found 0 targets", "Found 0 targets");
  }

  @Test
  public void testAfterCommandGrpcReportsBesResultsUrl() throws Exception {
    runBuildWithOptions(
        "--bes_backend=inprocess",
        "--bes_upload_mode=FULLY_ASYNC",
        "--bes_results_url=http://results-ui/");
    afterBuildCommand();

    events.assertContainsEventsInOrder(
        "Streaming build results to: http://results-ui/",
        "Found 0 targets",
        "Found 0 targets",
        "Streaming build results to: http://results-ui/",
        "Streaming build results to: http://results-ui/");
  }

  @Test
  public void testAfterCommand_waitForUploadComplete() throws Exception {
    buildEventService.setDelayBeforeClosingStream(Duration.ZERO);
    runBuildWithOptions(
        "--bes_backend=inprocess",
        "--bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE",
        "--bes_timeout=5s");
    afterBuildCommand();
    events.assertNoWarningsOrErrors();
  }

  @Test
  public void testAfterCommand_waitForUploadComplete_slowFullCloseError() throws Exception {
    buildEventService.setDelayBeforeClosingStream(Duration.ofSeconds(10));
    runBuildWithOptions(
        "--bes_backend=inprocess",
        "--bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE",
        "--bes_timeout=5s");
    ImmutableSet<BuildEventTransport> bepTransports = besModule.getBepTransports();
    assertThat(bepTransports).hasSize(1);
    afterBuildCommand();
    assertContainsError("The Build Event Protocol upload timed out");
    for (BuildEventTransport bepTransport : bepTransports) {
      assertThat(bepTransport.close().isDone()).isTrue();
    }
  }

  @Test
  public void testAfterCommand_waitForUploadComplete_slowHalfCloseError() throws Exception {
    buildEventService.setDelayBeforeHalfClosingStream(Duration.ofSeconds(10));
    runBuildWithOptions(
        "--bes_backend=inprocess",
        "--bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE",
        "--bes_timeout=5s");
    afterBuildCommand();
    assertContainsError("The Build Event Protocol upload timed out");
  }

  @Test
  public void testAfterCommand_noWaitForUploadComplete() throws Exception {
    buildEventService.setDelayBeforeClosingStream(Duration.ZERO);
    runBuildWithOptions("--bes_backend=inprocess", "--bes_upload_mode=NOWAIT_FOR_UPLOAD_COMPLETE");
    afterBuildCommand();
    events.assertNoWarningsOrErrors();
  }

  @Test
  public void testAfterCommand_noWaitForUploadComplete_slowFullCloseIgnored() throws Exception {
    buildEventService.setDelayBeforeClosingStream(Duration.ofSeconds(10));
    runBuildWithOptions("--bes_backend=inprocess", "--bes_upload_mode=NOWAIT_FOR_UPLOAD_COMPLETE");
    afterBuildCommand();
    events.assertNoWarningsOrErrors();
  }

  @Test
  public void testAfterCommand_noWaitForUploadComplete_slowHalfCloseIgnored() throws Exception {
    buildEventService.setDelayBeforeHalfClosingStream(Duration.ofSeconds(10));
    runBuildWithOptions("--bes_backend=inprocess", "--bes_upload_mode=NOWAIT_FOR_UPLOAD_COMPLETE");
    afterBuildCommand();
    events.assertNoWarningsOrErrors();
  }

  @Test
  public void testBeforeSecondCommand_noWaitForUploadComplete_slowFullCloseWarning()
      throws Exception {
    buildEventService.setDelayBeforeClosingStream(Duration.ofSeconds(10));
    runBuildWithOptions("--bes_backend=inprocess", "--bes_upload_mode=NOWAIT_FOR_UPLOAD_COMPLETE");
    afterBuildCommand();
    buildTarget();
    events.assertContainsWarning(
        "The background upload of the Build Event Protocol for the previous "
            + "invocation failed to complete in");
  }

  @Test
  public void testBeforeSecondCommand_noWaitForUploadComplete_slowHalfCloseWarning()
      throws Exception {
    buildEventService.setDelayBeforeHalfClosingStream(Duration.ofSeconds(10));
    runBuildWithOptions("--bes_backend=inprocess", "--bes_upload_mode=NOWAIT_FOR_UPLOAD_COMPLETE");
    afterBuildCommand();
    buildTarget();
    events.assertContainsWarning(
        "The background upload of the Build Event Protocol for the previous "
            + "invocation failed to complete in");
  }

  @Test
  public void testBeforeSecondCommand_noWaitForUploadComplete_besTimeout_slowFullCloseWarning()
      throws Exception {
    buildEventService.setDelayBeforeClosingStream(Duration.ofSeconds(10));
    runBuildWithOptions(
        "--bes_backend=inprocess",
        "--bes_upload_mode=NOWAIT_FOR_UPLOAD_COMPLETE",
        "--bes_timeout=1s");
    afterBuildCommand();
    buildTarget();
    events.assertContainsWarning(
        "The background upload of the Build Event Protocol for the previous "
            + "invocation failed due to a network timeout");
  }

  @Test
  public void testBeforeSecondCommand_noWaitForUpload_besTimeout_slowHalfCloseWarning()
      throws Exception {
    buildEventService.setDelayBeforeHalfClosingStream(Duration.ofSeconds(10));
    runBuildWithOptions(
        "--bes_backend=inprocess",
        "--bes_upload_mode=NOWAIT_FOR_UPLOAD_COMPLETE",
        "--bes_timeout=1s");
    afterBuildCommand();
    buildTarget();
    events.assertContainsWarning(
        "The background upload of the Build Event Protocol for the previous "
            + "invocation failed due to a network timeout");
  }

  @Test
  public void testAfterCommand_fullyAsync() throws Exception {
    runBuildWithOptions("--bes_backend=inprocess", "--bes_upload_mode=FULLY_ASYNC");
    afterBuildCommand();
    events.assertNoWarningsOrErrors();
  }

  @Test
  public void testAfterCommand_fullyAsync_slowHalfCloseIgnored() throws Exception {
    buildEventService.setDelayBeforeHalfClosingStream(Duration.ofSeconds(10));
    runBuildWithOptions("--bes_backend=inprocess", "--bes_upload_mode=FULLY_ASYNC");
    afterBuildCommand();
    events.assertNoWarningsOrErrors();
  }

  @Test
  public void testBeforeSecondCommand_fullyAsync_slowFullCloseIgnored() throws Exception {
    buildEventService.setDelayBeforeClosingStream(Duration.ofSeconds(10));
    runBuildWithOptions("--bes_backend=inprocess", "--bes_upload_mode=FULLY_ASYNC");
    afterBuildCommand();
    buildTarget();
    events.assertNoWarningsOrErrors();
  }

  // TODO(b/246912214): Deflake this by fixing the threading model to match the upstream gRPC
  // changes in https://github.com/grpc/grpc-java/pull/9319 that affect InProcessTransport.
  @Ignore("b/246912214")
  @Test
  public void testBeforeSecondCommand_fullyAsync_slowHalfCloseWarning() throws Exception {
    buildEventService.setDelayBeforeHalfClosingStream(Duration.ofSeconds(10));
    runBuildWithOptions("--bes_backend=inprocess", "--bes_upload_mode=FULLY_ASYNC");
    afterBuildCommand();
    buildTarget();
    events.assertContainsWarning(
        "The background upload of the Build Event Protocol for the previous "
            + "invocation failed to complete in");
  }

  @Test
  public void testBeforeSecondCommand_fullyAsync_besTimeout_slowFullCloseIgnored()
      throws Exception {
    buildEventService.setDelayBeforeClosingStream(Duration.ofSeconds(10));
    runBuildWithOptions(
        "--bes_backend=inprocess", "--bes_upload_mode=FULLY_ASYNC", "--bes_timeout=1s");
    afterBuildCommand();
    buildTarget();
    events.assertNoWarningsOrErrors();
  }

  // TODO(b/246912214): Deflake this by fixing the threading model to match the upstream gRPC
  // changes in https://github.com/grpc/grpc-java/pull/9319 that affect InProcessTransport.
  @Ignore("b/246912214")
  @Test
  public void testBeforeSecondCommand_fullyAsync_besTimeout_slowHalfCloseWarning()
      throws Exception {
    buildEventService.setDelayBeforeHalfClosingStream(Duration.ofSeconds(10));
    runBuildWithOptions(
        "--bes_backend=inprocess", "--bes_upload_mode=FULLY_ASYNC", "--bes_timeout=1s");
    afterBuildCommand();
    buildTarget();
    events.assertContainsWarning(
        "The background upload of the Build Event Protocol for the previous "
            + "invocation failed due to a network timeout.");
  }

  @Test
  public void testAfterCommandStreamerIsClosedNoWarning() throws Exception {
    runBuildWithOptions("--build_event_text_file=" + tmpFolder.newFile().getAbsolutePath());
    assertThat(besModule.getBepTransports()).hasSize(1);
    afterBuildCommand();
    events.assertNoWarningsOrErrors();
  }

  @Test
  public void testAfterCommand_waitForUploadComplete_errorOnComplete() throws Exception {
    buildEventService.setErrorMessage("Boom1");
    runBuildWithOptions("--bes_backend=inprocess", "--bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE");
    afterBuildCommand();
    events.assertContainsError("The Build Event Protocol upload failed: Boom1");
  }

  @Test
  public void testAfterCommand_waitForUploadComplete_besTimeout_errorOnComplete() throws Exception {
    buildEventService.setErrorMessage("Boom2");
    runBuildWithOptions(
        "--bes_backend=inprocess",
        "--bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE",
        "--bes_timeout=5s");
    afterBuildCommand();
    events.assertContainsError("The Build Event Protocol upload failed: Boom2");
  }

  @Test
  public void testAfterCommand_noWaitForUploadComplete_errorOnComplete() throws Exception {
    buildEventService.setErrorMessage("Boom3");
    runBuildWithOptions("--bes_backend=inprocess", "--bes_upload_mode=NOWAIT_FOR_UPLOAD_COMPLETE");
    afterBuildCommand();
    events.assertNoWarningsOrErrors();
  }

  @Test
  public void testBeforeSecondCommand_noWaitForUploadComplete_errorOnComplete() throws Exception {
    buildEventService.setErrorMessage("Boom4");
    runBuildWithOptions("--bes_backend=inprocess", "--bes_upload_mode=NOWAIT_FOR_UPLOAD_COMPLETE");
    afterBuildCommand();
    buildTarget();
    events.assertContainsWarning("The Build Event Protocol upload failed: Boom4");
  }

  @Test
  public void testAfterCommand_fullyAsync_errorOnComplete() throws Exception {
    buildEventService.setErrorMessage("Boom5");
    runBuildWithOptions("--bes_backend=inprocess", "--bes_upload_mode=FULLY_ASYNC");
    afterBuildCommand();
    events.assertNoWarningsOrErrors();
  }

  @Test
  public void testBeforeSecondCommand_fullyAsync_errorOnComplete() throws Exception {
    buildEventService.setErrorMessage("Boom6");
    runBuildWithOptions("--bes_backend=inprocess", "--bes_upload_mode=FULLY_ASYNC");
    afterBuildCommand();
    buildTarget();
    events.assertNoWarningsOrErrors();
  }

  @Test
  public void testCreatesStreamerForAllTransports() throws Exception {
    runBuildWithOptions(
        "--build_event_text_file=" + tmpFolder.newFile().getAbsolutePath(),
        "--build_event_binary_file=" + tmpFolder.newFile().getAbsolutePath(),
        "--build_event_json_file=" + tmpFolder.newFile().getAbsolutePath(),
        "--bes_backend=does.not.exist:1234");

    assertThat(besModule.getBepTransports()).hasSize(4);
    assertThat(besModule.getBepTransports().asList().get(0))
        .isInstanceOf(TextFormatFileTransport.class);
    assertThat(besModule.getBepTransports().asList().get(1))
        .isInstanceOf(BinaryFormatFileTransport.class);
    assertThat(besModule.getBepTransports().asList().get(2))
        .isInstanceOf(JsonFormatFileTransport.class);
    assertThat(besModule.getBepTransports().asList().get(3))
        .isInstanceOf(BuildEventServiceTransport.class);
  }

  @Test
  public void testUploaderSharing() throws Exception {
    runBuildWithOptions(
        "--build_event_text_file=" + tmpFolder.newFile().getAbsolutePath(),
        "--build_event_binary_file=" + tmpFolder.newFile().getAbsolutePath(),
        "--build_event_json_file=" + tmpFolder.newFile().getAbsolutePath(),
        "--bes_backend=does.not.exist:1234");

    assertThat(besModule.getBepTransports()).hasSize(4);

    BuildEventArtifactUploader uploader =
        Iterables.getFirst(besModule.getBepTransports(), null).getUploader();
    assertThat(uploader).isNotNull();
    for (BuildEventTransport transport : besModule.getBepTransports()) {
      assertThat(uploader).isSameInstanceAs(transport.getUploader());
    }
  }

  @Test
  public void testDoesNotCreatesStreamerWithoutTransports() throws Exception {
    runBuildWithOptions();
    assertThat(besModule.getBepTransports()).isEmpty();
  }

  @Test
  public void testKeywords() throws Exception {
    runBuildWithOptions();
    BuildEventServiceOptions besOptions = new BuildEventServiceOptions();
    besOptions.besKeywords = ImmutableList.of("keyword0", "keyword1", "keyword0");
    besOptions.besSystemKeywords = ImmutableList.of("sys_keyword0", "sys_keyword1", "sys_keyword0");

    assertThat(besModule.getBesKeywords(besOptions, null))
        .containsExactly(
            "user_keyword=keyword0", "user_keyword=keyword1", "sys_keyword0", "sys_keyword1");
  }

  @Test
  public void testMakeGrpcMetadata() throws Exception {
    runBuildWithOptions();
    BuildEventServiceOptions besOptions = new BuildEventServiceOptions();
    AuthAndTLSOptions authAndTLSOptions = new AuthAndTLSOptions();
    besOptions.besBackend = "bes-backend";
    besOptions.besProxy = "bes-proxy";
    besOptions.besHeaders =
        ImmutableMap.of("key1", "val1", "key2", "val2", "key3", "val3").entrySet().asList();
    BackendConfig newConfig = BackendConfig.create(besOptions, authAndTLSOptions);

    Metadata metadata = BazelBuildEventServiceModule.makeGrpcMetadata(newConfig);
    assertThat(metadata.get(Metadata.Key.of("key1", Metadata.ASCII_STRING_MARSHALLER)))
        .isEqualTo("val1");
    assertThat(metadata.get(Metadata.Key.of("key2", Metadata.ASCII_STRING_MARSHALLER)))
        .isEqualTo("val2");
    assertThat(metadata.get(Metadata.Key.of("key3", Metadata.ASCII_STRING_MARSHALLER)))
        .isEqualTo("val3");
  }

  /** Regression test for b/111653523. */
  @Test
  public void testCoverageFileIncluded() throws Exception {
    assumeFalse(AnalysisMock.get().isThisBazel());
    // Test aims to ensure that the TargetCompleted event for "//foo:foo_lib" includes the
    // "baseline_coverage.dat" file in its "baseline.lcov" output group.

    write("foo/BUILD", "cc_library(name = 'foo_lib', srcs = ['foo.cc'])");
    write("foo/foo.cc");
    File buildEventBinaryFile = tmpFolder.newFile();
    addOptions(
        "--build_event_binary_file=" + buildEventBinaryFile.getAbsolutePath(),
        "--collect_code_coverage",
        "--bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE");

    buildTarget("//foo:foo_lib");
    // We need to wait for all events to be written to the file, which is done in #afterCommand()
    // if --bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE.
    afterBuildCommand();

    List<BuildEvent> buildEvents = new ArrayList<>();
    try (InputStream in = new FileInputStream(buildEventBinaryFile)) {
      while (in.available() > 0) {
        buildEvents.add(BuildEvent.parseDelimitedFrom(in));
      }
    }

    // Find all the NamedSetOfFiles events and the OutputGroup named "baseline.lcov" for the
    // target "//foo:foo_lib".
    Map<String, NamedSetOfFiles> namedSets = new HashMap<>();
    OutputGroup coverageOutputGroup = null;
    for (BuildEvent buildEvent : buildEvents) {
      switch (buildEvent.getId().getIdCase()) {
        case NAMED_SET:
          namedSets.put(buildEvent.getId().getNamedSet().getId(), buildEvent.getNamedSetOfFiles());
          break;
        case TARGET_COMPLETED:
          if (buildEvent.getId().getTargetCompleted().getLabel().equals("//foo:foo_lib")) {
            for (OutputGroup outputGroup : buildEvent.getCompleted().getOutputGroupList()) {
              if (outputGroup.getName().equals("baseline.lcov")) {
                coverageOutputGroup = outputGroup;
              }
            }
          }
          break;
        default:
          break;
      }
    }
    assertThat(coverageOutputGroup).isNotNull();

    BuildEventStreamProtos.File baselineCoverageFile =
        findFileInNamedSets(namedSets, coverageOutputGroup, "foo/foo_lib/baseline_coverage.dat");
    assertThat(baselineCoverageFile).isNotNull();
  }

  /**
   * Recursively walks through NamedSetOfFiles events looking for a file with a given name, starting
   * with the file sets in a given output group.
   */
  @Nullable
  private static BuildEventStreamProtos.File findFileInNamedSets(
      Map<String, NamedSetOfFiles> namedSets,
      OutputGroup coverageOutputGroup,
      String fileNameToFind) {
    Deque<String> visit = new ArrayDeque<>();
    for (NamedSetOfFilesId namedSetOfFilesId : coverageOutputGroup.getFileSetsList()) {
      visit.add(namedSetOfFilesId.getId());
    }
    Set<String> seen = new HashSet<>(visit);
    while (!visit.isEmpty()) {
      String id = visit.removeFirst();
      NamedSetOfFiles set = namedSets.get(id);
      for (BuildEventStreamProtos.File file : set.getFilesList()) {
        if (file.getName().equals(fileNameToFind)) {
          return file;
        }
      }
      for (NamedSetOfFilesId transitiveSet : set.getFileSetsList()) {
        if (seen.add(transitiveSet.getId())) {
          visit.addLast(transitiveSet.getId());
        }
      }
    }
    return null;
  }

  @Test
  public void oom_firstReportedViaHandleCrash() throws Exception {
    testOom(
        () -> {
          OutOfMemoryError oom = new OutOfMemoryError();
          // Simulates an OOM coming from GcThrashingDetector, which reports the error by calling
          // handleCrash. Uses keepAlive() to avoid exiting the JVM and aborting the test, then
          // throw the original oom to ensure control flow terminates.
          BugReport.handleCrash(Crash.from(oom), CrashContext.keepAlive());
          throw oom;
        });
  }

  @Test
  public void oom_firstThrownFromSkyframe() throws Exception {
    testOom(
        () -> {
          throw new OutOfMemoryError();
        });
  }

  private void testOom(Runnable throwOom) throws Exception {
    write("foo/BUILD", "genrule(name = 'gen', outs = ['gen.out'], cmd = 'touch $@')");
    AtomicBoolean threwOom = new AtomicBoolean(false);
    getSkyframeExecutor()
        .getEvaluator()
        .injectGraphTransformerForTesting(
            NotifyingHelper.makeNotifyingTransformer(
                // To get the right configuration, some analysis has to already been done.
                // We're only throwing OOM here for non shareable ActionLookupData to exclude
                // workspace status actions, which in Skymeld mode can run without any analysis.
                (key, type, order, context) -> {
                  if (key instanceof ActionLookupData
                      && key.valueIsShareable()
                      && !threwOom.getAndSet(true)) {
                    throwOom.run();
                  }
                }));
    File buildEventBinaryFile = tmpFolder.newFile();
    addOptions(
        "--build_event_binary_file=" + buildEventBinaryFile.getAbsolutePath(),
        "--oom_message=Please build fewer targets.");

    assertThrows(OutOfMemoryError.class, () -> buildTarget("//foo:gen"));

    List<BuildEvent> buildEvents = new ArrayList<>();
    try (InputStream in = new FileInputStream(buildEventBinaryFile)) {
      while (in.available() > 0) {
        buildEvents.add(BuildEvent.parseDelimitedFrom(in));
      }
    }
    Aborted expectedAbort =
        Aborted.newBuilder()
            .setReason(AbortReason.OUT_OF_MEMORY)
            .setDescription(BugReport.constructOomExitMessage("Please build fewer targets."))
            .build();
    assertThat(buildEvents)
        .ignoringFields(BuildEvent.LAST_MESSAGE_FIELD_NUMBER)
        .containsAtLeast(
            BuildEvent.newBuilder()
                .setId(
                    BuildEventId.newBuilder()
                        .setBuildFinished(BuildFinishedId.getDefaultInstance()))
                .setAborted(expectedAbort)
                .build(),
            BuildEvent.newBuilder()
                .setId(
                    BuildEventId.newBuilder()
                        .setTargetCompleted(
                            TargetCompletedId.newBuilder()
                                .setLabel("//foo:gen")
                                .setConfiguration(
                                    ConfigurationId.newBuilder()
                                        .setId(
                                            getConfiguredTarget("//foo:gen")
                                                .getConfigurationChecksum()))))
                .setAborted(expectedAbort)
                .build());
    assertThat(runtimeWrapper.getCrashMessages())
        .containsExactly(
            TestConstants.PRODUCT_NAME + " is crashing: Crashed: (java.lang.OutOfMemoryError) ");
    assertAndClearBugReporterStoredCrash(OutOfMemoryError.class);
  }

  @Test
  public void oom_besClosesAfterSpecialCaseTimeoutThrownFromSkyframe() throws Exception {
    // BES server-side will never finish. The test will pass simply by completing and not waiting
    // until the test timeout.
    buildEventService.setDelayBeforeClosingStream(Duration.ofHours(10));
    write("foo/BUILD", "genrule(name = 'gen', outs = ['gen.out'], cmd = 'touch $@')");
    AtomicBoolean threwOom = new AtomicBoolean(false);
    getSkyframeExecutor()
        .getEvaluator()
        .injectGraphTransformerForTesting(
            NotifyingHelper.makeNotifyingTransformer(
                (key, type, order, context) -> {
                  if (key instanceof ActionLookupData && !threwOom.getAndSet(true)) {
                    throw new OutOfMemoryError();
                  }
                }));
    addOptions(
        "--bes_backend=inprocess",
        "--bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE",
        "--bes_oom_finish_upload_timeout=2s",
        "--oom_message=Please build fewer targets.");

    assertThrows(OutOfMemoryError.class, () -> buildTarget("//foo:gen"));

    assertThat(runtimeWrapper.getCrashMessages())
        .containsExactly(
            TestConstants.PRODUCT_NAME + " is crashing: Crashed: (java.lang.OutOfMemoryError) ");
    assertAndClearBugReporterStoredCrash(OutOfMemoryError.class);
  }

  /**
   * Trivial, in-memory implementation of a PublishBuildToolEventStream handler that can have
   * pre-configured sleeps triggered at critical junctures.
   */
  private static class DelayingPublishBuildStreamObserver
      implements StreamObserver<PublishBuildToolEventStreamRequest> {

    private final StreamObserver<PublishBuildToolEventStreamResponse> responseObserver;
    private final Duration delayBeforeClosingStream;
    private final Duration delayBeforeHalfClosingStream;

    @GuardedBy("this")
    private final SortedSet<Long> unackedSequenceNumbers = Sets.newTreeSet();

    private final BlockingQueue<Long> ackQueue = new ArrayBlockingQueue<>(10);

    @GuardedBy("this")
    private Thread ackingThread = null;

    @GuardedBy("this")
    private StreamId streamId = null;

    @GuardedBy("this")
    private boolean finished = false;

    private DelayingPublishBuildStreamObserver(
        StreamObserver<PublishBuildToolEventStreamResponse> responseObserver,
        Duration delayBeforeClosingStream,
        Duration delayBeforeHalfClosingStream) {
      this.responseObserver = responseObserver;
      this.delayBeforeClosingStream = delayBeforeClosingStream;
      this.delayBeforeHalfClosingStream = delayBeforeHalfClosingStream;
    }

    /** Creates the acking thread, safely callable after the constructor finishes. */
    synchronized void startAckingThread() {
      Preconditions.checkState(ackingThread == null, "startAckingThread() called twice");
      ackingThread = new Thread(new AckingThread());
      ackingThread.start();
    }

    @Override
    public void onNext(PublishBuildToolEventStreamRequest req) {
      List<Long> longsToPut = new ArrayList<>();
      synchronized (this) {
        if (!unackedSequenceNumbers.add(req.getOrderedBuildEvent().getSequenceNumber())) {
          return; // dupe, ignore
        }
        streamId = MoreObjects.firstNonNull(streamId, req.getOrderedBuildEvent().getStreamId());
        if (req.getOrderedBuildEvent().getEvent().getComponentStreamFinished().getType()
            == FinishType.FINISH_TYPE_UNSPECIFIED) {
          // We did not get the final event. Ack the *previous* event, if there is a previous event.
          if (unackedSequenceNumbers.size() > 1) {
            longsToPut.add(ackLowestSequenceNumber());
          }
        } else {
          Uninterruptibles.sleepUninterruptibly(delayBeforeHalfClosingStream);
          // final event. ack everything remaining.
          while (!unackedSequenceNumbers.isEmpty()) {
            longsToPut.add(ackLowestSequenceNumber());
          }
          if (finished) {
            longsToPut.add(SENTINEL_VALUE);
          }
        }
      }
      for (Long seqNum : longsToPut) {
        Uninterruptibles.putUninterruptibly(ackQueue, seqNum);
      }
    }

    @GuardedBy("this")
    private Long ackLowestSequenceNumber() {
      Long firstUnacked = unackedSequenceNumbers.first();
      unackedSequenceNumbers.remove(firstUnacked);
      return firstUnacked;
    }

    @Override
    public synchronized void onError(Throwable t) {
      finished = true;
      responseObserver.onError(t);
    }

    @Override
    public void onCompleted() {
      boolean putSentinel;
      synchronized (this) {
        finished = true;
        putSentinel = unackedSequenceNumbers.isEmpty();
      }
      if (putSentinel) {
        Uninterruptibles.putUninterruptibly(ackQueue, SENTINEL_VALUE);
      }
    }

    static final Long SENTINEL_VALUE = -1L;

    private class AckingThread implements Runnable {

      @Override
      public void run() {
        while (true) {
          Long firstUnacked = Uninterruptibles.takeUninterruptibly(ackQueue);
          synchronized (DelayingPublishBuildStreamObserver.this) {
            if (firstUnacked.equals(SENTINEL_VALUE)) {
              Uninterruptibles.sleepUninterruptibly(delayBeforeClosingStream);
              responseObserver.onCompleted();
              return;
            }
            responseObserver.onNext(
                PublishBuildToolEventStreamResponse.newBuilder()
                    .setStreamId(streamId)
                    .setSequenceNumber(firstUnacked)
                    .build());
          }
        }
      }
    }
  }

  /**
   * Trivial implementation of {@link PublishBuildEventImplBase} that can insert sleeps at critical
   * junctures.
   */
  private static final class DelayingPublishBuildEventService extends PublishBuildEventImplBase {

    @GuardedBy("this")
    private Duration delayBeforeClosingStream = Duration.ZERO;

    @GuardedBy("this")
    private Duration delayBeforeHalfClosingStream = Duration.ZERO;

    @GuardedBy("this")
    @Nullable
    private String errorMessage = null;

    /**
     * Synchronizing this method can lead to deadlocks -- it calls into {@link
     * io.grpc.inprocess.InProcessTransport} which takes a locks on itself. Opposite order of locks
     * happens for {@link #publishBuildToolEventStream} called while holding the lock on {@link
     * io.grpc.inprocess.InProcessTransport}.
     */
    @Override
    public void publishLifecycleEvent(
        PublishLifecycleEventRequest request, StreamObserver<Empty> responseObserver) {
      RequestMetadata metadata = TracingMetadataUtils.fromCurrentContext();
      assertThat(metadata.getToolInvocationId()).isNotEmpty();
      assertThat(metadata.getCorrelatedInvocationsId()).isNotEmpty();
      assertThat(metadata.getActionId()).isEqualTo("publish_lifecycle_event");

      responseObserver.onNext(Empty.getDefaultInstance());
      responseObserver.onCompleted();
    }

    @Override
    public synchronized StreamObserver<PublishBuildToolEventStreamRequest>
        publishBuildToolEventStream(
            StreamObserver<PublishBuildToolEventStreamResponse> responseObserver) {
      RequestMetadata metadata = TracingMetadataUtils.fromCurrentContext();
      assertThat(metadata.getToolInvocationId()).isNotEmpty();
      assertThat(metadata.getCorrelatedInvocationsId()).isNotEmpty();
      assertThat(metadata.getActionId()).isEqualTo("publish_build_tool_event_stream");

      if (errorMessage != null) {
        return new ErroringPublishBuildStreamObserver(responseObserver, errorMessage);
      }
      DelayingPublishBuildStreamObserver observer =
          new DelayingPublishBuildStreamObserver(
              responseObserver, delayBeforeClosingStream, delayBeforeHalfClosingStream);
      observer.startAckingThread();
      return observer;
    }

    synchronized void setErrorMessage(String errorMessage) {
      this.errorMessage = errorMessage;
    }

    synchronized void setDelayBeforeClosingStream(Duration delay) {
      this.delayBeforeClosingStream = delay;
    }

    synchronized void setDelayBeforeHalfClosingStream(Duration delay) {
      this.delayBeforeHalfClosingStream = delay;
    }
  }

  private static final class ErroringPublishBuildStreamObserver
      implements StreamObserver<PublishBuildToolEventStreamRequest> {

    private final StreamObserver<PublishBuildToolEventStreamResponse> responseObserver;
    private final String errorMessage;

    ErroringPublishBuildStreamObserver(
        StreamObserver<PublishBuildToolEventStreamResponse> responseObserver, String errorMessage) {
      this.responseObserver = responseObserver;
      this.errorMessage = errorMessage;
    }

    @Override
    public void onNext(PublishBuildToolEventStreamRequest value) {
      responseObserver.onNext(
          PublishBuildToolEventStreamResponse.newBuilder()
              .setStreamId(value.getOrderedBuildEventOrBuilder().getStreamId())
              .setSequenceNumber(value.getOrderedBuildEvent().getSequenceNumber())
              .build());
    }

    @Override
    public void onError(Throwable t) {}

    @Override
    public void onCompleted() {
      responseObserver.onError(
          new StatusRuntimeException(Status.DATA_LOSS.withDescription(errorMessage)));
    }
  }
}
