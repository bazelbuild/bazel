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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;
import static com.google.devtools.build.lib.buildeventservice.BuildEventServiceModule.RUNS_PER_TEST_LIMIT;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.Subscribe;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialModule;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.bugreport.Crash;
import com.google.devtools.build.lib.bugreport.CrashContext;
import com.google.devtools.build.lib.buildeventservice.BazelBuildEventServiceModule.BackendConfig;
import com.google.devtools.build.lib.buildeventservice.BuildEventServiceModule.BuildEventFileType;
import com.google.devtools.build.lib.buildeventservice.BuildEventServiceModule.BuildEventOutputStreamFactory;
import com.google.devtools.build.lib.buildeventstream.AnnounceBuildEventTransportsEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.Aborted;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.Aborted.AbortReason;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.BuildFinishedId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ConfigurationId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.TargetCompletedId;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.transports.BinaryFormatFileTransport;
import com.google.devtools.build.lib.buildeventstream.transports.JsonFormatFileTransport;
import com.google.devtools.build.lib.buildeventstream.transports.TextFormatFileTransport;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.network.ConnectivityStatus;
import com.google.devtools.build.lib.network.ConnectivityStatusProvider;
import com.google.devtools.build.lib.network.NoOpConnectivityModule;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.NoSpawnCacheModule;
import com.google.devtools.build.lib.runtime.proto.CommandLineOuterClass.CommandLineSection;
import com.google.devtools.build.lib.runtime.proto.CommandLineOuterClass.Option;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.skyframe.NotifyingHelper;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import io.grpc.ManagedChannel;
import io.grpc.Metadata;
import io.grpc.Server;
import io.grpc.ServerInterceptors;
import io.grpc.Status;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.util.MutableHandlerRegistry;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.Thread.UncaughtExceptionHandler;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;

/** Tests for {@link BazelBuildEventServiceModule}. */
@RunWith(TestParameterInjector.class)
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

  @Nullable private BuildEventOutputStreamFactory buildEventOutputStreamFactory;

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

              @Override
              BuildEventOutputStreamFactory createBuildEventOutputStreamFactory(
                  CommandEnvironment env) {
                return buildEventOutputStreamFactory == null
                    ? super.createBuildEventOutputStreamFactory(env)
                    : buildEventOutputStreamFactory;
              }
            });
  }

  private ImmutableSet<BuildEventTransport> bepTransports;

  private class BepTransportLogger {
    @Subscribe
    @SuppressWarnings("unused")
    public void transportsKnown(AnnounceBuildEventTransportsEvent event) {
      bepTransports = besModule.getBepTransports();
    }
  }

  private ImmutableSet<BuildEventTransport> getBepTransports() {
    return bepTransports;
  }

  private void runBuildWithOptions(String... options) throws Exception {
    addOptions(options);
    besModule = runtimeWrapper.getRuntime().getBlazeModule(BazelBuildEventServiceModule.class);
    if (buildEventOutputStreamFactory != null) {
      besModule.setBuildEventOutputStreamFactory(buildEventOutputStreamFactory);
    }
    runtimeWrapper.newCommand();
    runtimeWrapper.getSkyframeExecutor().getEventBus().register(new BepTransportLogger());
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
    assertThat(getBepTransports()).hasSize(1);
    assertThat(getBepTransports().asList().get(0)).isInstanceOf(TextFormatFileTransport.class);
  }

  @Test
  public void testCreatesStreamerForBinaryFormatFileTransport() throws Exception {
    runBuildWithOptions("--build_event_binary_file=" + tmpFolder.newFile().getAbsolutePath());
    assertThat(getBepTransports()).hasSize(1);
    assertThat(getBepTransports().asList().get(0)).isInstanceOf(BinaryFormatFileTransport.class);
  }

  @Test
  public void testCreatesStreamerForJsonFormatFileTransport() throws Exception {
    runBuildWithOptions("--build_event_json_file=" + tmpFolder.newFile().getAbsolutePath());
    assertThat(getBepTransports()).hasSize(1);
    assertThat(getBepTransports().asList().get(0)).isInstanceOf(JsonFormatFileTransport.class);
  }

  @Test
  public void testCreatesStreamerForBesTransport() throws Exception {
    runBuildWithOptions("--bes_backend=does.not.exist:1234");
    assertThat(getBepTransports()).hasSize(1);
    assertThat(getBepTransports().asList().get(0)).isInstanceOf(BuildEventServiceTransport.class);
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
    reinitializeAndPreserveOptions();
    addOptions("--bes_backend=does.not.exist:1234");
    addOptions("--spawn_strategy=standalone");
    runBuildWithOptions();
    assertThat(getBepTransports()).isEmpty();
  }

  @Test
  public void testCreatesStreamerForGrpcBesResultsUrl() throws Exception {
    runBuildWithOptions(
        "--bes_backend=inprocess",
        "--bes_upload_mode=FULLY_ASYNC",
        "--bes_results_url=http://results-ui/");

    assertThat(getBepTransports()).hasSize(1);
    assertThat(getBepTransports().asList().get(0)).isInstanceOf(BuildEventServiceTransport.class);
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
    assertThat(getBepTransports()).isEmpty();
    assertContainsError("The value of --runs_per_test");
  }

  @Test
  public void testBeforeCommandGrpcReportsBesResultsUrl() throws Exception {
    runBuildWithOptions(
        "--color=no", // disable ANSI color sequences
        "--bes_backend=inprocess",
        "--bes_upload_mode=FULLY_ASYNC",
        "--bes_results_url=http://results-ui/");
    events.assertContainsEventsInOrder(
        "Streaming build results to: http://results-ui/", "Found 0 targets", "Found 0 targets");
  }

  @Test
  public void testAfterCommandGrpcReportsBesResultsUrl() throws Exception {
    runBuildWithOptions(
        "--color=no", // disable ANSI color sequences
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
    ImmutableSet<BuildEventTransport> bepTransports = getBepTransports();
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

  private static String getBuildEventFileFlag(
      BuildEventFileType buildEventFileType, String filePath) {
    return switch (buildEventFileType) {
      case TEXT -> "--build_event_text_file=" + filePath;
      case JSON -> "--build_event_json_file=" + filePath;
      case BINARY -> "--build_event_binary_file=" + filePath;
    };
  }

  private static String getBuildEventFileUploadModeFlag(
      BuildEventFileType buildEventFileType, String mode) {
    return switch (buildEventFileType) {
      case TEXT -> "--build_event_text_file_upload_mode=" + mode;
      case JSON -> "--build_event_json_file_upload_mode=" + mode;
      case BINARY -> "--build_event_binary_file_upload_mode=" + mode;
    };
  }

  @Test
  public void testAfterCommand_buildEventFile_waitForUploadComplete(
      @TestParameter BuildEventFileType buildEventFileType) throws Exception {
    AtomicReference<DelayingCloseBufferedOutputStream> outRef = new AtomicReference<>(null);
    buildEventOutputStreamFactory =
        (type, filePath) -> {
          var out =
              new DelayingCloseBufferedOutputStream(
                  Files.newOutputStream(Path.of(filePath)), Duration.ofSeconds(1));
          outRef.set(out);
          return out;
        };
    buildEventService.setDelayBeforeClosingStream(Duration.ofSeconds(10));
    var file = tmpFolder.newFile();

    runBuildWithOptions(
        "--bes_backend=inprocess",
        "--bes_upload_mode=FULLY_ASYNC",
        "--bes_timeout=1s",
        getBuildEventFileFlag(buildEventFileType, file.getAbsolutePath()),
        getBuildEventFileUploadModeFlag(buildEventFileType, "wait_for_upload_complete"));
    afterBuildCommand();

    assertThat(outRef.get().isClosed()).isTrue();
    // Expect Bazel doesn't wait for uploading to bes_backend, otherwise there will be a timeout
    // error.
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
    assertThat(getBepTransports()).hasSize(1);
    afterBuildCommand();
    events.assertNoWarningsOrErrors();
  }

  @Test
  public void testAfterCommand_waitForUploadComplete_retryableErrorEarlyInStream()
      throws Exception {
    int numRetries = 3;
    buildEventService.setErrorMessageAndCode("Boom8", Status.UNAVAILABLE);
    buildEventService.setErrorEarlyInStream(true);
    runBuildWithOptions(
        "--bes_backend=inprocess",
        "--bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE",
        "--build_event_upload_max_retries=" + numRetries);
    afterBuildCommand();
    events.assertContainsError(
        Pattern.compile(
            "The Build Event Protocol upload failed: Not retrying publishBuildEvents, no more"
                + " attempts left.*Boom8"));
    assertThat(buildEventService.getRequestsReceivedCount()).isEqualTo(numRetries + 1);
  }

  @Test
  public void testAfterCommand_waitForUploadComplete_permissionDeniedErrorEarlyInStream()
      throws Exception {
    int numRetries = 3;
    buildEventService.setErrorMessageAndCode("Boom15", Status.PERMISSION_DENIED);
    buildEventService.setErrorEarlyInStream(true);
    runBuildWithOptions(
        "--bes_backend=inprocess",
        "--bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE",
        "--build_event_upload_max_retries=" + numRetries);
    afterBuildCommand();
    events.assertContainsError(
        Pattern.compile(
            "The Build Event Protocol upload failed: Not retrying publishBuildEvents.*Boom15"));
    assertThat(buildEventService.getRequestsReceivedCount()).isEqualTo(1);
  }

  @Test
  public void testAfterCommand_waitForUploadComplete_invalidArgumentErrorEarlyInStream()
      throws Exception {
    int numRetries = 3;
    buildEventService.setErrorMessageAndCode("Boom15", Status.INVALID_ARGUMENT);
    buildEventService.setErrorEarlyInStream(true);
    runBuildWithOptions(
        "--bes_backend=inprocess",
        "--bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE",
        "--build_event_upload_max_retries=" + numRetries);
    afterBuildCommand();
    events.assertContainsError(
        Pattern.compile(
            "The Build Event Protocol upload failed: Not retrying publishBuildEvents.*Boom15"));
    assertThat(buildEventService.getRequestsReceivedCount()).isEqualTo(1);
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

    assertThat(getBepTransports()).hasSize(4);
    assertThat(getBepTransports().asList().get(0)).isInstanceOf(TextFormatFileTransport.class);
    assertThat(getBepTransports().asList().get(1)).isInstanceOf(BinaryFormatFileTransport.class);
    assertThat(getBepTransports().asList().get(2)).isInstanceOf(JsonFormatFileTransport.class);
    assertThat(getBepTransports().asList().get(3)).isInstanceOf(BuildEventServiceTransport.class);
  }

  @Test
  public void testUploaderSharing() throws Exception {
    runBuildWithOptions(
        "--build_event_text_file=" + tmpFolder.newFile().getAbsolutePath(),
        "--build_event_binary_file=" + tmpFolder.newFile().getAbsolutePath(),
        "--build_event_json_file=" + tmpFolder.newFile().getAbsolutePath(),
        "--bes_backend=does.not.exist:1234");

    assertThat(getBepTransports()).hasSize(4);

    BuildEventArtifactUploader uploader =
        Iterables.getFirst(getBepTransports(), null).getUploader();
    assertThat(uploader).isNotNull();
    for (BuildEventTransport transport : getBepTransports()) {
      assertThat(uploader).isSameInstanceAs(transport.getUploader());
    }
  }

  @Test
  public void testDoesNotCreatesStreamerWithoutTransports() throws Exception {
    runBuildWithOptions();
    assertThat(getBepTransports()).isEmpty();
  }

  @Test
  public void testKeywords() throws Exception {
    runBuildWithOptions();
    BuildEventServiceOptions besOptions = new BuildEventServiceOptions();
    besOptions.besKeywords = ImmutableList.of("keyword0", "keyword1", "keyword0");
    besOptions.besSystemKeywords = ImmutableList.of("sys_keyword0", "sys_keyword1", "sys_keyword0");

    assertThat(besModule.getBesKeywords("build", besOptions, null))
        .containsExactly(
            "protocol_name=BEP",
            "command_name=build",
            "user_keyword=keyword0",
            "user_keyword=keyword1",
            "sys_keyword0",
            "sys_keyword1");
  }

  @Test
  public void testMakeGrpcMetadata() throws Exception {
    runBuildWithOptions();
    BuildEventServiceOptions besOptions = new BuildEventServiceOptions();
    AuthAndTLSOptions authAndTLSOptions = new AuthAndTLSOptions();
    besOptions.besBackend = "bes-backend";
    besOptions.besProxy = "bes-proxy";
    besOptions.besHeaders =
        ImmutableList.of(
            Map.entry("key1", "val1"),
            Map.entry("key2", "val2"),
            Map.entry("key3", "val3"),
            Map.entry("key1", "val4"));
    BackendConfig newConfig = BackendConfig.create(besOptions, authAndTLSOptions);

    Metadata metadata = BazelBuildEventServiceModule.makeGrpcMetadata(newConfig);
    assertThat(metadata.get(Metadata.Key.of("key1", Metadata.ASCII_STRING_MARSHALLER)))
        .isEqualTo("val4");
    assertThat(metadata.get(Metadata.Key.of("key2", Metadata.ASCII_STRING_MARSHALLER)))
        .isEqualTo("val2");
    assertThat(metadata.get(Metadata.Key.of("key3", Metadata.ASCII_STRING_MARSHALLER)))
        .isEqualTo("val3");
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
      BuildEvent ev;
      while ((ev = BuildEvent.parseDelimitedFrom(in)) != null) {
        buildEvents.add(ev);
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

  @Test
  public void commandLineEvents_includesFlagsFromFlagsets() throws Exception {
    write(
        "hello/BUILD",
        """
        genrule(name = "hello", outs = ["hello.out"], cmd = "touch $@")
        """);

    write(
        "flag/flag_def.bzl",
"""
string_flag = rule(
  implementation = lambda ctx: [],
  build_setting = config.string(flag = True),
)
""");
    write(
        "flag/BUILD",
"""
load(":flag_def.bzl", "string_flag")
string_flag(
  name = "my_flag",
  build_setting_default = "default_value",
)
""");
    writeProjectSclDefinition("test/project_proto.scl", /* alsoWriteBuildFile= */ true);
    write(
        "hello/PROJECT.scl",
"""
load(
  "//test:project_proto.scl",
  "buildable_unit_pb2",
  "project_pb2",
)
project = project_pb2.Project.create(
  enforcement_policy = "warn",
  buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "default_config",
          flags = ["--define=foo=bar", "--//flag:my_flag=my_value"],
          is_default = True,
      )
  ],
)
""");
    File buildEventBinaryFile = tmpFolder.newFile();
    addOptions(
        "--enforce_project_configs",
        "--build_event_binary_file=" + buildEventBinaryFile.getAbsolutePath());
    buildTarget("//hello:hello");

    BuildEvent canonicalCommandLineEvent = null;
    try (InputStream in = new FileInputStream(buildEventBinaryFile)) {
      BuildEvent ev;
      while ((ev = BuildEvent.parseDelimitedFrom(in)) != null) {
        if (ev.hasStructuredCommandLine()
            && ev.getStructuredCommandLine().getCommandLineLabel().equals("canonical")) {
          canonicalCommandLineEvent = ev;
        }
      }
    }
    ImmutableList<CommandLineSection> sections =
        canonicalCommandLineEvent.getStructuredCommandLine().getSectionsList().stream()
            .filter(s -> s.getSectionLabel().equals("command options"))
            .collect(toImmutableList());

    ImmutableList<String> options =
        sections.getFirst().getOptionList().getOptionList().stream()
            .map(Option::getCombinedForm)
            .collect(toImmutableList());
    assertThat(options).contains("--define=foo=bar");
    assertThat(options).contains("--//flag:my_flag=my_value");
  }

  private static final class DelayingCloseBufferedOutputStream extends BufferedOutputStream {
    private final Duration delay;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    DelayingCloseBufferedOutputStream(OutputStream out, Duration delay) {
      super(out);
      this.delay = delay;
      this.out = out;
    }

    @Override
    public void close() throws IOException {
      Uninterruptibles.sleepUninterruptibly(delay);
      super.close();
      closed.set(true);
    }

    public boolean isClosed() {
      return closed.get();
    }
  }
}
