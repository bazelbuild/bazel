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
package com.google.devtools.build.lib.buildeventservice;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;

import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient.AckCallback;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient.CommandContext;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient.LifecycleEvent;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient.StreamContext;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient.StreamStatus;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.LocalFilesArtifactUploader;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.common.options.Options;
import java.net.UnknownHostException;
import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * End-to-end test that a client-side transport cause (e.g. an {@link UnknownHostException}) attached
 * to a failing {@link StreamStatus} is surfaced both in the BES upload error message and in the
 * resulting exception's cause chain, instead of being collapsed into the bare gRPC status.
 */
@RunWith(JUnit4.class)
public class BuildEventServiceUploaderCauseTest extends FoundationTestCase {

  private static final CommandContext COMMAND_CONTEXT =
      CommandContext.builder()
          .setBuildId("feedbeef-dead-4321-beef-deaddeaddead")
          .setInvocationId("feedbeef-dead-4444-beef-deaddeaddead")
          .setAttemptNumber(1)
          .setKeywords(ImmutableSet.of("foo=bar"))
          .setProjectId(null)
          .setCheckPrecedingLifecycleEvents(false)
          .build();

  @Test
  public void uploadFailure_surfacesTransportCauseInMessageAndCauseChain() {
    // A transport failure as it appears client-side: an UNAVAILABLE status whose cause is the
    // concrete network error. Wrap it so getRootCause() unwrapping is also exercised.
    UnknownHostException rootCause = new UnknownHostException("sambanova-systems.buildbuddy.io");
    StreamStatus failingStatus =
        new FakeStreamStatus("UNAVAILABLE: ipv4:1.2.3.4:443", new RuntimeException(rootCause));

    BuildEventServiceTransport transport = newTransport(new FailingBesClient(failingStatus));

    ExecutionException exception =
        assertThrows(ExecutionException.class, () -> transport.close().get());

    // The user-facing message (which reaches the console / Jenkins log via the DetailedExitCode)
    // names the root transport cause, not just the bare gRPC status.
    assertThat(exception.getMessage()).contains("UNAVAILABLE");
    assertThat(exception.getMessage())
        .contains("(cause: java.net.UnknownHostException: sambanova-systems.buildbuddy.io)");

    // The cause chain is preserved end-to-end so logger.atSevere().withCause(e) records the full
    // transport stack trace.
    assertThat(exception).hasCauseThat().isInstanceOf(AbruptExitException.class);
    assertThat(Throwables.getCausalChain(exception)).contains(rootCause);
  }

  @Test
  public void uploadFailure_withoutCause_omitsCauseSuffix() {
    StreamStatus failingStatus = new FakeStreamStatus("UNAVAILABLE: server shutting down", null);

    BuildEventServiceTransport transport = newTransport(new FailingBesClient(failingStatus));

    ExecutionException exception =
        assertThrows(ExecutionException.class, () -> transport.close().get());

    assertThat(exception.getMessage()).contains("UNAVAILABLE: server shutting down");
    assertThat(exception.getMessage()).doesNotContain("(cause:");
  }

  private BuildEventServiceTransport newTransport(BuildEventServiceClient client) {
    BuildEventServiceOptions besOptions = Options.getDefaults(BuildEventServiceOptions.class);
    besOptions.setBesTimeout(Duration.ZERO);
    besOptions.setBesLifecycleEvents(true);

    return new BuildEventServiceTransport.Builder()
        .besOptions(besOptions)
        // No real sleeping between retries, to keep the test fast.
        .sleeper(sleepMillis -> {})
        .eventBus(eventBus)
        .besClient(client)
        .artifactGroupNamer(mock(ArtifactGroupNamer.class))
        .localFileUploader(new LocalFilesArtifactUploader())
        .bepOptions(Options.getDefaults(BuildEventProtocolOptions.class))
        .clock(new JavaClock())
        .commandContext(COMMAND_CONTEXT)
        .commandStartTime(Instant.ofEpochMilli(500L))
        .build();
  }

  /** A {@link BuildEventServiceClient} whose every lifecycle publish fails with a fixed status. */
  private static final class FailingBesClient implements BuildEventServiceClient {
    private final StreamStatus failingStatus;

    FailingBesClient(StreamStatus failingStatus) {
      this.failingStatus = failingStatus;
    }

    @Override
    public void publish(CommandContext commandContext, LifecycleEvent lifecycleEvent)
        throws StreamException {
      throw new StreamException(failingStatus, failingStatus.getCause());
    }

    @Override
    public StreamContext openStream(CommandContext commandContext, AckCallback callback) {
      throw new UnsupportedOperationException("stream should not be opened after lifecycle failure");
    }

    @Override
    public void shutdown() {}
  }

  /** A {@link StreamStatus} that optionally carries a client-side transport cause. */
  private static final class FakeStreamStatus implements StreamStatus {
    private final String errorMessage;
    @Nullable private final Throwable cause;

    FakeStreamStatus(String errorMessage, @Nullable Throwable cause) {
      this.errorMessage = errorMessage;
      this.cause = cause;
    }

    @Override
    public boolean isOk() {
      return false;
    }

    @Override
    public boolean isRetriable() {
      return true;
    }

    @Override
    public boolean isFailedPrecondition() {
      return false;
    }

    @Override
    public String getErrorMessage() {
      return errorMessage;
    }

    @Override
    @Nullable
    public Throwable getCause() {
      return cause;
    }
  }
}
