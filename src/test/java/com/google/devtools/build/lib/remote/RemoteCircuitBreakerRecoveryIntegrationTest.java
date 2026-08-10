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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.readContent;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialModule;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.remote.options.RemoteStartupOptions;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils.WorkerInstance;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlockWaitingModule;
import com.google.devtools.build.lib.runtime.BuildSummaryStatsModule;
import com.google.devtools.build.lib.standalone.StandaloneModule;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsBase;
import org.junit.After;
import org.junit.ClassRule;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration tests for the remote failure circuit breaker's opt-in half-open recovery, run under
 * {@code --remote_download_minimal} so that tripping the breaker happens while prefetching a
 * remote-only input.
 *
 * <p>Kept separate from {@link BuildWithoutTheBytesIntegrationTest} because these tests need a
 * dedicated remote worker that injects transient read failures, whereas every other
 * Build-without-the-Bytes test runs against a pristine worker.
 */
@RunWith(JUnit4.class)
public class RemoteCircuitBreakerRecoveryIntegrationTest extends BuildIntegrationTestCase {

  // A worker that returns UNAVAILABLE for the first armed ByteStream.Read, so a build can trip the
  // remote failure circuit breaker on a transient, self-healing failure.
  @ClassRule @Rule
  public static final WorkerInstance worker =
      IntegrationTestUtils.createFailFirstReadWorker(/* n= */ 1);

  // The worker logs this to stderr each time it injects a read failure (see FailFirstNInterceptor
  // in RemoteWorker). The tests match on it to prove a transient failure was actually delivered;
  // keep it in sync with the worker's message.
  private static final String INJECTED_READ_FAILURE_LOG =
      "INJECTED_UNAVAILABLE google.bytestream.ByteStream/Read";

  @Override
  protected ImmutableList<Class<? extends OptionsBase>> getStartupOptionClasses() {
    return ImmutableList.<Class<? extends OptionsBase>>builder()
        .addAll(super.getStartupOptionClasses())
        .add(RemoteStartupOptions.class)
        .build();
  }

  @Override
  protected void setupOptions() throws Exception {
    super.setupOptions();
    addOptions(
        "--remote_executor=grpc://localhost:" + worker.getPort(),
        "--remote_download_minimal",
        // Both tests recover a lost input via action rewinding: a failed prefetch (with
        // --remote_retries=0) is treated as a lost input, and rewinding re-runs the action.
        "--rewind_lost_inputs",
        // Disable build-level invocation retries (default 5) so recovery is exercised via action
        // rewinding within one build, not by retrying the whole build with a fresh breaker.
        "--experimental_remote_cache_eviction_retries=0");
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(new RemoteModule())
        .addBlazeModule(new BuildSummaryStatsModule())
        .addBlazeModule(new BlockWaitingModule());
  }

  @Override
  protected ImmutableList<BlazeModule> getSpawnModules() {
    return ImmutableList.<BlazeModule>builder()
        .addAll(super.getSpawnModules())
        .add(new StandaloneModule())
        .add(new CredentialModule())
        .build();
  }

  @After
  public void waitDownloads() throws Exception {
    runtimeWrapper.newCommand();
  }

  private Path getOutputPath(String binRelativePath) {
    return getTargetConfiguration().getBinDir().getRoot().getRelative(binRelativePath);
  }

  private void assertValidOutputFile(String binRelativePath, String content) throws Exception {
    Path output = getOutputPath(binRelativePath);
    assertThat(readContent(output, UTF_8)).isEqualTo(content);
    assertThat(output.isReadable()).isTrue();
    assertThat(output.isWritable()).isFalse();
    assertThat(output.isExecutable()).isTrue();
  }

  private void writeFooBar() throws Exception {
    write(
        "a/BUILD",
        """
        genrule(
            name = "foo",
            srcs = ["foo.in"],
            outs = ["foo.out"],
            cmd = "cat $(SRCS) > $@",
        )

        genrule(
            name = "bar",
            srcs = [
                "foo.out",
                "bar.in",
            ],
            outs = ["bar.out"],
            cmd = "cat $(SRCS) > $@",
            tags = ["no-remote-exec"],
        )
        """);
    write("a/foo.in", "foo");
    write("a/bar.in", "bar");
  }

  /**
   * Populates the remote cache and leaves foo.out remote-only, so a later build of //a:bar must
   * prefetch it. The failing worker is wired up in {@link #setupOptions}, which {@link
   * #createRuntimeWrapper} re-runs, so foo.out stays remote-only across the simulated server
   * restart.
   */
  private void setupRemoteOnlyFooOut() throws Exception {
    writeFooBar();

    buildTarget("//a:bar");
    getOutputPath("a/foo.out").delete();
    getOutputPath("a/bar.out").delete();
    getOutputBase().getRelative("action_cache").deleteTreesBelow();
    createRuntimeWrapper(); // Simulate a server restart.

    buildTarget("//a:bar");
    assertThat(getOutputPath("a/foo.out").exists()).isFalse();
  }

  /**
   * Asserts the worker delivered at least one injected read failure since {@code
   * previousStderrLength} (captured just before the build) — i.e. the breaker actually saw a
   * transient remote failure, so a green build is meaningful rather than vacuous.
   */
  private static void assertReadFailureInjectedSince(int previousStderrLength) {
    assertThat(worker.getStderr().substring(previousStderrLength))
        .contains(INJECTED_READ_FAILURE_LOG);
  }

  @Test
  public void recovers_whenPrefetchingInput_succeedsWithActionRewinding() throws Exception {
    setupRemoteOnlyFooOut();

    // Act: enable the failure breaker with a short recovery delay, arm a single transient read
    // failure, and force bar to re-prefetch the remote-only foo.out. The prefetch read trips the
    // breaker; with --remote_retries=0 that becomes a lost input and drives action rewinding, and
    // the breaker re-closes on a trial probe during the rewind so the build recovers.
    addOptions(
        "--experimental_circuit_breaker_strategy=failure",
        "--experimental_remote_min_fail_count_to_compute_failure_rate=1",
        "--experimental_remote_min_call_count_to_compute_failure_rate=1",
        "--experimental_remote_failure_rate_threshold=1",
        "--experimental_remote_circuit_breaker_recovery_delay=1ms",
        "--remote_retries=0");
    write("a/bar.in", "updated bar");
    int stderrLenBefore = worker.getStderr().length();
    worker.armReadFailures();

    buildTarget("//a:bar");

    // Assert: a transient failure was actually delivered (the breaker tripped) and, thanks to
    // recovery, the build still succeeded with the correct output.
    assertReadFailureInjectedSince(stderrLenBefore);
    assertValidOutputFile("a/bar.out", "foo\nupdated bar\n");
  }

  @Test
  public void trips_whenPrefetchingInput_recoveryDisabled_fails() throws Exception {
    setupRemoteOnlyFooOut();

    // Act: same as the recovery test, but recovery is disabled (the delay defaults to 0), so the
    // tripped breaker stays open for the rest of the build and the rewound re-execution is
    // rejected.
    addOptions(
        "--experimental_circuit_breaker_strategy=failure",
        "--experimental_remote_min_fail_count_to_compute_failure_rate=1",
        "--experimental_remote_min_call_count_to_compute_failure_rate=1",
        "--experimental_remote_failure_rate_threshold=1",
        "--remote_retries=0");
    write("a/bar.in", "updated bar");
    int stderrLenBefore = worker.getStderr().length();
    worker.armReadFailures();

    var error = assertThrows(BuildFailedException.class, () -> buildTarget("//a:bar"));

    // Assert: the breaker tripped and, without recovery, the build failed with a remote error.
    assertReadFailureInjectedSince(stderrLenBefore);
    assertThat(error.getDetailedExitCode().getExitCode().getNumericExitCode()).isEqualTo(34);
  }
}
