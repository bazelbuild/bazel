// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.collect.MoreCollectors.toOptional;
import static com.google.common.truth.Truth.assertThat;
import static java.util.stream.Collectors.joining;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMultiset;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialModule;
import com.google.devtools.build.lib.buildeventservice.BazelBuildEventServiceModule;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.IdCase;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.TargetSummary;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.TestStatus;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.TestSummary;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.exec.SpawnExecException;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.NoSpawnCacheModule;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Spawn.Code;
import com.google.devtools.build.lib.server.FailureDetails.TestAction;
import com.google.devtools.build.lib.skyframe.rewinding.RewindingTestsHelper;
import com.google.devtools.build.lib.testutil.ActionEventRecorder;
import com.google.devtools.build.lib.testutil.SpawnController.ExecResult;
import com.google.devtools.build.lib.testutil.SpawnInputUtils;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.NoSuchElementException;
import java.util.Optional;
import javax.annotation.Nullable;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration test verifying behavior of {@code
 * com.google.devtools.build.lib.runtime.TargetSummaryEvent} event.
 */
@RunWith(JUnit4.class)
public final class TargetSummaryEventTest extends BuildIntegrationTestCase {

  private static final SpawnResult FAILED_RESULT =
      new SpawnResult.Builder()
          .setStatus(SpawnResult.Status.NON_ZERO_EXIT)
          .setExitCode(1)
          .setFailureDetail(
              FailureDetail.newBuilder()
                  .setSpawn(FailureDetails.Spawn.newBuilder().setCode(Code.NON_ZERO_EXIT))
                  .build())
          .setRunnerName("remote")
          .build();

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  @Rule public final TemporaryFolder tmpFolder = new TemporaryFolder();

  private final ActionEventRecorder actionEventRecorder = new ActionEventRecorder();
  private final RewindingTestsHelper helper = new RewindingTestsHelper(this, actionEventRecorder);

  @Before
  public void stageEmbeddedTools() throws Exception {
    AnalysisMock.get().setupMockToolsRepository(mockToolsConfig);
  }

  @After
  public void verifyAllSpawnShimsConsumed() {
    helper.verifyAllSpawnShimsConsumed();
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(new NoSpawnCacheModule())
        .addBlazeModule(new CredentialModule())
        .addBlazeModule(new BazelBuildEventServiceModule())
        .addBlazeModule(helper.makeControllableActionStrategyModule("standalone"));
  }

  @Override
  protected void setupOptions() throws Exception {
    super.setupOptions();
    addOptions("--spawn_strategy=standalone", "--test_strategy=standalone");
    runtimeWrapper.registerSubscriber(actionEventRecorder);
  }

  private void afterBuildCommand() throws Exception {
    runtimeWrapper.newCommand();
  }

  @Test
  public void plainTarget_buildSuccess() throws Exception {
    write("foo/BUILD", "genrule(name = 'foobin', outs = ['out.txt'], cmd = 'echo -n Hello > $@')");

    File bep = buildTargetAndCaptureBuildEventProtocol("//foo:foobin");

    TargetSummary summary = findTargetSummaryEventInBuildEventStream(bep);
    assertThat(summary.getOverallBuildSuccess()).isTrue();
    assertThat(summary.getOverallTestStatus()).isEqualTo(TestStatus.NO_STATUS);
  }

  @Test
  public void plainTarget_buildFails() throws Exception {
    write("foo/BUILD", "genrule(name = 'foobin', outs = ['out.txt'], cmd = 'false')");

    File bep = buildFailingTargetAndCaptureBuildEventProtocol("//foo:foobin");

    TargetSummary summary = findTargetSummaryEventInBuildEventStream(bep);
    assertThat(summary.getOverallBuildSuccess()).isFalse();
    assertThat(summary.getOverallTestStatus()).isEqualTo(TestStatus.NO_STATUS);
  }

  @Test
  public void test_buildSucceeds_testSucceeds() throws Exception {
    write("foo/good_test.sh", "#!/bin/bash", "true").setExecutable(true);
    write(
        "foo/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name = 'good_test', srcs = ['good_test.sh'])");

    File bep = testTargetAndCaptureBuildEventProtocol("//foo:good_test");

    TargetSummary targetSummary = findTargetSummaryEventInBuildEventStream(bep);
    assertThat(targetSummary.getOverallBuildSuccess()).isTrue();
    assertThat(targetSummary.getOverallTestStatus()).isEqualTo(TestStatus.PASSED);

    TestSummary testSummary = findTestSummaryEventInBuildEventStream(bep);
    assertThat(testSummary.getOverallStatus()).isEqualTo(TestStatus.PASSED);
  }

  @Test
  public void test_buildSucceeds_testFails() throws Exception {
    write("foo/bad_test.sh", "#!/bin/bash", "false").setExecutable(true);
    write(
        "foo/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name = 'bad_test', srcs = ['bad_test.sh'])");

    File bep = testTargetAndCaptureBuildEventProtocol("//foo:bad_test");

    TargetSummary targetSummary = findTargetSummaryEventInBuildEventStream(bep);
    assertThat(targetSummary.getOverallBuildSuccess()).isTrue();
    assertThat(targetSummary.getOverallTestStatus()).isEqualTo(TestStatus.FAILED);

    TestSummary testSummary = findTestSummaryEventInBuildEventStream(bep);
    assertThat(testSummary.getOverallStatus()).isEqualTo(TestStatus.FAILED);
  }

  @Test
  public void test_buildSucceeds_testRuntimeFailsToBuild() throws Exception {
    write("foo/good_test.sh", "#!/bin/bash", "true").setExecutable(true);
    write(
        "foo/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name = 'good_test', srcs = ['good_test.sh'])");

    // Hack: the path to the tools/test/BUILD file is prefixed in the Bazel tests.
    String pathToToolsTestBuildPrefix = AnalysisMock.get().isThisBazel() ? "embedded_tools/" : "";
    Path toolsTestBuildPath =
        mockToolsConfig.getPath(pathToToolsTestBuildPrefix + "tools/test/BUILD");
    // Delete the test-setup.sh file and introduce a broken genrule to create test-setup.sh.
    mockToolsConfig.getPath(pathToToolsTestBuildPrefix + "tools/test/test-setup.sh").delete();
    String bogusTestSetupGenrule =
        """
        genrule(
            name = 'bogus-make-test-setup',
            outs = ['test-setup.sh'],
            cmd = 'false',
        )
        """;
    FileSystemUtils.appendIsoLatin1(toolsTestBuildPath, bogusTestSetupGenrule);

    File bep = testTargetAndCaptureBuildEventProtocol("//foo:good_test");

    TargetSummary targetSummary = findTargetSummaryEventInBuildEventStream(bep);
    assertThat(targetSummary.getOverallBuildSuccess()).isTrue();
    assertThat(targetSummary.getOverallTestStatus()).isEqualTo(TestStatus.FAILED_TO_BUILD);

    // TODO: b/186996003 - TestSummary is a child of TargetComplete and should be posted.
    TestSummary testSummary = findTestSummaryEventInBuildEventStream(bep);
    assertThat(testSummary).isNull();
  }

  @Test
  public void test_testActionThrowsExecException() throws Exception {
    addOptions("--rewind_lost_inputs");
    write(
        "foo/BUILD",
        """
        load("//test_defs:foo_test.bzl", "foo_test")
        foo_test(name = "test", srcs = ["test.sh"], tags = ["cpu:invalid"])
        """);
    write("foo/test.sh", "#!/bin/bash", "true").setExecutable(true);
    helper.addSpawnShim(
        "Testing //foo:test",
        (spawn, context) ->
            ExecResult.ofException(
                new UserExecException(
                    FailureDetail.newBuilder()
                        .setMessage("Invalid cpu tag: 'cpu:invalid'")
                        .setTestAction(
                            TestAction.newBuilder().setCode(TestAction.Code.INVALID_CPU_TAG))
                        .build())));

    File bep = testTargetAndCaptureBuildEventProtocol("//foo:test");

    TargetSummary targetSummary = findTargetSummaryEventInBuildEventStream(bep);
    assertThat(targetSummary.getOverallBuildSuccess()).isTrue();
    assertThat(targetSummary.getOverallTestStatus()).isEqualTo(TestStatus.FAILED_TO_BUILD);

    // TODO: b/186996003 - TestSummary is a child of TargetComplete and should be posted.
    TestSummary testSummary = findTestSummaryEventInBuildEventStream(bep);
    assertThat(testSummary).isNull();
  }

  @Test
  public void test_testActionLosesInput_rewindingSucceeds() throws Exception {
    addOptions("--rewind_lost_inputs");
    write(
        "foo/BUILD",
        """
        load("//test_defs:foo_test.bzl", "foo_test")
        foo_test(name = "test", srcs = ["test.sh"], data = [":lost"])
        genrule(name = "lost", outs = ["lost.out"], cmd = "echo lost > $@")
        """);
    write("foo/test.sh", "#!/bin/bash", "true").setExecutable(true);
    helper.addSpawnShim(
        "Testing //foo:test",
        (spawn, context) -> {
          Artifact lost = SpawnInputUtils.getRunfilesArtifactWithName(spawn, context, "lost.out");
          return helper.createLostInputsExecException(context, lost);
        });

    File bep = testTargetAndCaptureBuildEventProtocol("//foo:test");

    TargetSummary targetSummary = findTargetSummaryEventInBuildEventStream(bep);
    assertThat(targetSummary.getOverallBuildSuccess()).isTrue();
    assertThat(targetSummary.getOverallTestStatus()).isEqualTo(TestStatus.PASSED);

    TestSummary testSummary = findTestSummaryEventInBuildEventStream(bep);
    assertThat(testSummary.getOverallStatus()).isEqualTo(TestStatus.PASSED);

    assertThat(ImmutableMultiset.copyOf(helper.getExecutedSpawnDescriptions()))
        .hasCount("Executing genrule //foo:lost", 2);
  }

  @Test
  public void test_testActionLosesInput_flakyActionFailsAfterRewind() throws Exception {
    addOptions("--rewind_lost_inputs");
    write(
        "foo/BUILD",
        """
        load("//test_defs:foo_test.bzl", "foo_test")
        foo_test(name = "test", srcs = ["test.sh"], data = [":flaky_lost"])
        genrule(name = "flaky_lost", outs = ["flaky_lost.out"], cmd = "echo flaky_lost > $@")
        """);
    write("foo/test.sh", "#!/bin/bash", "true").setExecutable(true);
    helper.addSpawnShim(
        "Testing //foo:test",
        (spawn, context) -> {
          helper.addSpawnShim(
              "Executing genrule //foo:flaky_lost",
              (spawn2, context2) ->
                  ExecResult.ofException(
                      new SpawnExecException(
                          "Flaky action failure",
                          FAILED_RESULT,
                          /* forciblyRunRemotely= */ false,
                          /* catastrophe= */ false)));
          Artifact flakyLost =
              SpawnInputUtils.getRunfilesArtifactWithName(spawn, context, "flaky_lost.out");
          return helper.createLostInputsExecException(context, flakyLost);
        });

    File bep = testTargetAndCaptureBuildEventProtocol("//foo:test");

    TargetSummary targetSummary = findTargetSummaryEventInBuildEventStream(bep);
    assertThat(targetSummary.getOverallBuildSuccess()).isTrue();
    assertThat(targetSummary.getOverallTestStatus()).isEqualTo(TestStatus.FAILED_TO_BUILD);

    // TODO: b/186996003 - TestSummary is a child of TargetComplete and should be posted.
    TestSummary testSummary = findTestSummaryEventInBuildEventStream(bep);
    assertThat(testSummary).isNull();

    assertThat(ImmutableMultiset.copyOf(helper.getExecutedSpawnDescriptions()))
        .hasCount("Executing genrule //foo:flaky_lost", 2);
  }

  private File buildTargetAndCaptureBuildEventProtocol(String target) throws Exception {
    File bep = tmpFolder.newFile();
    // We use WAIT_FOR_UPLOAD_COMPLETE because it's the easiest way to force the BES module to
    // wait until the BEP binary file has been written.
    addOptions(
        "--keep_going",
        "--experimental_bep_target_summary",
        "--build_event_binary_file=" + bep.getAbsolutePath(),
        "--bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE");
    buildTarget(target);
    // We need to wait for all events to be written to the file, which is done in #afterCommand()
    // if --bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE.
    afterBuildCommand();
    return bep;
  }

  private File buildFailingTargetAndCaptureBuildEventProtocol(String target) throws Exception {
    File bep = tmpFolder.newFile();
    // We use WAIT_FOR_UPLOAD_COMPLETE because it's the easiest way to force the BES module to
    // wait until the BEP binary file has been written.
    addOptions(
        "--keep_going",
        "--experimental_bep_target_summary",
        "--build_event_binary_file=" + bep.getAbsolutePath(),
        "--bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE");
    assertThrows(BuildFailedException.class, () -> buildTarget(target));
    // We need to wait for all events to be written to the file, which is done in #afterCommand()
    // if --bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE.
    afterBuildCommand();
    return bep;
  }

  private File testTargetAndCaptureBuildEventProtocol(String target) throws Exception {
    File bep = tmpFolder.newFile();
    BlazeCommandDispatcher dispatcher = new BlazeCommandDispatcher(getRuntime());
    ImmutableList.Builder<String> args = ImmutableList.builder();
    args.add("test", target);
    args.addAll(runtimeWrapper.getOptions());
    // We use WAIT_FOR_UPLOAD_COMPLETE because it's the easiest way to force the BES module to
    // wait until the BEP binary file has been written.
    args.add(
        "--default_visibility=public",
        "--test_output=all",
        "--keep_going",
        "--client_env=PATH=/bin:/usr/bin:/usr/sbin:/sbin",
        "--experimental_bep_target_summary",
        "--build_event_binary_file=" + bep.getAbsolutePath(),
        "--bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE");
    dispatcher.exec(args.build(), /* clientDescription= */ "test", outErr);
    return bep;
  }

  private static ImmutableList<BuildEvent> parseBuildEventsFromBuildEventStream(File bep)
      throws IOException {
    ImmutableList.Builder<BuildEvent> buildEvents = ImmutableList.builder();
    try (InputStream in = new FileInputStream(bep)) {
      BuildEvent ev;
      while ((ev = BuildEvent.parseDelimitedFrom(in)) != null) {
        buildEvents.add(ev);
      }
    }
    return buildEvents.build();
  }

  private static TargetSummary findTargetSummaryEventInBuildEventStream(File bep)
      throws IOException {
    ImmutableList<BuildEvent> events = parseBuildEventsFromBuildEventStream(bep);
    Optional<TargetSummary> targetSummary =
        events.stream()
            .filter(e -> e.getId().getIdCase() == IdCase.TARGET_SUMMARY)
            .map(BuildEvent::getTargetSummary)
            .collect(toOptional());
    if (targetSummary.isEmpty()) {
      logger.atSevere().log(
          "No TargetSummary event found, dumping BEP:\n%s",
          events.stream().map(BuildEvent::toString).collect(joining("\n")));
      throw new NoSuchElementException("No TargetSummary event found, see test log for full BEP");
    }
    return targetSummary.get();
  }

  @Nullable
  private static TestSummary findTestSummaryEventInBuildEventStream(File bep) throws IOException {
    return parseBuildEventsFromBuildEventStream(bep).stream()
        .filter(e -> e.getId().getIdCase() == IdCase.TEST_SUMMARY)
        .map(BuildEvent::getTestSummary)
        .collect(toOptional())
        .orElse(null);
  }
}
