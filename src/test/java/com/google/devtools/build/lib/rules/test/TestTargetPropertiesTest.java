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

package com.google.devtools.build.lib.rules.test;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.analysis.test.TestTargetProperties;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TestTargetProperties}. */
@RunWith(JUnit4.class)
public class TestTargetPropertiesTest extends BuildViewTestCase {
  @Test
  public void testTestWithCpusTagHasCorrectLocalResourcesEstimate() throws Exception {
    scratch.file("tests/test.sh", "#!/bin/bash", "exit 0");
    scratch.file(
        "tests/BUILD",
        """
        sh_test(
            name = "test",
            size = "small",
            srcs = ["test.sh"],
            tags = ["cpu:4"],
        )
        """);
    ConfiguredTarget testTarget = getConfiguredTarget("//tests:test");
    TestRunnerAction testAction =
        (TestRunnerAction)
            getGeneratingAction(TestProvider.getTestStatusArtifacts(testTarget).get(0));
    ResourceSet localResourceUsage =
        testAction
            .getTestProperties()
            .getLocalResourceUsage(testAction.getOwner().getLabel(), false);
    assertThat(localResourceUsage.getCpuUsage()).isEqualTo(4.0);
  }

  @Test
  public void testTestResourcesFlag() throws Exception {
    scratch.file("tests/test.sh", "#!/bin/bash", "exit 0");
    scratch.file(
        "tests/BUILD",
        """
        sh_test(
            name = "test",
            size = "medium",
            srcs = ["test.sh"],
            tags = ["resources:gpu:4"],
        )
        """);
    useConfiguration(
        "--default_test_resources=memory=10,20,30,40",
        "--default_test_resources=cpu=1,2,3,4",
        "--default_test_resources=gpu=1",
        "--default_test_resources=cpu=5");
    ConfiguredTarget testTarget = getConfiguredTarget("//tests:test");
    TestRunnerAction testAction =
        (TestRunnerAction)
            getGeneratingAction(TestProvider.getTestStatusArtifacts(testTarget).get(0));
    ResourceSet localResourceUsage =
        testAction
            .getTestProperties()
            .getLocalResourceUsage(testAction.getOwner().getLabel(), false);
    // Tags-specified resources overrides --default_test_resources=gpu.
    assertThat(localResourceUsage.getResources()).containsEntry("gpu", 4.0);
    // The last-specified value of --default_test_resources=cpu is used.
    assertThat(localResourceUsage.getCpuUsage()).isEqualTo(5.0);
    assertThat(localResourceUsage.getMemoryMb()).isEqualTo(20);
  }

  @Test
  public void testTestWithExclusiveRunLocallyByDefault() throws Exception {
    useConfiguration("--noincompatible_exclusive_test_sandboxed");
    scratch.file("tests/test.sh", "#!/bin/bash", "exit 0");
    scratch.file(
        "tests/BUILD",
        """
        sh_test(
            name = "test",
            size = "small",
            srcs = ["test.sh"],
            tags = ["exclusive"],
        )
        """);
    ConfiguredTarget testTarget = getConfiguredTarget("//tests:test");
    TestRunnerAction testAction =
        (TestRunnerAction)
            getGeneratingAction(TestProvider.getTestStatusArtifacts(testTarget).get(0));
    assertThat(testAction.getExecutionInfo()).containsKey(ExecutionRequirements.LOCAL);
    assertThat(testAction.getExecutionInfo()).hasSize(1);
  }

  @Test
  public void testTestWithExclusiveIfRunLocally_notTaggedLocal() throws Exception {
    useConfiguration("--noincompatible_exclusive_test_sandboxed");
    scratch.file("tests/test.sh", "#!/bin/bash", "exit 0");
    scratch.file(
        "tests/BUILD",
        """
        sh_test(
            name = "test",
            size = "small",
            srcs = ["test.sh"],
            tags = ["exclusive-if-local"],
        )
        """);
    ConfiguredTarget testTarget = getConfiguredTarget("//tests:test");
    TestRunnerAction testAction =
        (TestRunnerAction)
            getGeneratingAction(TestProvider.getTestStatusArtifacts(testTarget).get(0));
    // "exclusive" tests become local when TestTargetProperties adds local executionInfo.
    // Ensure this is not the case for "exclusive-if-local"
    assertThat(testAction.getExecutionInfo()).isEmpty();
  }

  @Test
  public void testTestWithExclusiveDisablesRemoteExecution() throws Exception {
    scratch.file("tests/test.sh", "#!/bin/bash", "exit 0");
    scratch.file(
        "tests/BUILD",
        """
        sh_test(
            name = "test",
            size = "small",
            srcs = ["test.sh"],
            tags = ["exclusive"],
        )
        """);
    ConfiguredTarget testTarget = getConfiguredTarget("//tests:test");
    TestRunnerAction testAction =
        (TestRunnerAction)
            getGeneratingAction(TestProvider.getTestStatusArtifacts(testTarget).get(0));
    assertThat(testAction.getExecutionInfo()).containsKey(ExecutionRequirements.NO_REMOTE_EXEC);
    assertThat(testAction.getExecutionInfo()).hasSize(1);
  }

  @Test
  public void testTestWithExclusiveIfRunLocally_notTaggedNoRemote() throws Exception {
    scratch.file("tests/test.sh", "#!/bin/bash", "exit 0");
    scratch.file(
        "tests/BUILD",
        """
        sh_test(
            name = "test",
            size = "small",
            srcs = ["test.sh"],
            tags = ["exclusive-if-local"],
        )
        """);
    ConfiguredTarget testTarget = getConfiguredTarget("//tests:test");
    TestRunnerAction testAction =
        (TestRunnerAction)
            getGeneratingAction(TestProvider.getTestStatusArtifacts(testTarget).get(0));
    // "exclusive" tests become local when TestTargetProperties adds a no-remote-exec requirement
    // to the execution info. Ensure this is not the case for "exclusive-if-local"
    assertThat(testAction.getExecutionInfo()).isEmpty();
  }

  @Test
  public void testTestWithExclusiveAndLocalRunLocally() throws Exception {
    useConfiguration("--incompatible_exclusive_test_sandboxed");
    scratch.file("tests/test.sh", "#!/bin/bash", "exit 0");
    scratch.file(
        "tests/BUILD",
        """
        sh_test(
            name = "test",
            size = "small",
            srcs = ["test.sh"],
            tags = [
                "exclusive",
                "local",
            ],
        )
        """);
    ConfiguredTarget testTarget = getConfiguredTarget("//tests:test");
    TestRunnerAction testAction =
        (TestRunnerAction)
            getGeneratingAction(TestProvider.getTestStatusArtifacts(testTarget).get(0));
    assertThat(testAction.getExecutionInfo()).containsKey(ExecutionRequirements.LOCAL);
    assertThat(testAction.getExecutionInfo()).hasSize(1);
  }
}
