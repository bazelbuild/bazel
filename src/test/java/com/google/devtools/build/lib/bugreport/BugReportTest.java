// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bugreport;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.bugreport.BugReport.BlazeRuntimeInterface;
import com.google.devtools.build.lib.server.FailureDetails.Crash;
import com.google.devtools.build.lib.server.FailureDetails.Crash.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Throwable;
import com.google.devtools.build.lib.util.CustomExitCodePublisher;
import com.google.devtools.build.lib.util.CustomFailureDetailPublisher;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.protobuf.ExtensionRegistry;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;

/** Tests for {@link BugReport}. */
@RunWith(JUnit4.class)
public class BugReportTest {

  private static final String TEST_EXCEPTION_NAME =
      "com.google.devtools.build.lib.bugreport.BugReportTest$TestException";
  @Rule public final TemporaryFolder temporaryFolder = new TemporaryFolder();

  @After
  public void resetPublishers() {
    CustomExitCodePublisher.resetAbruptExitStatusFile();
    CustomFailureDetailPublisher.resetFailureDetailFilePath();
  }

  @Test
  public void handleCrash() throws Exception {
    BlazeRuntimeInterface mockRuntime = mock(BlazeRuntimeInterface.class);
    when(mockRuntime.getProductName()).thenReturn("myProductName");
    BugReport.setRuntime(mockRuntime);

    File folderForExitStatusFile = temporaryFolder.newFolder();
    CustomExitCodePublisher.setAbruptExitStatusFileDir(folderForExitStatusFile.getPath());

    Path failureDetailFilePath =
        folderForExitStatusFile.toPath().resolve("failure_detail.rawproto");
    CustomFailureDetailPublisher.setFailureDetailFilePath(failureDetailFilePath.toString());

    assertThrows(
        SecurityException.class,
        () ->
            BugReport.handleCrashWithoutSendingBugReport(
                functionForStackFrameTest(), ExitCode.RESERVED));

    Path exitStatusFile =
        folderForExitStatusFile.toPath().resolve("exit_code_to_use_on_abrupt_exit");
    List<String> strings = Files.readAllLines(exitStatusFile, StandardCharsets.UTF_8);
    assertThat(strings).containsExactly("40");

    FailureDetail failureDetail =
        FailureDetail.parseFrom(
            Files.readAllBytes(failureDetailFilePath), ExtensionRegistry.getEmptyRegistry());
    assertThat(failureDetail.getMessage())
        .isEqualTo(String.format("Crashed: (%s) myMessage", TEST_EXCEPTION_NAME));
    assertThat(failureDetail.hasCrash()).isTrue();
    Crash crash = failureDetail.getCrash();
    assertThat(crash.getCode()).isEqualTo(Code.CRASH_UNKNOWN);
    assertThat(crash.getCausesList()).hasSize(1);
    Throwable cause = crash.getCauses(0);
    assertThat(cause.getMessage()).isEqualTo("myMessage");
    assertThat(cause.getThrowableClass()).isEqualTo(TEST_EXCEPTION_NAME);
    assertThat(cause.getStackTraceCount()).isAtLeast(1);
    assertThat(cause.getStackTrace(0))
        .contains(
            "com.google.devtools.build.lib.bugreport.BugReportTest.functionForStackFrameTest");
    ArgumentCaptor<DetailedExitCode> exitCodeCaptor =
        ArgumentCaptor.forClass(DetailedExitCode.class);
    verify(mockRuntime).cleanUpForCrash(exitCodeCaptor.capture());
    DetailedExitCode exitCode = exitCodeCaptor.getValue();
    assertThat(exitCode.getExitCode()).isEqualTo(ExitCode.RESERVED);
    assertThat(exitCode.getFailureDetail()).isEqualTo(failureDetail);
  }

  private TestException functionForStackFrameTest() {
    return new TestException("myMessage");
  }

  private static class TestException extends Exception {
    private TestException(String message) {
      super(message);
    }
  }
}
