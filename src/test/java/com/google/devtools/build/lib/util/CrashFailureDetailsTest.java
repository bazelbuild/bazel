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

package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.Crash;
import com.google.devtools.build.lib.server.FailureDetails.Crash.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.protobuf.ProtocolStringList;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.List;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link CrashFailureDetails}. */
@RunWith(TestParameterInjector.class)
public final class CrashFailureDetailsTest {

  private static final String TEST_EXCEPTION_NAME =
      "com.google.devtools.build.lib.util.CrashFailureDetailsTest$TestException";

  @After
  public void restoreDefaultOomDetector() {
    CrashFailureDetails.setOomDetector(() -> false);
  }

  @Test
  public void nestedThrowables() {
    // This test confirms that throwables' details are recorded: their messages, types, stack
    // frames, and causes. The outermost throwable is recorded at index 0.

    FailureDetail failureDetail =
        CrashFailureDetails.forThrowable(
            functionForStackFrameTests_A(functionForStackFrameTests_B()));

    assertThat(failureDetail.getMessage())
        .isEqualTo(
            String.format(
                "Crashed: (%s) myMessage_A, (%s) myMessage_B",
                TEST_EXCEPTION_NAME, TEST_EXCEPTION_NAME));
    assertThat(failureDetail.hasCrash()).isTrue();
    Crash crash = failureDetail.getCrash();
    assertThat(crash.getCode()).isEqualTo(Code.CRASH_UNKNOWN);
    assertThat(crash.getOomDetectorOverride()).isFalse();

    assertThat(crash.getCausesCount()).isEqualTo(2);

    FailureDetails.Throwable outerCause = crash.getCauses(0);
    assertThat(outerCause.getMessage()).isEqualTo("myMessage_A");
    assertThat(outerCause.getThrowableClass()).isEqualTo(TEST_EXCEPTION_NAME);
    assertThat(outerCause.getStackTraceCount()).isAtLeast(2);
    assertThat(outerCause.getStackTrace(0))
        .contains(
            "com.google.devtools.build.lib.util.CrashFailureDetailsTest."
                + "functionForStackFrameTests_A");
    assertThat(outerCause.getStackTrace(1))
        .contains("com.google.devtools.build.lib.util.CrashFailureDetailsTest.nestedThrowables");

    FailureDetails.Throwable innerCause = crash.getCauses(1);
    assertThat(innerCause.getMessage()).isEqualTo("myMessage_B");
    assertThat(innerCause.getThrowableClass()).isEqualTo(TEST_EXCEPTION_NAME);
    assertThat(innerCause.getStackTraceCount()).isAtLeast(2);
    assertThat(innerCause.getStackTrace(0))
        .contains(
            "com.google.devtools.build.lib.util.CrashFailureDetailsTest."
                + "functionForStackFrameTests_B");
    assertThat(innerCause.getStackTrace(1))
        .contains("com.google.devtools.build.lib.util.CrashFailureDetailsTest.nestedThrowables");
  }

  @Test
  public void causeLimit() {
    // This test confirms that at most 5 throwables are recorded.
    TestException inner5 = new TestException("inner5");
    TestException inner4 = new TestException("inner4", inner5);
    TestException inner3 = new TestException("inner3", inner4);
    TestException inner2 = new TestException("inner2", inner3);
    TestException inner1 = new TestException("inner1", inner2);
    TestException outer = new TestException("outer", inner1);

    assertThat(CrashFailureDetails.forThrowable(outer).getCrash().getCausesCount()).isEqualTo(5);
  }

  @Test
  public void testMessageLimit() {
    TestException exception = new TestException("x".repeat(5000));

    String crashMessage =
        CrashFailureDetails.forThrowable(exception).getCrash().getCauses(0).getMessage();

    assertThat(crashMessage).hasLength(2000);
    assertThat(crashMessage).endsWith("[truncated]");
  }

  @Test
  public void causeCycle() {
    // This test confirms that throwables in a cause cycle are visited at most once.
    TestException inner2 = new TestException("inner2");
    TestException inner1 = new TestException("inner1", inner2);
    TestException outer = new TestException("outer", inner1);
    inner2.initCause(inner1);

    List<FailureDetails.Throwable> causesList =
        CrashFailureDetails.forThrowable(outer).getCrash().getCausesList();
    assertThat(causesList.stream().map(FailureDetails.Throwable::getMessage))
        .containsExactly("outer", "inner1", "inner2")
        .inOrder();
  }

  @Test
  public void deepStack() {
    ProtocolStringList stackTraceList =
        CrashFailureDetails.forThrowable(functionForDeepStackTrace(1001))
            .getCrash()
            .getCauses(0)
            .getStackTraceList();
    assertThat(stackTraceList).hasSize(1000);

    // Check that the deepest 1000 frames were recorded:
    for (String stackFrame : stackTraceList) {
      assertThat(stackFrame).contains("CrashFailureDetailsTest.functionForDeepStackTrace");
    }
  }

  @Test
  public void detailedExitConstruction_oom() {
    var detailedExitCode = CrashFailureDetails.detailedExitCodeForThrowable(new OutOfMemoryError());

    assertThat(detailedExitCode.getExitCode()).isEqualTo(ExitCode.OOM_ERROR);
    assertThat(detailedExitCode.getFailureDetail().getCrash().getOomDetectorOverride()).isFalse();
  }

  @Test
  public void detailedExitConstruction_wrappedOom() {
    var detailedExitCode =
        CrashFailureDetails.detailedExitCodeForThrowable(
            new IllegalStateException(new OutOfMemoryError()));

    assertThat(detailedExitCode.getExitCode()).isEqualTo(ExitCode.OOM_ERROR);
    assertThat(detailedExitCode.getFailureDetail().getCrash().getOomDetectorOverride()).isFalse();
  }

  @Test
  public void detailedExitConstruction_otherCrash() {
    var detailedExitCode =
        CrashFailureDetails.detailedExitCodeForThrowable(new IllegalStateException());

    assertThat(detailedExitCode.getExitCode()).isEqualTo(ExitCode.BLAZE_INTERNAL_ERROR);
    assertThat(detailedExitCode.getFailureDetail().getCrash().getOomDetectorOverride()).isFalse();
  }

  private enum ThrowableType {
    OUT_OF_MEMORY_ERROR(new OutOfMemoryError()),
    ILLEGAL_STATE_EXCEPTION(new IllegalStateException());

    ThrowableType(Throwable throwable) {
      this.throwable = throwable;
    }

    @SuppressWarnings("ImmutableEnumChecker")
    final Throwable throwable;
  }

  @Test
  public void detailExitConstruction_crashWithOomDetector_returnsOomCrash(
      @TestParameter ThrowableType throwableType) {
    CrashFailureDetails.setOomDetector(() -> true);
    var detailedExitCode =
        CrashFailureDetails.detailedExitCodeForThrowable(throwableType.throwable);

    assertThat(detailedExitCode.getExitCode()).isEqualTo(ExitCode.OOM_ERROR);
    if (throwableType == ThrowableType.OUT_OF_MEMORY_ERROR) {
      assertThat(detailedExitCode.getFailureDetail().getCrash().getOomDetectorOverride()).isFalse();
    } else {
      assertThat(detailedExitCode.getFailureDetail().getCrash().getOomDetectorOverride()).isTrue();
    }
  }

  private static TestException functionForStackFrameTests_A(TestException cause) {
    return new TestException("myMessage_A", cause);
  }

  private static TestException functionForStackFrameTests_B() {
    return new TestException("myMessage_B");
  }

  private static TestException functionForDeepStackTrace(int framesToBuild) {
    if (framesToBuild <= 1) {
      return new TestException("myMessage_deep");
    } else {
      return functionForDeepStackTrace(framesToBuild - 1);
    }
  }

  private static class TestException extends Exception {
    private TestException(String message) {
      super(message);
    }

    private TestException(String message, Throwable cause) {
      super(message, cause);
    }
  }
}
