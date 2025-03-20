// Copyright 2022 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

import com.google.devtools.build.lib.runtime.BlockWaitingModule.Task;
import com.google.devtools.build.lib.server.FailureDetails.Crash;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import java.util.concurrent.ExecutionException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;

/** Tests for {@link BlockWaitingModule}. */
@RunWith(JUnit4.class)
public final class BlockWaitingModuleTest {

  private static final DetailedExitCode CRASH =
      DetailedExitCode.of(
          FailureDetail.newBuilder()
              .setMessage("crash")
              .setCrash(Crash.newBuilder().setCode(Crash.Code.CRASH_UNKNOWN))
              .build());

  @Mock CommandEnvironment env;

  @Test
  public void testSubmitZeroTasks() throws Exception {
    // arrange
    BlockWaitingModule m = new BlockWaitingModule();

    // act
    m.beforeCommand(env);
    m.afterCommand();

    // nothing to assert
  }

  @Test
  public void testSubmitOneTask() throws Exception {
    // arrange
    BlockWaitingModule m = new BlockWaitingModule();
    Task t = mock(Task.class);

    // act
    m.beforeCommand(env);
    m.submit(t);
    m.afterCommand();

    // assert
    verify(t).call();
  }

  @Test
  public void testSubmitMultipleTasks() throws Exception {
    // arrange
    BlockWaitingModule m = new BlockWaitingModule();
    Task t1 = mock(Task.class);
    Task t2 = mock(Task.class);
    Task t3 = mock(Task.class);

    // act
    m.beforeCommand(env);
    m.submit(t1);
    m.submit(t2);
    m.submit(t3);
    m.afterCommand();

    // assert
    verify(t1).call();
    verify(t2).call();
    verify(t3).call();
  }

  @Test
  public void testTaskThrowsAbruptExitException() throws Exception {
    // arrange
    BlockWaitingModule m = new BlockWaitingModule();
    Task t = mock(Task.class);
    doThrow(new AbruptExitException(CRASH)).when(t).call();

    // act
    m.beforeCommand(env);
    m.submit(t);

    // assert
    Throwable e = assertThrows(AbruptExitException.class, m::afterCommand);
    assertThat(((AbruptExitException) e).getDetailedExitCode()).isEqualTo(CRASH);
  }

  @Test
  public void testTaskThrowsUnrecognizedException() throws Exception {
    // arrange
    BlockWaitingModule m = new BlockWaitingModule();
    Task t = mock(Task.class);
    doThrow(new IllegalStateException("illegal state")).when(t).call();

    // act
    m.beforeCommand(env);
    m.submit(t);

    // assert
    Throwable e = assertThrows(RuntimeException.class, m::afterCommand);
    assertThat(e).hasCauseThat().isInstanceOf(ExecutionException.class);
    assertThat(e).hasCauseThat().hasCauseThat().isInstanceOf(IllegalStateException.class);
    assertThat(e).hasCauseThat().hasCauseThat().hasMessageThat().contains("illegal state");
  }
}
