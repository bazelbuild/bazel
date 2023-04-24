// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.TestAction.DummyAction;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.DetailedExitCode;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** {@link ActionExecutionException}Test */
@RunWith(JUnit4.class)
public final class ActionExecutionExceptionTest {

  @Test
  public void containsCauseMessage() {
    Exception e =
        new ActionExecutionException(
            "message",
            new Exception("cause"),
            new DummyAction(ActionsTestUtil.DUMMY_ARTIFACT, ActionsTestUtil.DUMMY_ARTIFACT),
            false,
            DetailedExitCode.of(
                FailureDetail.newBuilder().setExecution(Execution.getDefaultInstance()).build()));
    assertThat(e).hasMessageThat().contains("message");
    assertThat(e).hasMessageThat().contains("cause");
  }
}
