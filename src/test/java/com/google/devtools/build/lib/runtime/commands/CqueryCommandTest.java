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

package com.google.devtools.build.lib.runtime.commands;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.server.FailureDetails.ConfigurableQuery.Code;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CqueryCommand}. */
@RunWith(JUnit4.class)
public final class CqueryCommandTest extends BuildIntegrationTestCase {
  private BlazeCommandDispatcher dispatcher;

  @Before
  public void setUpDispatcher() throws Exception {
    getRuntime().getCommandMap().put("cquery", new CqueryCommand());
    dispatcher = new BlazeCommandDispatcher(getRuntime());
    write(
        "foo/BUILD",
        """
        load("//test_defs:foo_library.bzl", "foo_library")

        foo_library(name = "a")
        """);
  }

  @Test
  public void maxQueryExpressionDepthExceededFails() throws Exception {
    RecordingOutErr outErr = new RecordingOutErr();
    BlazeCommandResult result =
        dispatcher.exec(
            ImmutableList.<String>builder()
                .add("cquery")
                .addAll(runtimeWrapper.getOptions())
                .add("--max_query_expression_depth=1", "deps(deps(//foo:a))")
                .build(),
            "test",
            outErr);

    assertThat(result.isSuccess()).isFalse();
    assertThat(result.getDetailedExitCode().getFailureDetail().getConfigurableQuery().getCode())
        .isEqualTo(Code.EXPRESSION_PARSE_FAILURE);
    assertThat(result.getDetailedExitCode().getFailureDetail().getMessage())
        .contains("query exceeds maximum nesting depth of 1");
  }

  @Test
  public void maxQueryExpressionDepthWithinLimitSucceeds() throws Exception {
    RecordingOutErr outErr = new RecordingOutErr();
    BlazeCommandResult result =
        dispatcher.exec(
            ImmutableList.<String>builder()
                .add("cquery")
                .addAll(runtimeWrapper.getOptions())
                .add("--max_query_expression_depth=2", "deps(deps(//foo:a))")
                .build(),
            "test",
            outErr);

    assertThat(result.isSuccess()).isTrue();
    assertThat(outErr.outAsLatin1()).contains("//foo:a");
  }
}
