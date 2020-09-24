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

package com.google.devtools.build.lib.runtime.commands;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.common.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.query.output.OutputFormatter;
import com.google.devtools.build.lib.query2.query.output.QueryOptions;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.QueryRuntimeHelper;
import com.google.devtools.build.lib.server.FailureDetails.Query.Code;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Either;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.common.options.Options;
import java.util.Optional;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/** Test for {@link QueryCommand}. */
@RunWith(JUnit4.class)
public class QueryCommandTest {

  @Rule public final MockitoRule mockito = MockitoJUnit.rule();

  @Mock private AbstractBlazeQueryEnvironment<Target> mockQueryEnvironment;

  private QueryCommand underTest;

  @Before
  public void setUp() {
    this.underTest = new QueryCommand();
    when(mockQueryEnvironment.getFunctions()).thenReturn(ImmutableList.of());
  }

  @Test
  public void testQuerySyntaxErrorResultsInCommandLineExitStatusWithDetails() {
    StoredEventHandler storedEventHandler = new StoredEventHandler();

    Either<BlazeCommandResult, QueryEvalResult> result =
        underTest.doQuery(
            "terrible syntax",
            mockCommandEnvironment(new Reporter(new EventBus(), storedEventHandler)),
            Options.getDefaults(QueryOptions.class),
            /*streamResults=*/ false,
            mock(OutputFormatter.class),
            mockQueryEnvironment,
            mock(QueryRuntimeHelper.class));

    Optional<DetailedExitCode> detailedExitCode =
        result.map(r -> Optional.of(r.getDetailedExitCode()), r -> Optional.empty());
    assertWithMessage("Expected to contain BlazeCommandResult, got: %s", result)
        .that(detailedExitCode.isPresent())
        .isTrue();

    assertThat(detailedExitCode.get().getExitCode()).isEqualTo(ExitCode.COMMAND_LINE_ERROR);
    assertThat(detailedExitCode.get().getFailureDetail().getQuery().getCode())
        .isEqualTo(Code.SYNTAX_ERROR);

    assertThat(storedEventHandler.getEvents()).hasSize(1);
    assertThat(storedEventHandler.getEvents().get(0).getMessage())
        .startsWith("Error while parsing 'terrible syntax'");
  }

  private static CommandEnvironment mockCommandEnvironment(Reporter reporter) {
    CommandEnvironment result = mock(CommandEnvironment.class);
    when(result.getReporter()).thenReturn(reporter);
    return result;
  }
}
