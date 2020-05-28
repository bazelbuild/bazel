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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.buildtool.AqueryBuildTool.AqueryActionFilterException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.query2.aquery.ActionGraphQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.commands.QueryCommand;
import com.google.devtools.common.options.OptionsParser;
import java.util.ArrayList;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for aquery. */
@RunWith(JUnit4.class)
public class AqueryBuildToolTest extends BuildIntegrationTestCase {
  private ImmutableMap<String, QueryFunction> functions;

  @Before
  public final void setFunctions() throws Exception {
    ImmutableMap.Builder<String, QueryFunction> builder = ImmutableMap.builder();

    for (QueryFunction queryFunction : ActionGraphQueryEnvironment.FUNCTIONS) {
      builder.put(queryFunction.getName(), queryFunction);
    }

    for (QueryFunction queryFunction : ActionGraphQueryEnvironment.AQUERY_FUNCTIONS) {
      builder.put(queryFunction.getName(), queryFunction);
    }

    functions = builder.build();
  }

  @Test
  public void testAqueryBuildToolConstructor_wrongAqueryFilterFormat_throwsError()
      throws Exception {
    QueryExpression expr = QueryParser.parse("deps(inputs('abc', //abc))", functions);
    OptionsParser optionsParser = runtimeWrapper.createOptionsParser();
    Command command = QueryCommand.class.getAnnotation(Command.class);
    CommandEnvironment env =
        getBlazeWorkspace().initCommand(command, optionsParser, new ArrayList<>(), 0L, 0L);

    assertThrows(AqueryActionFilterException.class, () -> new AqueryBuildTool(env, expr));
  }

  @Test
  public void testAqueryBuildToolConstructor_wrongPatternSyntax_throwsError() throws Exception {
    QueryExpression expr = QueryParser.parse("inputs('*abc', //abc)", functions);
    OptionsParser optionsParser = runtimeWrapper.createOptionsParser();
    Command command = QueryCommand.class.getAnnotation(Command.class);
    CommandEnvironment env =
        getBlazeWorkspace().initCommand(command, optionsParser, new ArrayList<>(), 0L, 0L);
    AqueryActionFilterException thrown =
        assertThrows(AqueryActionFilterException.class, () -> new AqueryBuildTool(env, expr));
    assertThat(thrown).hasMessageThat().contains("Wrong query syntax:");
  }
}
