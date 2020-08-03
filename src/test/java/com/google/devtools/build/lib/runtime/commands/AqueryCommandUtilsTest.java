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
package com.google.devtools.build.lib.runtime.commands;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.query2.aquery.ActionGraphQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import com.google.devtools.build.lib.server.FailureDetails.ActionQuery;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link AqueryCommandUtils}. */
@RunWith(JUnit4.class)
public class AqueryCommandUtilsTest {
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
  public void testAqueryCommandGetTopLevelTargets_skyframeState_targetLabelSpecified()
      throws Exception {
    String query = "//some_target";
    QueryExpression expr = QueryParser.parse(query, functions);
    QueryException exception =
        assertThrows(
            QueryException.class,
            () ->
                AqueryCommandUtils.getTopLevelTargets(
                    /* universeScope= */ ImmutableList.of(),
                    expr,
                    /* queryCurrentSkyframeState= */ true,
                    query));
    assertThat(exception).hasMessageThat().contains("Error while parsing '" + query);
    assertThat(exception)
        .hasMessageThat()
        .contains("with --skyframe_state is currently not supported");
    assertThat(exception.getFailureDetail().getActionQuery().getCode())
        .isEqualTo(ActionQuery.Code.TOP_LEVEL_TARGETS_WITH_SKYFRAME_STATE_NOT_SUPPORTED);
  }
}
