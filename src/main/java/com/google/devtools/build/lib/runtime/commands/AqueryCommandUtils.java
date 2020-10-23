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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.server.FailureDetails.ActionQuery;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/** The utility class for {@link AqueryCommand} */
public final class AqueryCommandUtils {

  private AqueryCommandUtils() {}

  /**
   * Get the list of top-level targets of the query from universe scope and the query expression.
   *
   * @throws QueryException if targets were specified in the query expression together with
   *     --skyframe_state flag
   */
  static ImmutableList<String> getTopLevelTargets(
      List<String> universeScope, @Nullable QueryExpression expr, boolean queryCurrentSkyframeState)
      throws QueryException {
    if (expr == null) {
      return ImmutableList.copyOf(universeScope);
    }

    ImmutableList<String> topLevelTargets;
    if (universeScope.isEmpty()) {
      Set<String> targetPatternSet = new LinkedHashSet<>();
      expr.collectTargetPatterns(targetPatternSet);
      topLevelTargets = ImmutableList.copyOf(targetPatternSet);
    } else {
      topLevelTargets = ImmutableList.copyOf(universeScope);
    }

    if (queryCurrentSkyframeState && !topLevelTargets.isEmpty()) {
      throw new QueryException(
          "Error while parsing '"
              + expr.toTrunctatedString()
              + "': Specifying build target(s) "
              + topLevelTargets
              + " with --skyframe_state is currently not supported.",
          ActionQuery.Code.TOP_LEVEL_TARGETS_WITH_SKYFRAME_STATE_NOT_SUPPORTED);
    }

    return topLevelTargets;
  }
}
