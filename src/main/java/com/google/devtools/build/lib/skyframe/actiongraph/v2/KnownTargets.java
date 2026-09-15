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
package com.google.devtools.build.lib.skyframe.actiongraph.v2;

import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Target;
import com.google.devtools.build.lib.util.Pair;
import java.io.IOException;

/**
 * Cache for RuleConfiguredTargets in the action graph.
 *
 * <p>The cache maps a target identifier, which is a {@code Pair<String, String>} of
 * (label, rule class string).
 * */
public class KnownTargets extends BaseCache<Pair<String, String>, Target> {

  private final KnownRuleClassStrings knownRuleClassStrings;

  KnownTargets(
      AqueryOutputHandler aqueryOutputHandler, KnownRuleClassStrings knownRuleClassStrings) {
    super(aqueryOutputHandler);
    this.knownRuleClassStrings = knownRuleClassStrings;
  }

  @Override
  Target createProto(Pair<String, String> targetIdentifier, int id)
      throws IOException, InterruptedException {
    String labelString = targetIdentifier.getFirst();
    String ruleClassString = targetIdentifier.getSecond();
    Target.Builder targetBuilder = Target.newBuilder().setId(id).setLabel(labelString);
    if (ruleClassString != null) {
      targetBuilder.setRuleClassId(
          knownRuleClassStrings.dataToIdAndStreamOutputProto(ruleClassString));
    }
    return targetBuilder.build();
  }

  @Override
  void toOutput(Target targetProto) throws IOException {
    aqueryOutputHandler.outputTarget(targetProto);
  }
}
