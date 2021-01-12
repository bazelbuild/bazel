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

package com.google.devtools.build.docgen;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/** Assembles a list of native rules that can be exported to a builtin.proto file. */
public class ProtoFileBuildEncyclopediaProcessor extends BuildEncyclopediaProcessor {
  private ImmutableList<RuleDocumentation> nativeRules = null;

  public ProtoFileBuildEncyclopediaProcessor(
      String productName, ConfiguredRuleClassProvider ruleClassProvider) {
    super(productName, ruleClassProvider);
  }

  /*
   * Collects and processes all rule and attribute documentation in inputDirs and generates a list
   * of RuleDocumentation objects.
   */
  @Override
  public void generateDocumentation(List<String> inputDirs, String outputFile, String denyList)
      throws BuildEncyclopediaDocException, IOException {
    BuildDocCollector collector = new BuildDocCollector(productName, ruleClassProvider, false);
    RuleLinkExpander expander = new RuleLinkExpander(productName, true);
    Map<String, RuleDocumentation> ruleDocEntries =
        collector.collect(inputDirs, denyList, expander);
    RuleFamilies ruleFamilies = assembleRuleFamilies(ruleDocEntries.values());
    ImmutableList.Builder<RuleDocumentation> ruleDocsBuilder = new ImmutableList.Builder<>();

    for (RuleFamily entry : ruleFamilies.all) {
      for (RuleDocumentation doc : entry.getRules()) {
        ruleDocsBuilder.add(doc);
      }
    }
    nativeRules = ruleDocsBuilder.build();
  }

  public ImmutableList<RuleDocumentation> getNativeRules() {
    return nativeRules;
  }
}
