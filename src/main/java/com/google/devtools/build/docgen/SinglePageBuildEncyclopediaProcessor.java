// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Assembles the single-page version of the Build Encyclopedia.
 */
public class SinglePageBuildEncyclopediaProcessor extends BuildEncyclopediaProcessor {
  public SinglePageBuildEncyclopediaProcessor(
      String productName, ConfiguredRuleClassProvider ruleClassProvider) {
    super(productName, ruleClassProvider);
  }

  /**
   * Collects and processes all the rule and attribute documentation in inputDirs and generates the
   * Build Encyclopedia into the outputDir.
   *
   * @param inputDirs list of directory to scan for document in the source code
   * @param outputDir output directory where to write the build encyclopedia
   * @param denyList optional path to a file listing rules to not document
   */
  @Override
  public void generateDocumentation(List<String> inputDirs, String outputDir, String denyList)
      throws BuildEncyclopediaDocException, IOException {
    BuildDocCollector collector = new BuildDocCollector(productName, ruleClassProvider, false);
    RuleLinkExpander expander = new RuleLinkExpander(productName, true);
    Map<String, RuleDocumentation> ruleDocEntries =
        collector.collect(inputDirs, denyList, expander);
    warnAboutUndocumentedRules(
        Sets.difference(ruleClassProvider.getRuleClassMap().keySet(), ruleDocEntries.keySet()));
    RuleFamilies ruleFamilies = assembleRuleFamilies(ruleDocEntries.values());

    Page page = TemplateEngine.newPage(DocgenConsts.SINGLE_BE_TEMPLATE);

    // Add the rule link expander.
    page.add("expander", expander);

    // Populate variables for Common Definitions section.
    page.add("commonAttributes",
        expandCommonAttributes(PredefinedAttributes.COMMON_ATTRIBUTES, expander));
    page.add("testAttributes",
        expandCommonAttributes(PredefinedAttributes.TEST_ATTRIBUTES, expander));
    page.add("binaryAttributes",
        expandCommonAttributes(PredefinedAttributes.BINARY_ATTRIBUTES, expander));

    // Popualte variables for Overview section.
    page.add("langSpecificRuleFamilies", ruleFamilies.langSpecific);
    page.add("genericRuleFamilies", ruleFamilies.generic);

    // Populate variables for Rules section.
    page.add("ruleFamilies", ruleFamilies.all);
    page.add("singlePage", true);

    writePage(page, outputDir, "build-encyclopedia.html");
  }
}
