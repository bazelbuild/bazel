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

/** Assembles the single-page version of the Build Encyclopedia. */
public class SinglePageBuildEncyclopediaProcessor extends BuildEncyclopediaProcessor {
  public SinglePageBuildEncyclopediaProcessor(
      RuleLinkExpander linkExpander,
      SourceUrlMapper urlMapper,
      ConfiguredRuleClassProvider ruleClassProvider) {
    super(linkExpander, urlMapper, ruleClassProvider);
  }

  /**
   * Collects and processes all the rule and attribute documentation in inputJavaDirs and generates
   * the Build Encyclopedia into outputDir.
   *
   * @param inputJavaDirs list of directories to scan for documentation in Java source code
   * @param buildEncyclopediaStardocProtos list of file paths of stardoc_output.ModuleInfo binary
   *     proto files generated from Build Encyclopedia entry point .bzl files; documentation from
   *     these protos takes precedence over documentation from {@code inputJavaDirs}
   * @param outputDir output directory where to write the build encyclopedia
   * @param denyList optional path to a file listing rules to not document
   */
  @Override
  public void generateDocumentation(
      List<String> inputJavaDirs,
      List<String> buildEncyclopediaStardocProtos,
      String outputDir,
      String denyList)
      throws BuildEncyclopediaDocException, IOException {
    BuildDocCollector collector = new BuildDocCollector(linkExpander, urlMapper, ruleClassProvider);
    Map<String, RuleDocumentation> ruleDocEntries =
        collector.collect(inputJavaDirs, buildEncyclopediaStardocProtos, denyList);
    warnAboutUndocumentedRules(
        Sets.difference(ruleClassProvider.getRuleClassMap().keySet(), ruleDocEntries.keySet()));
    RuleFamilies ruleFamilies = assembleRuleFamilies(ruleDocEntries.values());

    Page page = TemplateEngine.newPage(DocgenConsts.SINGLE_BE_TEMPLATE);

    // Add the rule link expander.
    page.add("expander", linkExpander);

    // Populate variables for Common Definitions section.
    page.add("typicalAttributes", expandCommonAttributes(PredefinedAttributes.TYPICAL_ATTRIBUTES));
    page.add("commonAttributes", expandCommonAttributes(PredefinedAttributes.COMMON_ATTRIBUTES));
    page.add("testAttributes", expandCommonAttributes(PredefinedAttributes.TEST_ATTRIBUTES));
    page.add("binaryAttributes", expandCommonAttributes(PredefinedAttributes.BINARY_ATTRIBUTES));

    // Popualte variables for Overview section.
    page.add("langSpecificRuleFamilies", ruleFamilies.langSpecific);
    page.add("genericRuleFamilies", ruleFamilies.generic);

    // Populate variables for Rules section.
    page.add("ruleFamilies", ruleFamilies.all);
    page.add("singlePage", true);

    writePage(page, outputDir, "build-encyclopedia.html");
  }
}
