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

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.Iterables;
import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import com.google.devtools.build.docgen.DocgenConsts.RuleType;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.util.Pair;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * A class to assemble documentation for the Build Encyclopedia. This class uses
 * {@link BuildDocCollector} to extract documentation fragments from rule classes.
 */
public class BuildEncyclopediaProcessor {
  private static final Predicate<String> RULE_WORTH_DOCUMENTING = new Predicate<String>() {
    @Override
    public boolean apply(String name) {
      return !name.contains("$");
    }
  };

  private ConfiguredRuleClassProvider ruleClassProvider;

  /**
   * Creates the BuildEncyclopediaProcessor instance. The ruleClassProvider parameter
   * is used for rule class hierarchy and attribute checking.
   *
   */
  public BuildEncyclopediaProcessor(ConfiguredRuleClassProvider ruleClassProvider) {
    this.ruleClassProvider = Preconditions.checkNotNull(ruleClassProvider);
  }

  /**
   * Collects and processes all the rule and attribute documentation in inputDirs and
   * generates the Build Encyclopedia into the outputDir.
   *
   * @param inputDirs list of directory to scan for document in the source code
   * @param outputRootDir output directory where to write the build encyclopedia
   * @param blackList optional path to a file listing rules to not document
   */
  public void generateDocumentation(String[] inputDirs, String outputDir, String blackList)
      throws BuildEncyclopediaDocException, IOException {
    writeStaticPage(outputDir, "make-variables");
    writeStaticPage(outputDir, "predefined-python-variables");
    writeStaticPage(outputDir, "functions");
    writeCommonDefinitionsPage(outputDir);

    BuildDocCollector collector = new BuildDocCollector(ruleClassProvider, false);
    Map<String, RuleDocumentation> ruleDocEntries = collector.collect(inputDirs, blackList);
    warnAboutUndocumentedRules(
        Sets.difference(ruleClassProvider.getRuleClassMap().keySet(), ruleDocEntries.keySet()));

    writeRuleDocs(outputDir, ruleDocEntries.values());
  }

  private void writeStaticPage(String outputDir, String name) throws IOException {
    File file = new File(outputDir + "/" + name + ".html");
    Page page = TemplateEngine.newPage(
        "com/google/devtools/build/docgen/templates/be/" + name + ".vm");
    page.write(file);
  }

  private void writeCommonDefinitionsPage(String outputDir) throws IOException {
    File file = new File(outputDir + "/common-definitions.html");
    Page page = TemplateEngine.newPage(DocgenConsts.COMMON_DEFINITIONS_TEMPLATE);
    page.add("commonAttributes", PredefinedAttributes.COMMON_ATTRIBUTES);
    page.add("testAttributes", PredefinedAttributes.TEST_ATTRIBUTES);
    page.add("binaryAttributes", PredefinedAttributes.BINARY_ATTRIBUTES);
    page.write(file);
  }

  private void writeRuleDocs(String outputDir, Iterable<RuleDocumentation> docEntries)
      throws BuildEncyclopediaDocException, IOException {
    // Separate rule families into language-specific and generic ones.
    Set<String> languageSpecificRuleFamilies = new TreeSet<>();
    Set<String> genericRuleFamilies = new TreeSet<>();
    separateRuleFamilies(docEntries, languageSpecificRuleFamilies, genericRuleFamilies);

    // Create a mapping of rules based on rule type and family.
    Map<String, ListMultimap<RuleType, RuleDocumentation>> ruleMapping = new HashMap<>();
    createRuleMapping(docEntries, ruleMapping);

    // Pairs of (normalized rule family name, rule family name), which used for generating
    // the BE navigation with rule families listed in the same order as those listed in
    // the overview table.
    List<Pair<String, String>> ruleFamilyNames = new ArrayList<>(
        languageSpecificRuleFamilies.size() + genericRuleFamilies.size());

    List<SummaryRuleFamily> languageSpecificSummaryFamilies =
        new ArrayList<SummaryRuleFamily>(languageSpecificRuleFamilies.size());
    for (String ruleFamily : languageSpecificRuleFamilies) {
      ListMultimap<RuleType, RuleDocumentation> ruleTypeMap = ruleMapping.get(ruleFamily);
      String ruleFamilyId = RuleDocumentation.normalize(ruleFamily);
      languageSpecificSummaryFamilies.add(
          new SummaryRuleFamily(ruleTypeMap, ruleFamily, ruleFamilyId));
      writeRuleDoc(outputDir, ruleFamily, ruleFamilyId, ruleTypeMap);
      ruleFamilyNames.add(Pair.<String, String>of(ruleFamilyId, ruleFamily));
    }

    List<SummaryRuleFamily> otherSummaryFamilies =
        new ArrayList<SummaryRuleFamily>(genericRuleFamilies.size());
    for (String ruleFamily : genericRuleFamilies) {
      ListMultimap<RuleType, RuleDocumentation> ruleTypeMap = ruleMapping.get(ruleFamily);
      String ruleFamilyId = RuleDocumentation.normalize(ruleFamily);
      otherSummaryFamilies.add(new SummaryRuleFamily(ruleTypeMap, ruleFamily, ruleFamilyId));
      writeRuleDoc(outputDir, ruleFamily, ruleFamilyId, ruleTypeMap);
      ruleFamilyNames.add(Pair.<String, String>of(ruleFamilyId, ruleFamily));
    }
    writeOverviewPage(outputDir, languageSpecificSummaryFamilies, otherSummaryFamilies);
    writeBeNav(outputDir, ruleFamilyNames);
  }

  private void writeOverviewPage(String outputDir,
      List<SummaryRuleFamily> languageSpecificSummaryFamilies,
      List<SummaryRuleFamily> otherSummaryFamilies)
      throws BuildEncyclopediaDocException, IOException {
    File file = new File(outputDir + "/overview.html");
    Page page = TemplateEngine.newPage(DocgenConsts.OVERVIEW_TEMPLATE);
    page.add("langSpecificSummaryFamilies", languageSpecificSummaryFamilies);
    page.add("otherSummaryFamilies", otherSummaryFamilies);
    page.write(file);
  }

  private void writeRuleDoc(String outputDir, String ruleFamily, String ruleFamilyId,
      ListMultimap<RuleType, RuleDocumentation> ruleTypeMap)
      throws BuildEncyclopediaDocException, IOException {
    List<RuleDocumentation> rules = new LinkedList<>();
    rules.addAll(ruleTypeMap.get(RuleType.BINARY));
    rules.addAll(ruleTypeMap.get(RuleType.LIBRARY));
    rules.addAll(ruleTypeMap.get(RuleType.TEST));
    rules.addAll(ruleTypeMap.get(RuleType.OTHER));

    File file = new File(outputDir + "/" + ruleFamilyId + ".html");
    Page page = TemplateEngine.newPage(DocgenConsts.RULES_TEMPLATE);
    page.add("ruleFamily", ruleFamily);
    page.add("ruleFamilyId", ruleFamilyId);
    page.add("ruleDocs", rules);
    page.write(file);
  }

  private void writeBeNav(String outputDir, List<Pair<String, String>> ruleFamilyNames)
      throws IOException {
    File file = new File(outputDir + "/be-nav.html");
    Page page = TemplateEngine.newPage(DocgenConsts.BE_NAV_TEMPLATE);
    page.add("ruleFamilyNames", ruleFamilyNames);
    page.write(file);
  }

  /**
   * Create a mapping of rules based on rule type and family.
   */
  private void createRuleMapping(Iterable<RuleDocumentation> docEntries,
      Map<String, ListMultimap<RuleType, RuleDocumentation>> ruleMapping)
      throws BuildEncyclopediaDocException {
    for (RuleDocumentation ruleDoc : docEntries) {
      RuleClass ruleClass = ruleClassProvider.getRuleClassMap().get(ruleDoc.getRuleName());
      if (ruleClass != null) {
        String ruleFamily = ruleDoc.getRuleFamily();
        if (!ruleMapping.containsKey(ruleFamily)) {
          ruleMapping.put(ruleFamily, LinkedListMultimap.<RuleType, RuleDocumentation>create());
        }
        if (ruleClass.isDocumented()) {
          ruleMapping.get(ruleFamily).put(ruleDoc.getRuleType(), ruleDoc);
        }
      } else {
        throw ruleDoc.createException("Can't find RuleClass for " + ruleDoc.getRuleName());
      }
    }
  }

  /**
   * Separates all rule families in docEntries into language-specific rules and generic rules.
   */
  private void separateRuleFamilies(Iterable<RuleDocumentation> docEntries,
      Set<String> languageSpecificRuleFamilies, Set<String> genericRuleFamilies)
      throws BuildEncyclopediaDocException {
    for (RuleDocumentation ruleDoc : docEntries) {
      if (ruleDoc.isLanguageSpecific()) {
        if (genericRuleFamilies.contains(ruleDoc.getRuleFamily())) {
          throw ruleDoc.createException("The rule is marked as being language-specific, but other "
              + "rules of the same family have already been marked as being not.");
        }
        languageSpecificRuleFamilies.add(ruleDoc.getRuleFamily());
      } else {
        if (languageSpecificRuleFamilies.contains(ruleDoc.getRuleFamily())) {
          throw ruleDoc.createException("The rule is marked as being generic, but other rules of "
              + "the same family have already been marked as being language-specific.");
        }
        genericRuleFamilies.add(ruleDoc.getRuleFamily());
      }
    }
  }

  private static void warnAboutUndocumentedRules(Iterable<String> rulesWithoutDocumentation) {
      Iterable<String> undocumentedRules = Iterables.filter(rulesWithoutDocumentation,
          RULE_WORTH_DOCUMENTING);
      System.err.printf("WARNING: The following rules are undocumented: [%s]\n",
          Joiner.on(", ").join(Ordering.<String>natural().immutableSortedCopy(undocumentedRules)));
  }
}
