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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
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
   * generates the Build Encyclopedia into the outputRootDir.
   * 
   * @param inputDirs list of directory to scan for document in the source code
   * @param outputRootDir output directory where to write the build encyclopedia
   * @param blackList optional path to a file listing rules to not document
   */
  public void generateDocumentation(String[] inputDirs, String outputRootDir, String blackList)
      throws BuildEncyclopediaDocException, IOException {
    File buildEncyclopediaPath = setupDirectories(outputRootDir);

    Page page = TemplateEngine.newPage(
        "com/google/devtools/build/docgen/templates/build-encyclopedia.vm");

    BuildDocCollector collector = new BuildDocCollector(ruleClassProvider, false);
    Map<String, RuleDocumentation> ruleDocEntries = collector.collect(inputDirs, blackList);
    warnAboutUndocumentedRules(
        Sets.difference(ruleClassProvider.getRuleClassMap().keySet(), ruleDocEntries.keySet()));
    writeRuleClassDocs(ruleDocEntries.values(), page);
    page.write(buildEncyclopediaPath);
  }


  /**
   * Categorizes, checks and prints all the rule-class documentations.
   */
  private void writeRuleClassDocs(Iterable<RuleDocumentation> docEntries, Page page)
      throws BuildEncyclopediaDocException, IOException {
    Set<RuleDocumentation> binaryDocs = new TreeSet<>();
    Set<RuleDocumentation> libraryDocs = new TreeSet<>();
    Set<RuleDocumentation> testDocs = new TreeSet<>();
    Set<RuleDocumentation> otherDocs = new TreeSet<>();

    for (RuleDocumentation doc : docEntries) {
      RuleClass ruleClass = ruleClassProvider.getRuleClassMap().get(doc.getRuleName());
      if (!ruleClass.isDocumented()) {
        continue;
      }

      if (doc.isLanguageSpecific()) {
        switch(doc.getRuleType()) {
          case BINARY:
            binaryDocs.add(doc);
            break;
          case LIBRARY:
            libraryDocs.add(doc);
            break;
          case TEST:
            testDocs.add(doc);
            break;
          case OTHER:
            otherDocs.add(doc);
            break;
        }
      } else {
        otherDocs.add(doc);
      }
    }

    renderBeHeader(docEntries, page);

    page.add("binaryDocs", binaryDocs);
    page.add("libraryDocs", libraryDocs);
    page.add("testDocs", testDocs);
    page.add("otherDocs", otherDocs);
  }

  private void renderBeHeader(Iterable<RuleDocumentation> docEntries, Page page)
      throws BuildEncyclopediaDocException {
    // Separate rule families into language-specific and generic ones.
    Set<String> languageSpecificRuleFamilies = new TreeSet<>();
    Set<String> genericRuleFamilies = new TreeSet<>();
    separateRuleFamilies(docEntries, languageSpecificRuleFamilies, genericRuleFamilies);

    // Create a mapping of rules based on rule type and family.
    Map<String, ListMultimap<RuleType, RuleDocumentation>> ruleMapping = new HashMap<>();
    createRuleMapping(docEntries, ruleMapping);

    List<SummaryRuleFamily> languageSpecificSummaryFamilies =
        new ArrayList<SummaryRuleFamily>(languageSpecificRuleFamilies.size());
    for (String ruleFamily : languageSpecificRuleFamilies) {
      languageSpecificSummaryFamilies.add(
          new SummaryRuleFamily(ruleMapping.get(ruleFamily), ruleFamily));
    }

    List<SummaryRuleFamily> otherSummaryFamilies =
        new ArrayList<SummaryRuleFamily>(genericRuleFamilies.size());
    for (String ruleFamily : genericRuleFamilies) {
      otherSummaryFamilies.add(
          new SummaryRuleFamily(ruleMapping.get(ruleFamily), ruleFamily));
    }

    page.add("langSpecificSummaryFamilies", languageSpecificSummaryFamilies);
    page.add("otherSummaryFamilies", otherSummaryFamilies);
    page.add("commonAttributes", PredefinedAttributes.COMMON_ATTRIBUTES);
    page.add("testAttributes", PredefinedAttributes.TEST_ATTRIBUTES);
    page.add("binaryAttributes", PredefinedAttributes.BINARY_ATTRIBUTES);
    page.add(DocgenConsts.VAR_LEFT_PANEL, generateLeftNavigationPanel(docEntries));
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

  private String generateLeftNavigationPanel(Iterable<RuleDocumentation> docEntries) {
    // Order the rules alphabetically. At this point they are ordered according to
    // RuleDocumentation.compareTo() which is not alphabetical.
    TreeMap<String, String> ruleNames = new TreeMap<>();
    for (RuleDocumentation ruleDoc : docEntries) {
      String ruleName = ruleDoc.getRuleName();
      ruleNames.put(ruleName.toLowerCase(), ruleName);
    }
    StringBuilder sb = new StringBuilder();
    for (String ruleName : ruleNames.values()) {
      RuleClass ruleClass = ruleClassProvider.getRuleClassMap().get(ruleName);
      Preconditions.checkNotNull(ruleClass);
      if (ruleClass.isDocumented()) {
        sb.append(String.format("<a href=\"#%s\">%s</a><br/>\n", ruleName, ruleName));
      }
    }
    return sb.toString();
  }

  private File setupDirectories(String outputRootDir) {
    if (outputRootDir != null) {
      File outputRootPath = new File(outputRootDir);
      outputRootPath.mkdirs();
      return new File(outputRootDir + File.separator + DocgenConsts.BUILD_ENCYCLOPEDIA_NAME);
    } else {
      return new File(DocgenConsts.BUILD_ENCYCLOPEDIA_NAME);
    }
  }

  private static void warnAboutUndocumentedRules(Iterable<String> rulesWithoutDocumentation) {
      Iterable<String> undocumentedRules = Iterables.filter(rulesWithoutDocumentation,
          RULE_WORTH_DOCUMENTING);
      System.err.printf("WARNING: The following rules are undocumented: [%s]\n",
          Joiner.on(", ").join(Ordering.<String>natural().immutableSortedCopy(undocumentedRules)));
  }
}
