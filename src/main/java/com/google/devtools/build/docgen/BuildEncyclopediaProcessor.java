// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.collect.ImmutableSet;
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
   */
  public void generateDocumentation(String[] inputDirs, String outputRootDir)
      throws BuildEncyclopediaDocException, IOException {
    File buildEncyclopediaPath = setupDirectories(outputRootDir);

    Page page = TemplateEngine.newPage(
        "com/google/devtools/build/docgen/templates/build-encyclopedia.vm");

    BuildDocCollector collector = new BuildDocCollector(ruleClassProvider, false);
    Map<String, RuleDocumentation> ruleDocEntries = collector.collect(inputDirs);
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

    page.add(DocgenConsts.VAR_SECTION_BINARY, getRuleDocs(binaryDocs));
    page.add(DocgenConsts.VAR_SECTION_LIBRARY, getRuleDocs(libraryDocs));
    page.add(DocgenConsts.VAR_SECTION_TEST, getRuleDocs(testDocs));
    page.add(DocgenConsts.VAR_SECTION_OTHER, getRuleDocs(otherDocs));
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

    String languageSpecificTable;
    {
      StringBuilder sb = new StringBuilder();

      sb.append("<colgroup span=\"6\" width=\"20%\"></colgroup>\n")
        .append("<tr><th>Language</th><th>Binary rules</th><th>Library rules</th>"
          + "<th>Test rules</th><th>Other rules</th><th></th></tr>\n");

      // Generate the table.
      for (String ruleFamily : languageSpecificRuleFamilies) {
        generateHeaderTableRuleFamily(sb, ruleMapping.get(ruleFamily), ruleFamily);
      }
      languageSpecificTable = sb.toString();
    }

    String otherRulesTable;
    {
      StringBuilder sb = new StringBuilder();

      sb.append("<colgroup span=\"6\" width=\"20%\"></colgroup>\n");
      for (String ruleFamily : genericRuleFamilies) {
        generateHeaderTableRuleFamily(sb, ruleMapping.get(ruleFamily), ruleFamily);
      }
      otherRulesTable = sb.toString();
    }
    page.add(DocgenConsts.VAR_LANG_SPECIFIC_HEADER_TABLE, languageSpecificTable);
    page.add(DocgenConsts.VAR_OTHER_RULES_HEADER_TABLE, otherRulesTable);
    page.add(DocgenConsts.VAR_COMMON_ATTRIBUTE_DEFINITION,
        generateCommonAttributeDocs(
            PredefinedAttributes.COMMON_ATTRIBUTES, DocgenConsts.COMMON_ATTRIBUTES));
    page.add(DocgenConsts.VAR_TEST_ATTRIBUTE_DEFINITION,
        generateCommonAttributeDocs(
            PredefinedAttributes.TEST_ATTRIBUTES, DocgenConsts.TEST_ATTRIBUTES));
    page.add(DocgenConsts.VAR_BINARY_ATTRIBUTE_DEFINITION,
        generateCommonAttributeDocs(
            PredefinedAttributes.BINARY_ATTRIBUTES, DocgenConsts.BINARY_ATTRIBUTES));
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

  private String generateCommonAttributeDocs(Map<String, RuleDocumentationAttribute> attributes,
      String attributeGroupName) throws BuildEncyclopediaDocException {
    RuleDocumentation ruleDoc = new RuleDocumentation(
        attributeGroupName, "OTHER", null, null, 0, null, ImmutableSet.<String>of(),
        ruleClassProvider);
    for (RuleDocumentationAttribute attribute : attributes.values()) {
      ruleDoc.addAttribute(attribute);
    }
    return ruleDoc.generateAttributeDefinitions();
  }

  private void generateHeaderTableRuleFamily(StringBuilder sb,
      ListMultimap<RuleType, RuleDocumentation> ruleTypeMap, String ruleFamily) {
    sb.append("<tr>\n")
      .append(String.format("<td class=\"lang\">%s</td>\n", ruleFamily));
    boolean otherRulesSplitted = false;
    for (RuleType ruleType : DocgenConsts.RuleType.values()) {
      sb.append("<td>");
      int i = 0;
      List<RuleDocumentation> ruleDocList = ruleTypeMap.get(ruleType);
      for (RuleDocumentation ruleDoc : ruleDocList) {
        if (i > 0) {
          if (ruleType.equals(RuleType.OTHER)
              && ruleDocList.size() >= 4 && i == (ruleDocList.size() + 1) / 2) {
            // Split 'other rules' into two columns if there are too many of them.
            sb.append("</td>\n<td>");
            otherRulesSplitted = true;
          } else {
            sb.append("<br/>");
          }
        }
        String ruleName = ruleDoc.getRuleName();
        String deprecatedString = ruleDoc.hasFlag(DocgenConsts.FLAG_DEPRECATED)
            ? " class=\"deprecated\"" : "";
        sb.append(String.format("<a href=\"#%s\"%s>%s</a>", ruleName, deprecatedString, ruleName));
        i++;
      }
      sb.append("</td>\n");
    }
    // There should be 6 columns.
    if (!otherRulesSplitted) {
      sb.append("<td></td>\n");
    }
    sb.append("</tr>\n");
  }

  private String getRuleDocs(Iterable<RuleDocumentation> docEntries) {
    StringBuilder sb = new StringBuilder();
    for (RuleDocumentation doc : docEntries) {
      sb.append(doc.getHtmlDocumentation());
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
