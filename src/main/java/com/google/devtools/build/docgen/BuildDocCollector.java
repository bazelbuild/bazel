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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Splitter;
import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.docgen.DocgenConsts.RuleType;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.RuleClass;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * Class that parses the documentation fragments of rule-classes and
 * generates the html format documentation.
 */
@VisibleForTesting
public class BuildDocCollector {
  private static final Splitter SHARP_SPLITTER = Splitter.on('#').limit(2).trimResults();

  private ConfiguredRuleClassProvider ruleClassProvider;
  private boolean printMessages;

  public BuildDocCollector(ConfiguredRuleClassProvider ruleClassProvider,
      boolean printMessages) {
    this.ruleClassProvider = ruleClassProvider;
    this.printMessages = printMessages;
  }

  /**
   * Parse the file containing black-listed rules for documentation. The list is simply a list of
   * rules separated by new lines. Line comments can be added to the file by starting them with #.
   *
   * @param blackList The name of the file containing the black list.
   * @return The set of black listed rules.
   * @throws IOException
   */
  @VisibleForTesting
  public static Set<String> readBlackList(String blackList) throws IOException {
    Set<String> result = new HashSet<String>();
    if (blackList != null && !blackList.isEmpty()) {
      File file = new File(blackList);
      try (BufferedReader reader = Files.newBufferedReader(file.toPath(), UTF_8)) {
        for (String line = reader.readLine(); line != null; line = reader.readLine()) {
          String rule = SHARP_SPLITTER.split(line).iterator().next();
          if (!rule.isEmpty()) {
            result.add(rule);
          }
        }
      }
    }
    return result;
  }

  /**
   * Collects all the rule and attribute documentation present in inputDirs, integrates the
   * attribute documentation in the rule documentation and returns the rule documentation.
   *
   * @param inputDirs list of directories to scan for documentation
   * @param blackList specify an optional black list file that list some rules that should
   *                  not be listed in the output.
   */
  public Map<String, RuleDocumentation> collect(String[] inputDirs, String blackList)
      throws BuildEncyclopediaDocException, IOException {
    // Read the blackList file
    Set<String> blacklistedRules = readBlackList(blackList);
    // RuleDocumentations are generated in order (based on rule type then alphabetically).
    // The ordering is also used to determine in which rule doc the common attribute docs are
    // generated (they are generated at the first appearance).
    Map<String, RuleDocumentation> ruleDocEntries = new TreeMap<>();
    // RuleDocumentationAttribute objects equal based on attributeName so they have to be
    // collected in a List instead of a Set.
    ListMultimap<String, RuleDocumentationAttribute> attributeDocEntries =
        LinkedListMultimap.create();

    // Map of rule class name to file that defined it.
    Map<String, File> ruleClassFiles = new HashMap<>();

    // Set of files already processed. The same file may be encountered multiple times because
    // directories are processed recursively, and an input directory may be a subdirectory of
    // another one.
    Set<File> processedFiles = new HashSet<>();

    for (String inputDir : inputDirs) {
      if (printMessages) {
        System.out.println(" Processing input directory: " + inputDir);
      }
      int ruleNum = ruleDocEntries.size();
      collectDocs(processedFiles, ruleClassFiles, ruleDocEntries, blacklistedRules,
          attributeDocEntries, new File(inputDir));
      if (printMessages) {
        System.out.println(" " + (ruleDocEntries.size() - ruleNum)
            + " rule documentations found.");
      }
    }

    processAttributeDocs(ruleDocEntries.values(), attributeDocEntries);
    RuleLinkExpander expander = buildRuleLinkExpander(ruleDocEntries.values());
    for (RuleDocumentation rule : ruleDocEntries.values()) {
      rule.setRuleLinkExpander(expander);
    }
    return ruleDocEntries;
  }

  /**
   * Generates an index mapping rule name to its normalized rule family name.
   */
  private RuleLinkExpander buildRuleLinkExpander(Iterable<RuleDocumentation> rules) {
    Map<String, String> index = new HashMap<>();
    for (RuleDocumentation rule : rules) {
      index.put(rule.getRuleName(), RuleFamily.normalize(rule.getRuleFamily()));
    }
    return new RuleLinkExpander(index);
  }

  /**
   * Go through all attributes of all documented rules and search the best attribute documentation
   * if exists. The best documentation is the closest documentation in the ancestor graph. E.g. if
   * java_library.deps documented in $rule and $java_rule then the one in $java_rule is going to
   * apply since it's a closer ancestor of java_library.
   */
  private void processAttributeDocs(Iterable<RuleDocumentation> ruleDocEntries,
      ListMultimap<String, RuleDocumentationAttribute> attributeDocEntries)
          throws BuildEncyclopediaDocException {
    for (RuleDocumentation ruleDoc : ruleDocEntries) {
      RuleClass ruleClass = ruleClassProvider.getRuleClassMap().get(ruleDoc.getRuleName());
      if (ruleClass != null) {
        if (ruleClass.isDocumented()) {
          Class<? extends RuleDefinition> ruleDefinition =
              ruleClassProvider.getRuleClassDefinition(ruleDoc.getRuleName());
          for (Attribute attribute : ruleClass.getAttributes()) {
            String attrName = attribute.getName();
            List<RuleDocumentationAttribute> attributeDocList =
                attributeDocEntries.get(attrName);
            if (attributeDocList != null) {
              // There are attribute docs for this attribute.
              // Search the closest one in the ancestor graph.
              // Note that there can be only one 'closest' attribute since we forbid multiple
              // inheritance of the same attribute in RuleClass.
              int minLevel = Integer.MAX_VALUE;
              RuleDocumentationAttribute bestAttributeDoc = null;
              for (RuleDocumentationAttribute attributeDoc : attributeDocList) {
                int level = attributeDoc.getDefinitionClassAncestryLevel(ruleDefinition);
                if (level >= 0 && level < minLevel) {
                  bestAttributeDoc = attributeDoc;
                  minLevel = level;
                }
              }
              if (bestAttributeDoc != null) {
                // Add reference to the Attribute that the attribute doc is associated with
                // in order to generate documentation for the Attribute.
                bestAttributeDoc.setAttribute(attribute);
                ruleDoc.addAttribute(bestAttributeDoc);
              // If there is no matching attribute doc try to add the common.
              } else if (ruleDoc.getRuleType().equals(RuleType.BINARY)
                  && PredefinedAttributes.BINARY_ATTRIBUTES.containsKey(attrName)) {
                ruleDoc.addAttribute(PredefinedAttributes.BINARY_ATTRIBUTES.get(attrName));
              } else if (ruleDoc.getRuleType().equals(RuleType.TEST)
                  && PredefinedAttributes.TEST_ATTRIBUTES.containsKey(attrName)) {
                ruleDoc.addAttribute(PredefinedAttributes.TEST_ATTRIBUTES.get(attrName));
              } else if (PredefinedAttributes.COMMON_ATTRIBUTES.containsKey(attrName)) {
                ruleDoc.addAttribute(PredefinedAttributes.COMMON_ATTRIBUTES.get(attrName));
              }
            }
          }
        }
      } else {
        throw ruleDoc.createException("Can't find RuleClass for " + ruleDoc.getRuleName());
      }
    }
  }

  /**
   * Goes through all the html files and subdirs under inputPath and collects the rule
   * and attribute documentations using the ruleDocEntries and attributeDocEntries variable.
   */
  public void collectDocs(
      Set<File> processedFiles,
      Map<String, File> ruleClassFiles,
      Map<String, RuleDocumentation> ruleDocEntries,
      Set<String> blackList,
      ListMultimap<String, RuleDocumentationAttribute> attributeDocEntries,
      File inputPath) throws BuildEncyclopediaDocException, IOException {
    if (processedFiles.contains(inputPath)) {
      return;
    }

    if (inputPath.isFile()) {
      if (DocgenConsts.JAVA_SOURCE_FILE_SUFFIX.apply(inputPath.getName())) {
        SourceFileReader sfr = new SourceFileReader(
            ruleClassProvider, inputPath.getAbsolutePath());
        sfr.readDocsFromComments();
        for (RuleDocumentation d : sfr.getRuleDocEntries()) {
          String ruleName = d.getRuleName();
          if (!blackList.contains(ruleName)) {
            if (ruleDocEntries.containsKey(ruleName)
                && !ruleClassFiles.get(ruleName).equals(inputPath)) {
              System.err.printf(
                  "WARNING: '%s' from '%s' overrides value already in map from '%s'\n",
                  d.getRuleName(), inputPath, ruleClassFiles.get(ruleName));
            }
            ruleClassFiles.put(ruleName, inputPath);
            ruleDocEntries.put(ruleName, d);
          }
        }
        if (attributeDocEntries != null) {
          // Collect all attribute documentations from this file.
          attributeDocEntries.putAll(sfr.getAttributeDocEntries());
        }
      }
    } else if (inputPath.isDirectory()) {
      for (File childPath : inputPath.listFiles()) {
        collectDocs(processedFiles, ruleClassFiles, ruleDocEntries, blackList,
            attributeDocEntries, childPath);
      }
    }

    processedFiles.add(inputPath);
  }
}
