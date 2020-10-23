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

  private final String productName;
  private final ConfiguredRuleClassProvider ruleClassProvider;
  private final boolean printMessages;

  public BuildDocCollector(
      String productName, ConfiguredRuleClassProvider ruleClassProvider, boolean printMessages) {
    this.productName = productName;
    this.ruleClassProvider = ruleClassProvider;
    this.printMessages = printMessages;
  }

  /**
   * Parse the file containing rules blocked from documentation. The list is simply a list of rules
   * separated by new lines. Line comments can be added to the file by starting them with #.
   *
   * @param denyList The name of the file containing the denylist.
   * @return The set of denylisted rules.
   * @throws IOException
   */
  @VisibleForTesting
  public static Set<String> readDenyList(String denyList) throws IOException {
    Set<String> result = new HashSet<String>();
    if (denyList != null && !denyList.isEmpty()) {
      File file = new File(denyList);
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
   * Creates a map of rule names (keys) to rule documentation (values).
   *
   * <p>This method crawls the specified input directories for rule class definitions (as Java
   * source files) which contain the rules' and attributes' definitions as comments in a specific
   * format. The keys in the returned Map correspond to these rule classes.
   *
   * <p>In the Map's values, all references pointing to other rules, rule attributes, and general
   * documentation (e.g. common definitions, make variables, etc.) are expanded into hyperlinks. The
   * links generated follow either the multi-page or single-page Build Encyclopedia model depending
   * on the mode set for the provided {@link RuleLinkExpander}.
   *
   * @param inputDirs list of directories to scan for documentation
   * @param denyList specify an optional denylist file that list some rules that should not be
   *     listed in the output.
   * @param expander The RuleLinkExpander, which is used for expanding links in the rule doc.
   * @throws BuildEncyclopediaDocException
   * @throws IOException
   * @return Map of rule class to rule documentation.
   */
  public Map<String, RuleDocumentation> collect(
      List<String> inputDirs, String denyList, RuleLinkExpander expander)
      throws BuildEncyclopediaDocException, IOException {
    // Read the denyList file
    Set<String> denylistedRules = readDenyList(denyList);
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
      collectDocs(
          processedFiles,
          ruleClassFiles,
          ruleDocEntries,
          denylistedRules,
          attributeDocEntries,
          new File(inputDir));
      if (printMessages) {
        System.out.println(" " + (ruleDocEntries.size() - ruleNum)
            + " rule documentations found.");
      }
    }

    processAttributeDocs(ruleDocEntries.values(), attributeDocEntries);
    expander.addIndex(buildRuleIndex(ruleDocEntries.values()));
    for (RuleDocumentation rule : ruleDocEntries.values()) {
      rule.setRuleLinkExpander(expander);
    }
    return ruleDocEntries;
  }

  /**
   * Creates a map of rule names (keys) to rule documentation (values).
   *
   * <p>This method crawls the specified input directories for rule class definitions (as Java
   * source files) which contain the rules' and attributes' definitions as comments in a specific
   * format. The keys in the returned Map correspond to these rule classes.
   *
   * <p>In the Map's values, all references pointing to other rules, rule attributes, and general
   * documentation (e.g. common definitions, make variables, etc.) are expanded into hyperlinks. The
   * links generated follow the multi-page Build Encyclopedia model (one page per rule class.).
   *
   * @param inputDirs list of directories to scan for documentation
   * @param denyList specify an optional denylist file that list some rules that should not be
   *     listed in the output.
   * @throws BuildEncyclopediaDocException
   * @throws IOException
   * @return Map of rule class to rule documentation.
   */
  public Map<String, RuleDocumentation> collect(List<String> inputDirs, String denyList)
      throws BuildEncyclopediaDocException, IOException {
    RuleLinkExpander expander = new RuleLinkExpander(productName, /* singlePage */ false);
    return collect(inputDirs, denyList, expander);
  }

  /**
   * Generates an index mapping rule name to its normalized rule family name.
   */
  private Map<String, String> buildRuleIndex(Iterable<RuleDocumentation> rules) {
    Map<String, String> index = new HashMap<>();
    for (RuleDocumentation rule : rules) {
      index.put(rule.getRuleName(), RuleFamily.normalize(rule.getRuleFamily()));
    }
    return index;
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
              ruleClassProvider.getRuleClassDefinition(ruleDoc.getRuleName()).getClass();
          for (Attribute attribute : ruleClass.getAttributes()) {
            if (!attribute.isDocumented()) {
              continue;
            }
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
                int level = attributeDoc.getDefinitionClassAncestryLevel(
                    ruleDefinition,
                    ruleClassProvider);
                if (level >= 0 && level < minLevel) {
                  bestAttributeDoc = attributeDoc;
                  minLevel = level;
                }
              }
              if (bestAttributeDoc != null) {
                try {
                  // We have to clone the matching RuleDocumentationAttribute here so that we don't
                  // overwrite the reference to the actual attribute later by another attribute with
                  // the same ancestor but different default values.
                  bestAttributeDoc = (RuleDocumentationAttribute) bestAttributeDoc.clone();
                } catch (CloneNotSupportedException e) {
                  throw new BuildEncyclopediaDocException(
                      bestAttributeDoc.getFileName(),
                      bestAttributeDoc.getStartLineCnt(),
                      "attribute doesn't support clone: " + e.toString());
                }
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
   * Crawls the specified inputPath and collects the raw rule and rule attribute documentation.
   *
   * <p>This method crawls the specified input directory (recursively calling itself for all
   * subdirectories) and reads each Java source file using {@link SourceFileReader} to extract the
   * raw rule and attribute documentation embedded in comments in a specific format. The extracted
   * documentation is then further processed, such as by {@link
   * BuildDocCollector#collect(List<String>, String, RuleLinkExpander), collect}, in order to
   * associate each rule's documentation with its attribute documentation.
   *
   * <p>This method returns the following through its parameters: the set of Java source files
   * processed, a map of rule name to the source file it was extracted from, a map of rule name to
   * the documentation to the rule, and a multimap of attribute name to attribute documentation.
   *
   * @param processedFiles The set of Java source files files that have already been processed in
   *     order to avoid reprocessing the same file.
   * @param ruleClassFiles Map of rule name to the source file it was extracted from.
   * @param ruleDocEntries Map of rule name to rule documentation.
   * @param denyList The set of denylisted rules whose documentation should not be extracted.
   * @param attributeDocEntries Multimap of rule attribute name to attribute documentation.
   * @param inputPath The File representing the file or directory to read.
   * @throws BuildEncyclopediaDocException
   * @throws IOException
   */
  public void collectDocs(
      Set<File> processedFiles,
      Map<String, File> ruleClassFiles,
      Map<String, RuleDocumentation> ruleDocEntries,
      Set<String> denyList,
      ListMultimap<String, RuleDocumentationAttribute> attributeDocEntries,
      File inputPath)
      throws BuildEncyclopediaDocException, IOException {
    if (processedFiles.contains(inputPath)) {
      return;
    }

    if (inputPath.isFile()) {
      if (DocgenConsts.JAVA_SOURCE_FILE_SUFFIX.apply(inputPath.getName())) {
        SourceFileReader sfr = new SourceFileReader(ruleClassProvider, inputPath.getAbsolutePath());
        sfr.readDocsFromComments();
        for (RuleDocumentation d : sfr.getRuleDocEntries()) {
          String ruleName = d.getRuleName();
          if (!denyList.contains(ruleName)) {
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
        collectDocs(
            processedFiles,
            ruleClassFiles,
            ruleDocEntries,
            denyList,
            attributeDocEntries,
            childPath);
      }
    }

    processedFiles.add(inputPath);
  }
}
