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
import java.util.TreeSet;

/**
 * A class to assemble documentation for the Build Encyclopedia. This class uses
 * {@link BuildDocCollector} to extract documentation fragments from rule classes.
 */
public abstract class BuildEncyclopediaProcessor {
  protected static final Predicate<String> RULE_WORTH_DOCUMENTING = new Predicate<String>() {
    @Override
    public boolean apply(String name) {
      return !name.contains("$");
    }
  };

  /** Name of the product to insert into the documentation. */
  protected final String productName;

  /** Rule class provider from which to extract the rule class hierarchy and attributes. */
  protected final ConfiguredRuleClassProvider ruleClassProvider;

  /**
   * Creates the BuildEncyclopediaProcessor instance. The ruleClassProvider parameter is used for
   * rule class hierarchy and attribute checking.
   */
  public BuildEncyclopediaProcessor(
      String productName, ConfiguredRuleClassProvider ruleClassProvider) {
    this.productName = productName;
    this.ruleClassProvider = Preconditions.checkNotNull(ruleClassProvider);
  }

  /**
   * Collects and processes all the rule and attribute documentation in inputDirs and generates the
   * Build Encyclopedia into the outputDir.
   *
   * @param inputDirs list of directory to scan for document in the source code
   * @param outputRootDir output directory where to write the build encyclopedia
   * @param denyList optional path to a file listing rules to not document
   */
  public abstract void generateDocumentation(
      List<String> inputDirs, String outputDir, String denyList)
      throws BuildEncyclopediaDocException, IOException;

  /**
   * POD class for containing lists of rule families separated into language-specific and generic as
   * returned by {@link #assembleRuleFamilies(Iterable<RuleDocumentation>) assembleRuleFamilies}.
   */
  protected static class RuleFamilies {
    public List<RuleFamily> langSpecific;
    public List<RuleFamily> generic;
    public List<RuleFamily> all;

    public RuleFamilies(List<RuleFamily> langSpecific, List<RuleFamily> generic,
        List<RuleFamily> all) {
      this.langSpecific = langSpecific;
      this.generic = generic;
      this.all = all;
    }
  }

  protected RuleFamilies assembleRuleFamilies(Iterable<RuleDocumentation> docEntries)
      throws BuildEncyclopediaDocException, IOException {
    // Separate rule families into language-specific and generic ones.
    Set<String> langSpecificRuleFamilyNames = new TreeSet<>();
    Set<String> genericRuleFamilyNames = new TreeSet<>();
    separateRuleFamilies(docEntries, langSpecificRuleFamilyNames, genericRuleFamilyNames);

    // Create a mapping of rules based on rule type and family.
    Map<String, ListMultimap<RuleType, RuleDocumentation>> ruleMapping = new HashMap<>();
    createRuleMapping(docEntries, ruleMapping);

    // Create a mapping with the summary string for the individual rule families
    Map<String, StringBuilder> familySummary = new HashMap<>();
    createFamilySummary(docEntries, familySummary);

    // Create lists of RuleFamily objects that will be used to generate the documentation.
    // The separate language-specific and general rule families will be used to generate
    // the Overview page while the list containing all rule families will be used to
    // generate all other documentation.
    List<RuleFamily> langSpecificRuleFamilies =
        filterRuleFamilies(ruleMapping, langSpecificRuleFamilyNames, familySummary);
    List<RuleFamily> genericRuleFamilies =
        filterRuleFamilies(ruleMapping, genericRuleFamilyNames, familySummary);
    List<RuleFamily> allRuleFamilies = new ArrayList<>(langSpecificRuleFamilies);
    allRuleFamilies.addAll(genericRuleFamilies);
    return new RuleFamilies(langSpecificRuleFamilies, genericRuleFamilies, allRuleFamilies);
  }

  private List<RuleFamily> filterRuleFamilies(
      Map<String, ListMultimap<RuleType, RuleDocumentation>> ruleMapping,
      Set<String> ruleFamilyNames,
      Map<String, StringBuilder> familySummary) {
    List<RuleFamily> ruleFamilies = new ArrayList<>(ruleFamilyNames.size());
    for (String name : ruleFamilyNames) {
      ListMultimap<RuleType, RuleDocumentation> ruleTypeMap = ruleMapping.get(name);
      ruleFamilies.add(new RuleFamily(ruleTypeMap, name, familySummary.get(name).toString()));
    }
    return ruleFamilies;
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
   * Obtain the summary string for a rule family from whatever member of the family providing it (if
   * any; otherwise use the empty string).
   */
  private void createFamilySummary(
      Iterable<RuleDocumentation> docEntries, Map<String, StringBuilder> familySummary) {
    for (RuleDocumentation ruleDoc : docEntries) {
      RuleClass ruleClass = ruleClassProvider.getRuleClassMap().get(ruleDoc.getRuleName());
      if (ruleClass != null) {
        String ruleFamily = ruleDoc.getRuleFamily();
        familySummary.computeIfAbsent(ruleFamily, (String k) -> new StringBuilder());
        familySummary.get(ruleFamily).append(ruleDoc.getFamilySummary());
      }
    }
  }

  /**
   * Separates all rule families in docEntries into language-specific rules and generic rules.
   */
  private void separateRuleFamilies(Iterable<RuleDocumentation> docEntries,
      Set<String> langSpecific, Set<String> generic)
      throws BuildEncyclopediaDocException {
    for (RuleDocumentation ruleDoc : docEntries) {
      if (ruleDoc.isLanguageSpecific()) {
        if (generic.contains(ruleDoc.getRuleFamily())) {
          throw ruleDoc.createException("The rule is marked as being language-specific, but other "
              + "rules of the same family have already been marked as being not.");
        }
        langSpecific.add(ruleDoc.getRuleFamily());
      } else {
        if (langSpecific.contains(ruleDoc.getRuleFamily())) {
          throw ruleDoc.createException("The rule is marked as being generic, but other rules of "
              + "the same family have already been marked as being language-specific.");
        }
        generic.add(ruleDoc.getRuleFamily());
      }
    }
  }

  /**
   * Helper method for displaying an warning message about undocumented rules.
   *
   * @param rulesWithoutDocumentation Undocumented rules to list in the warning message.
   */
  protected static void warnAboutUndocumentedRules(Iterable<String> rulesWithoutDocumentation) {
      Iterable<String> undocumentedRules = Iterables.filter(rulesWithoutDocumentation,
          RULE_WORTH_DOCUMENTING);
      System.err.printf("WARNING: The following rules are undocumented: [%s]\n",
          Joiner.on(", ").join(Ordering.<String>natural().immutableSortedCopy(undocumentedRules)));
  }

  /**
   * Sets the {@link RuleLinkExpander} for the provided {@link RuleDocumentationAttributes}.
   *
   * <p>This method is used to set the {@link RuleLinkExpander} for common attributes, such as
   * those defined in {@link PredefinedAttributes}, so that rule references in the docs for those
   * attributes can be expanded.
   *
   * @param attributes The map containing the RuleDocumentationAttributes, keyed by attribute name.
   * @param expander The RuleLinkExpander to set in each of the RuleDocumentationAttributes.
   * @return A map of name to RuleDocumentationAttribute with the RuleLinkExpander set for each
   *     attribute.
   */
  protected static Map<String, RuleDocumentationAttribute> expandCommonAttributes(
      Map<String, RuleDocumentationAttribute> attributes, RuleLinkExpander expander) {
    Map<String, RuleDocumentationAttribute> expanded = new HashMap<>(attributes.size());
    for (Map.Entry<String, RuleDocumentationAttribute> entry : attributes.entrySet()) {
      RuleDocumentationAttribute attribute = entry.getValue();
      attribute.setRuleLinkExpander(expander);
      expanded.put(entry.getKey(), attribute);
    }
    return expanded;
  }

  /**
   * Writes the {@link Page} using the provided file name in the specified output directory.
   *
   * @param page The page to write.
   * @param outputDir The output directory to write the file.
   * @param fileName The name of the file to write the page to.
   * @throws IOException
   */
  protected static void writePage(Page page, String outputDir, String fileName) throws IOException {
    page.write(new File(outputDir + "/" + fileName));
  }
}
