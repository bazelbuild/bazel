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
package com.google.devtools.build.lib.testutil;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.AllowedValueSet;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Type.LabelClass;
import com.google.devtools.build.lib.packages.Type.ListType;
import com.google.devtools.build.lib.packages.Types;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * A helper class to generate valid rules with filled attributes if necessary.
 */
public class BuildRuleWithDefaultsBuilder extends BuildRuleBuilder {

  private Set<String> generateFiles;
  private Map<String, BuildRuleBuilder> generateRules;

  public BuildRuleWithDefaultsBuilder(String ruleClass, String ruleName) {
    super(ruleClass, ruleName);
    this.generateFiles = new HashSet<>();
    this.generateRules = new HashMap<>();
  }

  private BuildRuleWithDefaultsBuilder(String ruleClass, String ruleName,
      Map<String, RuleClass> ruleClassMap, Set<String> generateFiles,
      Map<String, BuildRuleBuilder> generateRules) {
    super(ruleClass, ruleName, ruleClassMap);
    this.generateFiles = generateFiles;
    this.generateRules = generateRules;
  }

  /**
   * Creates a dummy file with the given extension in the given package and returns a valid Blaze
   * label referring to the file. Note, the created label depends on the package of the rule.
   */
  private String getDummyFileLabel(String rulePkg, String filePkg, String extension,
      Type<?> attrType) {
    boolean isOutput = attrType.getLabelClass() == LabelClass.OUTPUT;
    String fileName = (isOutput ? "dummy_output" : "dummy_input") + extension;
    generateFiles.add(filePkg + "/" + fileName);
    if (rulePkg.equals(filePkg)) {
      return ":" + fileName;
    } else {
      return filePkg + ":" + fileName;
    }
  }

  private String getDummyRuleLabel(String rulePkg, RuleClass referencedRuleClass) {
    String referencedRuleName = ruleName + "_ref_" + referencedRuleClass.getName()
        .replace("$", "").replace(":", "");
    // The new generated rule should have the same generatedFiles and generatedRules
    // in order to avoid duplications
    BuildRuleWithDefaultsBuilder builder = new BuildRuleWithDefaultsBuilder(
        referencedRuleClass.getName(), referencedRuleName, ruleClassMap, generateFiles,
        generateRules);
    builder.populateAttributes(rulePkg, true);
    generateRules.put(referencedRuleClass.getName(), builder);
    return referencedRuleName;
  }

  public BuildRuleWithDefaultsBuilder populateLabelAttribute(String pkg, Attribute attribute) {
    return populateLabelAttribute(pkg, pkg, attribute);
  }

  /**
   * Populates the label type attribute with generated values. Populates with a file if possible, or
   * generates an appropriate rule. Note, that the rules are always generated in the same package.
   */
  @CanIgnoreReturnValue
  public BuildRuleWithDefaultsBuilder populateLabelAttribute(
      String rulePkg, String filePkg, Attribute attribute) {
    Type<?> attrType = attribute.getType();
    String label = null;
    if (attribute.getAllowedFileTypesPredicate() != FileTypeSet.NO_FILE) {
      // Try to populate with files first
      String extension = "";
      if (attribute.getAllowedFileTypesPredicate() == FileTypeSet.ANY_FILE) {
        extension = ".txt";
      } else if (attribute.getAllowedFileTypesPredicate() != null) {
        FileTypeSet fileTypes = attribute.getAllowedFileTypesPredicate();
        // This argument should always hold, if not that means a Blaze design/implementation error
        Preconditions.checkArgument(
            !fileTypes.getExtensions().isEmpty(),
            "Attribute %s does not have any allowed file types",
            attribute.getName());
        extension = fileTypes.getExtensions().get(0);
      }
      label = getDummyFileLabel(rulePkg, filePkg, extension, attrType);
    } else {
      Predicate<RuleClass> allowedRuleClasses = attribute.getAllowedRuleClassObjectPredicate();
      if (allowedRuleClasses != Predicates.<RuleClass>alwaysFalse()) {
        // See if there is an applicable rule among the already enqueued rules
        BuildRuleBuilder referencedRuleBuilder = getFirstApplicableRule(attribute);
        if (referencedRuleBuilder != null) {
          label = ":" + referencedRuleBuilder.ruleName;
        } else {
          RuleClass referencedRuleClass = getFirstApplicableRuleClass(attribute);
          if (referencedRuleClass != null) {
            // Generate a rule with the appropriate ruleClass and a label for it in
            // the original rule
            label = ":" + getDummyRuleLabel(rulePkg, referencedRuleClass);
          }
        }
      }
    }
    if (label != null) {
      if (attrType instanceof ListType<?>) {
        addMultiValueAttributes(attribute.getName(), label);
      } else {
        setSingleValueAttribute(attribute.getName(), label);
      }
    }
    return this;
  }

  private boolean doesRuleClassMatch(Attribute attribute, RuleClass ruleClass) {
    // The rule class isn't in the allowed list.
    if (!attribute.getAllowedRuleClassObjectPredicate().apply(ruleClass)) {
      return false;
    }

    // Does this rule class have the correct providers?
    if (!attribute.getRequiredProviders().acceptsAny()) {
      // This attribute requires specific providers, so ignore any rule that claims to have every
      // provider.
      if (ruleClass.getAdvertisedProviders().canHaveAnyProvider()) {
        return false;
      }

      if (!attribute.getRequiredProviders().isSatisfiedBy(ruleClass.getAdvertisedProviders())) {
        return false;
      }
    }

    // Default to accept if nothing else prevents.
    return true;
  }

  private BuildRuleBuilder getFirstApplicableRule(Attribute attribute) {
    // There is no direct way to get the set of allowedRuleClasses from the Attribute
    // The Attribute API probably should not be modified for sole testing purposes
    Optional<BuildRuleBuilder> result =
        generateRules.entrySet().stream()
            .filter(entry -> doesRuleClassMatch(attribute, ruleClassMap.get(entry.getKey())))
            .map(Map.Entry::getValue)
            .findFirst();
    return result.orElse(null);
  }

  private RuleClass getFirstApplicableRuleClass(Attribute attribute) {
    Optional<RuleClass> result =
        ruleClassMap.values().stream()
            .filter(ruleClass -> doesRuleClassMatch(attribute, ruleClass))
            .findFirst();
    return result.orElse(null);
  }

  @CanIgnoreReturnValue
  public BuildRuleWithDefaultsBuilder populateStringListAttribute(Attribute attribute) {
    addMultiValueAttributes(attribute.getName(), "x");
    return this;
  }

  @CanIgnoreReturnValue
  public BuildRuleWithDefaultsBuilder populateStringAttribute(Attribute attribute) {
    setSingleValueAttribute(attribute.getName(), "x");
    return this;
  }

  @CanIgnoreReturnValue
  public BuildRuleWithDefaultsBuilder populateBooleanAttribute(Attribute attribute) {
    setSingleValueAttribute(attribute.getName(), "false");
    return this;
  }

  @CanIgnoreReturnValue
  public BuildRuleWithDefaultsBuilder populateIntegerAttribute(Attribute attribute) {
    setSingleValueAttribute(attribute.getName(), 1);
    return this;
  }

  @CanIgnoreReturnValue
  public BuildRuleWithDefaultsBuilder populateAttributes(String rulePkg, boolean heuristics) {
    for (Attribute attribute : ruleClass.getAttributes()) {
      if (attribute.isMandatory()) {
        if (BuildType.isLabelType(attribute.getType())) {
          // TODO(bazel-team): actually an empty list would be fine in the case where
          // attribute instanceof ListType && !attribute.isNonEmpty(), but BuildRuleBuilder
          // doesn't support that, and it makes little sense anyway
          populateLabelAttribute(rulePkg, attribute);
        } else {
          // Non label type attributes
          if (attribute.getAllowedValues() instanceof AllowedValueSet) {
            Collection<Object> allowedValues =
                ((AllowedValueSet) attribute.getAllowedValues()).getAllowedValues();
            setSingleValueAttribute(attribute.getName(), allowedValues.iterator().next());
          } else if (attribute.getType() == Type.STRING) {
            populateStringAttribute(attribute);
          } else if (attribute.getType() == Type.BOOLEAN) {
            populateBooleanAttribute(attribute);
          } else if (attribute.getType() == Type.INTEGER) {
            populateIntegerAttribute(attribute);
          } else if (attribute.getType() == Types.STRING_LIST) {
            populateStringListAttribute(attribute);
          }
        }
        // TODO(bazel-team): populate for other data types
      } else if (heuristics) {
        populateAttributesHeuristics(rulePkg, attribute);
      }
    }
    return this;
  }

  // Heuristics which might help to generate valid rules.
  // This is a bit hackish, but it helps some generated ruleclasses to pass analysis phase.
  private void populateAttributesHeuristics(String rulePkg, Attribute attribute) {
    if (attribute.getName().equals("srcs") && attribute.getType() == BuildType.LABEL_LIST) {
      // If there is a srcs attribute it might be better to populate it even if it's not mandatory
      populateLabelAttribute(rulePkg, attribute);
    } else if (attribute.getName().equals("main_class") && attribute.getType() == Type.STRING) {
      populateStringAttribute(attribute);
    }
  }

  @Override
  public Collection<String> getFilesToGenerate() {
    return generateFiles;
  }

  @Override
  public Collection<BuildRuleBuilder> getRulesToGenerate() {
    return generateRules.values();
  }
}
