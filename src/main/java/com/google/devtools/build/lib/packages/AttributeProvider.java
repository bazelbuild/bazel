// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.Attribute.StarlarkComputedDefaultTemplate;
import com.google.devtools.build.lib.packages.Attribute.StarlarkComputedDefaultTemplate.CannotPrecomputeDefaultsException;
import com.google.devtools.build.lib.packages.RuleFactory.AttributeValues;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.spelling.SpellChecker;

/** Provides access to the attributes of a rule or macro class. */
public class AttributeProvider {
  /**
   * A (unordered) mapping from attribute names to small integers indexing into the {@code
   * attributes} array.
   */
  private final Map<String, Integer> attributeIndex;

  /**
   * All attributes of this rule or macro class (including inherited ones) ordered by attributeIndex
   * value.
   */
  private final ImmutableList<Attribute> attributes;

  /**
   * Names of the non-configurable attributes of this rule or macro class. This is null for macros
   * because it isn't used.
   */
  @Nullable private final ImmutableList<String> nonConfigurableAttributes;

  /* The name of the rule or macro class that owns these attributes. */
  private final String owner;

  private final boolean ignoreLicenses;

  AttributeProvider(
      ImmutableList<Attribute> attributes,
      Map<String, Integer> attributeIndex,
      @Nullable ImmutableList<String> nonConfigurableAttributes,
      String owner,
      boolean ignoreLicenses) {
    this.attributes = attributes;
    this.attributeIndex = attributeIndex;
    this.nonConfigurableAttributes = nonConfigurableAttributes;
    this.owner = owner;
    this.ignoreLicenses = ignoreLicenses;
  }

  @Override
  public String toString() {
    return owner;
  }

  /**
   * If true, no rule of this class ever declares a license regardless of what the rule's or
   * package's <code>licenses</code> attribute says.
   *
   * <p>This is useful for rule types that don't make sense for license checking.
   */
  boolean ignoreLicenses() {
    return ignoreLicenses;
  }

  /**
   * Returns true iff the attribute 'attrName' is defined for this rule or macro class, and has type
   * 'type'.
   */
  public boolean hasAttr(String attrName, Type<?> type) {
    Integer index = getAttributeIndex(attrName);
    return index != null && getAttribute(index).getType() == type;
  }

  /**
   * Returns the index of the specified attribute name. Use of indices allows space-efficient
   * storage of attribute values in rules or macros, since hashtables are not required. (The index
   * mapping is specific to each RuleClass and an attribute may have a different index in the parent
   * RuleClass.)
   *
   * <p>Returns null if the named attribute is not defined for this class of rule or macro.
   */
  public Integer getAttributeIndex(String attrName) {
    return attributeIndex.get(attrName);
  }

  /** Returns the attribute whose index is 'attrIndex'. Fails if attrIndex is not in range. */
  public Attribute getAttribute(int attrIndex) {
    return attributes.get(attrIndex);
  }

  /**
   * Returns the attribute whose name is 'attrName'; fails with NullPointerException if not found.
   */
  public Attribute getAttributeByName(String attrName) {
    Integer attrIndex =
        Preconditions.checkNotNull(
            getAttributeIndex(attrName), "Attribute %s does not exist", attrName);
    return attributes.get(attrIndex);
  }

  /** Returns the attribute whose name is {@code attrName}, or null if not found. */
  @Nullable
  public Attribute getAttributeByNameMaybe(String attrName) {
    Integer i = getAttributeIndex(attrName);
    return i == null ? null : attributes.get(i);
  }

  /** Returns the number of attributes defined for this rule or macro class. */
  public int getAttributeCount() {
    return attributeIndex.size();
  }

  /**
   * Returns an (immutable) list of all Attributes defined for this class of rule or macro, ordered
   * by increasing index.
   */
  public ImmutableList<Attribute> getAttributes() {
    return attributes;
  }

  /**
   * Returns set of non-configurable attribute names defined for this class of rule. null for macros
   * to save memory, since this field is never read for macros.
   */
  @Nullable
  public List<String> getNonConfigurableAttributes() {
    return nonConfigurableAttributes;
  }

  /**
   * Populates the attributes table of the new {@link RuleOrMacroInstance} with the values in the
   * {@code attributeValues} map and with default values provided by this {@link AttributeProvider}
   * and the {@code pkgBuilder}.
   *
   * <p>Errors are reported on {@code eventHandler}.
   */
  <T> void populateRuleAttributeValues(
      RuleOrMacroInstance ruleOrMacroInstance,
      TargetDefinitionContext targetDefinitionContext,
      AttributeValues<T> attributeValues,
      boolean failOnUnknownAttributes,
      boolean isStarlark)
      throws InterruptedException, CannotPrecomputeDefaultsException {

    BitSet definedAttrIndices =
        populateDefinedRuleAttributeValues(
            ruleOrMacroInstance,
            targetDefinitionContext.getLabelConverter(),
            attributeValues,
            failOnUnknownAttributes,
            targetDefinitionContext.getListInterner(),
            targetDefinitionContext.getLocalEventHandler(),
            targetDefinitionContext.simplifyUnconditionalSelectsInRuleAttrs());
    populateDefaultRuleAttributeValues(
        ruleOrMacroInstance, targetDefinitionContext, definedAttrIndices, isStarlark);
    // Now that all attributes are bound to values, collect and store configurable attribute keys.
    populateConfigDependenciesAttribute(ruleOrMacroInstance);
  }

  /**
   * Populates the attributes table of the new {@link RuleOrMacroInstance} with the values in the
   * {@code attributeValues} map.
   *
   * <p>Handles the special cases of the attribute named {@code "name"} and attributes with value
   * {@link Starlark#NONE}.
   *
   * <p>Returns a bitset {@code b} where {@code b.get(i)} is {@code true} if this method set a value
   * for the attribute with index {@code i} in this {@link AttributeProvider}. Errors are reported
   * on {@code eventHandler}.
   */
  private <T> BitSet populateDefinedRuleAttributeValues(
      RuleOrMacroInstance ruleOrMacroInstance,
      LabelConverter labelConverter,
      AttributeValues<T> attributeValues,
      boolean failOnUnknownAttributes,
      Interner<ImmutableList<?>> listInterner,
      EventHandler eventHandler,
      boolean simplifyUnconditionalSelects) {
    BitSet definedAttrIndices = new BitSet();
    for (T attributeAccessor : attributeValues.getAttributeAccessors()) {
      String attributeName = attributeValues.getName(attributeAccessor);
      Object attributeValue = attributeValues.getValue(attributeAccessor);
      // Ignore all None values.
      if (attributeValue == Starlark.NONE && !failOnUnknownAttributes) {
        continue;
      }

      // If the user sets "applicable_liceneses", change it to the correct name.
      // TODO(aiuto): In the time frame of Bazel 9, remove this alternate spelling.
      if (attributeName.equals(RuleClass.APPLICABLE_METADATA_ATTR_ALT)) {
        attributeName = RuleClass.APPLICABLE_METADATA_ATTR;
      }

      // Check that the attribute's name belongs to a valid attribute for this rule or macro class.
      Integer attrIndex = getAttributeIndex(attributeName);
      if (attrIndex == null) {
        ruleOrMacroInstance.reportError(
            String.format(
                "%s: no such attribute '%s' in '%s' %s%s",
                ruleOrMacroInstance.getLabel(),
                attributeName,
                owner,
                ruleOrMacroInstance.isRuleInstance() ? "rule" : "macro",
                SpellChecker.didYouMean(
                    attributeName,
                    ruleOrMacroInstance.getAttributes().stream()
                        .filter(Attribute::isDocumented)
                        .map(Attribute::getName)
                        .collect(ImmutableList.toImmutableList()))),
            eventHandler);
        continue;
      }
      // Ignore all None values (after reporting an error)
      if (attributeValue == Starlark.NONE) {
        continue;
      }

      Attribute attr = getAttribute(attrIndex);

      if (attributeName.equals("licenses") && ignoreLicenses) {
        ruleOrMacroInstance.setAttributeValue(attr, License.NO_LICENSE, /* explicit= */ false);
        definedAttrIndices.set(attrIndex);
        continue;
      }

      // Convert the build-lang value to a native value, if necessary.
      Object nativeAttributeValue;
      if (attributeValues.valuesAreBuildLanguageTyped()) {
        try {
          nativeAttributeValue =
              BuildType.convertFromBuildLangType(
                  ruleOrMacroInstance.getAttributeProvider().toString(),
                  attr,
                  attributeValue,
                  labelConverter,
                  listInterner,
                  simplifyUnconditionalSelects);
        } catch (ConversionException e) {
          ruleOrMacroInstance.reportError(
              String.format("%s: %s", ruleOrMacroInstance.getLabel(), e.getMessage()),
              eventHandler);
          continue;
        }
        // Ignore select({"//conditions:default": None}) values for attr types with null default.
        if (nativeAttributeValue == null) {
          continue;
        }
      } else {
        nativeAttributeValue = attributeValue;
      }

      if (attr.getName().equals("visibility")) {
        @SuppressWarnings("unchecked")
        List<Label> vis = (List<Label>) nativeAttributeValue;
        try {
          nativeAttributeValue = RuleVisibility.validateAndSimplify(vis);
        } catch (EvalException e) {
          ruleOrMacroInstance.reportError(
              ruleOrMacroInstance.getLabel() + " " + e.getMessage(), eventHandler);
        }
      }

      boolean explicit = attributeValues.isExplicitlySpecified(attributeAccessor);
      ruleOrMacroInstance.setAttributeValue(attr, nativeAttributeValue, explicit);
      checkAllowedValues(ruleOrMacroInstance, attr, eventHandler);
      definedAttrIndices.set(attrIndex);
    }
    return definedAttrIndices;
  }

  /**
   * Populates the attributes table of the new {@link RuleOrMacroInstance} with default values
   * provided by this {@link AttributeProvider} and the {@code pkgBuilder}. This will only provide
   * values for attributes that haven't already been populated, using {@code definedAttrIndices} to
   * determine whether an attribute was populated.
   *
   * <p>Errors are reported on {@code eventHandler}.
   */
  private void populateDefaultRuleAttributeValues(
      RuleOrMacroInstance ruleOrMacroInstance,
      TargetDefinitionContext targetDefinitionContext,
      BitSet definedAttrIndices,
      boolean isStarlark)
      throws InterruptedException, CannotPrecomputeDefaultsException {
    // Set defaults; ensure that every mandatory attribute has a value. Use the default if none
    // is specified.
    List<Attribute> attrsWithComputedDefaults = new ArrayList<>();
    int numAttributes = getAttributeCount();
    for (int attrIndex = 0; attrIndex < numAttributes; ++attrIndex) {
      if (definedAttrIndices.get(attrIndex)) {
        continue;
      }
      Attribute attr = getAttribute(attrIndex);
      if (attr.isMandatory()) {
        ruleOrMacroInstance.reportError(
            String.format(
                "%s: missing value for mandatory attribute '%s' in '%s' %s",
                ruleOrMacroInstance.getLabel(),
                attr.getName(),
                owner,
                ruleOrMacroInstance.isRuleInstance() ? "rule" : "macro"),
            targetDefinitionContext.getLocalEventHandler());
      }

      // Macros don't have computed defaults or special logic for licenses or distributions.
      if (ruleOrMacroInstance instanceof Rule ruleInstance) {
        // We must check both the name and the type of each attribute below in case a Starlark rule
        // defines a licenses or distributions attribute of another type.

        if (attr.hasComputedDefault()) {
          // Note that it is necessary to set all non-computed default values before calling
          // Attribute#getDefaultValue for computed default attributes. Computed default attributes
          // may have a condition predicate (i.e. the predicate returned by Attribute#getCondition)
          // that depends on non-computed default attribute values, and that condition predicate is
          // evaluated by the call to Attribute#getDefaultValue.
          attrsWithComputedDefaults.add(attr);

        } else if (attr.isLateBound()) {
          ruleInstance.setAttributeValue(attr, attr.getLateBoundDefault(), /* explicit= */ false);
        } else if (attr.isMaterializing()) {
          ruleInstance.setAttributeValue(attr, attr.getMaterializer(), false);
        } else if (attr.getName().equals(RuleClass.APPLICABLE_METADATA_ATTR)
            && attr.getType() == BuildType.LABEL_LIST) {
          // The check here is preventing against a corner case where the license()/package_info()
          // rule can get itself as applicable_metadata. This breaks the graph because there is now
          // a self-edge.
          //
          // There are two ways that I can see to resolve this. The first, what is shown here,
          // simply prunes the attribute if the source is a new-style license/metadata rule, based
          // on what's been provided publicly. This does create a tight coupling to the
          // implementation, but this is unavoidable since licenses are no longer a first-class type
          // but we want first class behavior in Bazel core.
          //
          // A different approach that would not depend on the implementation of the rule could
          // filter the list of default_applicable_metadata and not include the metadata rule if it
          // matches the name of the current rule. This obviously fixes the self-assignment rule,
          // but the resulting graph is semantically strange. The interpretation of the graph would
          // be that the metadata rule is subject to the metadata of the *other* default metadata,
          // but not itself. That looks very odd, and it's not semantically accurate.
          // As an alternate, if the self-edge is detected, why not simply drop all the
          // default_applicable_metadata attributes and avoid this oddness? That would work and
          // fix the self-edge problem, but for nodes that don't have the self-edge problem, they
          // would get all default_applicable_metadata and now the graph is inconsistent in that
          // license() rules have applicable_metadata while others do not.
          if (ruleInstance.getRuleClassObject().isPackageMetadataRule()) {
            ruleInstance.setAttributeValue(attr, ImmutableList.of(), /* explicit= */ false);
          }

        } else if (attr.getName().equals("licenses") && attr.getType() == BuildType.LICENSE) {
          ruleInstance.setAttributeValue(
              attr,
              ignoreLicenses
                  ? License.NO_LICENSE
                  : targetDefinitionContext.getPartialPackageArgs().license(),
              /* explicit= */ false);

        }
        // Don't store default values, querying materializes them at read time.
      }
    }
    // An instance of the built-in 'test_suite' rule with an undefined or empty 'tests' attribute
    // attribute gets an '$implicit_tests' attribute, whose value is a shared per-package list
    // of all test labels, populated later.
    // TODO(blaze-rules-team): This should be in test_suite's implementation, not
    // here.
    if (owner.equals("test_suite") && !isStarlark) {
      Attribute implicitTests = this.getAttributeByName("$implicit_tests");
      NonconfigurableAttributeMapper attributeMapper =
          NonconfigurableAttributeMapper.of(ruleOrMacroInstance);
      if (implicitTests != null && attributeMapper.get("tests", BuildType.LABEL_LIST).isEmpty()) {
        boolean explicit = true; // so that it appears in query output
        ruleOrMacroInstance.setAttributeValue(
            implicitTests,
            targetDefinitionContext.getTestSuiteImplicitTestsRef(
                attributeMapper.get("tags", Types.STRING_LIST)),
            explicit);
      }
    }
    // Set computed default attribute values now that all other (i.e. non-computed) default values
    // have been set. Macros won't hit this because they don't have attrs with computed defaults.
    for (Attribute attr : attrsWithComputedDefaults) {
      // If Attribute#hasComputedDefault was true above, Attribute#getDefaultValue returns the
      // computed default function object or a Starlark computed default template. Note that we
      // cannot determine the exact value of the computed default function here because it may
      // depend on other attribute values that are configurable (i.e. they came from select({..})
      // expressions in the build language, and they require configuration data from the analysis
      // phase to be resolved). Instead, we're setting the attribute value to a reference to the
      // computed default function, or if #getDefaultValue is a Starlark computed default
      // template, setting the attribute value to a reference to the StarlarkComputedDefault
      // returned from StarlarkComputedDefaultTemplate#computePossibleValues.
      //
      // StarlarkComputedDefaultTemplate#computePossibleValues pre-computes all possible values the
      // function may evaluate to, and records them in a lookup table. By calling it here, with an
      // EventHandler, any errors that might occur during the function's evaluation can
      // be discovered and propagated here.
      Object valueToSet;
      Object defaultValue = attr.getDefaultValue(null);
      if (defaultValue instanceof StarlarkComputedDefaultTemplate template) {
        valueToSet =
            template.computePossibleValues(
                attr, ruleOrMacroInstance, targetDefinitionContext.getLocalEventHandler());
      } else if (defaultValue instanceof ComputedDefault computedDefault) {
        // Compute all possible values to verify that the ComputedDefault is well-defined. This
        // was previously done implicitly as part of visiting all labels to check for null-ness in
        // Rule.checkForNullLabels, but that was changed to skip non-label attributes to improve
        // performance.
        // TODO: b/287492305 - This is technically an illegal call to getPossibleValues as the
        // package has not yet finished loading. Do we even need this still?
        var unused = computedDefault.getPossibleValues(attr.getType(), ruleOrMacroInstance);
        valueToSet = defaultValue;
      } else {
        valueToSet = defaultValue;
      }
      ruleOrMacroInstance.setAttributeValue(attr, valueToSet, /* explicit= */ false);
    }
  }

  /**
   * Collects all labels used as keys for configurable attributes and places them into the special
   * implicit attribute that tracks them.
   */
  private static void populateConfigDependenciesAttribute(RuleOrMacroInstance ruleOrMacroInstance) {
    RawAttributeMapper attributes = RawAttributeMapper.of(ruleOrMacroInstance);
    Attribute configDepsAttribute =
        attributes.getAttributeDefinition(RuleClass.CONFIG_SETTING_DEPS_ATTRIBUTE);
    if (configDepsAttribute == null) {
      return;
    }

    LinkedHashSet<Label> configLabels = new LinkedHashSet<>();
    for (Attribute attr : ruleOrMacroInstance.getAttributeProvider().getAttributes()) {
      BuildType.SelectorList<?> selectorList =
          attributes.getSelectorList(attr.getName(), attr.getType());
      if (selectorList != null) {
        configLabels.addAll(selectorList.getKeyLabels());
      }
    }

    ruleOrMacroInstance.setAttributeValue(
        configDepsAttribute, ImmutableList.copyOf(configLabels), /* explicit= */ false);
  }

  /**
   * Verifies that the {@link RuleOrMacroInstance} has a valid value for the attribute according to
   * its allowed values.
   *
   * <p>If the value for the given attribute on the given {@link RuleOrMacroInstance} is invalid, an
   * error will be recorded in the given EventHandler.
   *
   * <p>If the {@code attribute} is configurable, all of its potential values are evaluated, and
   * errors for each of the invalid values are reported.
   */
  private static void checkAllowedValues(
      RuleOrMacroInstance ruleOrMacroInstance, Attribute attribute, EventHandler eventHandler) {
    if (attribute.checkAllowedValues()) {
      PredicateWithMessage<Object> allowedValues = attribute.getAllowedValues();
      Iterable<?> values =
          AggregatingAttributeMapper.of(ruleOrMacroInstance)
              .visitAttribute(attribute.getName(), attribute.getType());
      for (Object value : values) {
        if (!allowedValues.apply(value)) {
          ruleOrMacroInstance.reportError(
              String.format(
                  "%s: invalid value in '%s' attribute: %s",
                  ruleOrMacroInstance.getLabel(),
                  attribute.getName(),
                  allowedValues.getErrorReason(value)),
              eventHandler);
        }
      }
    }
  }
  
}
