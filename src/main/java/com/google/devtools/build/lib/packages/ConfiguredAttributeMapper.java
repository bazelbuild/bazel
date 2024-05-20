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
package com.google.devtools.build.lib.packages;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.auto.value.AutoOneOf;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.packages.BuildType.Selector;
import com.google.devtools.build.lib.packages.BuildType.SelectorList;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * {@link AttributeMap} implementation that binds a rule's attribute as follows:
 *
 * <ol>
 *   <li>If the attribute is selectable (i.e. its BUILD declaration is of the form "attr = {
 *       config1: "value1", "config2: "value2", ... }", returns the subset of values chosen by the
 *       current configuration in accordance with Bazel's documented policy on configurable
 *       attribute selection.
 *   <li>If the attribute is not selectable (i.e. its value is static), returns that value with no
 *       additional processing.
 * </ol>
 *
 * <p>Example usage:
 *
 * <pre>
 *   Label fooLabel = ConfiguredAttributeMapper.of(ruleConfiguredTarget).get("foo", Type.LABEL);
 * </pre>
 */
public class ConfiguredAttributeMapper extends AbstractAttributeMapper {

  /**
   * "Manual" constructor that requires the caller to pass the set of configurability conditions
   * that trigger this rule's configurable attributes.
   *
   * <p>If you don't know how to do this, you really want to use one of the "do-it-all"
   * constructors.
   */
  public static ConfiguredAttributeMapper of(
      Rule rule,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      String configHash,
      boolean alwaysSucceed) {
    return new ConfiguredAttributeMapper(rule, configConditions, configHash, alwaysSucceed);
  }

  /**
   * "Manual" constructor that requires the caller to pass the set of configurability conditions
   * that trigger this rule's configurable attributes.
   *
   * <p>If you don't know how to do this, you really want to use one of the "do-it-all"
   * constructors.
   */
  public static ConfiguredAttributeMapper of(
      Rule rule,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      BuildConfigurationValue configuration) {
    boolean alwaysSucceed =
        configuration.getOptions().get(CoreOptions.class).debugSelectsAlwaysSucceed;
    return of(rule, configConditions, configuration.shortId(), alwaysSucceed);
  }

  private final ImmutableMap<Label, ConfigMatchingProvider> configConditions;
  private final String configHash;
  private final boolean alwaysSucceed;

  private ConfiguredAttributeMapper(
      Rule rule,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      String configHash,
      boolean alwaysSucceed) {
    super(Preconditions.checkNotNull(rule));
    this.configConditions = configConditions;
    this.configHash = configHash;
    this.alwaysSucceed = alwaysSucceed;
  }

  @Override
  public String describeRule() {
    return String.format("%s (%s)", super.describeRule(), this.configHash.substring(0, 6));
  }

  /**
   * Checks that all attributes can be mapped to their configured values. This is useful for
   * checking that the configuration space in a configured attribute doesn't contain unresolvable
   * contradictions.
   *
   * @throws ValidationException if any attribute's value can't be resolved under this mapper
   */
  public void validateAttributes() throws ValidationException {
    for (String attrName : getAttributeNames()) {
      getAndValidate(attrName, getAttributeType(attrName));
    }
  }

  /** ValidationException indicates an error during attribute validation. */
  public static final class ValidationException extends Exception {
    private ValidationException(String message) {
      super(message);
    }
  }

  /**
   * Variation of {@link #get} that throws an informative exception if the attribute can't be
   * resolved due to intrinsic contradictions in the configuration.
   */
  public <T> T getAndValidate(String attributeName, Type<T> type) throws ValidationException {
    AttributeResolutionResult<T> result = getResolvedAttribute(attributeName, type);
    if (result.getType() == AttributeResolutionResult.AttributeResolutionResultType.FAILURE) {
      throw new ValidationException(result.getFailure());
    }

    return result.getSuccess().orElse(null);
  }

  public <T> AttributeResolutionResult<T> getResolvedAttribute(Attribute attr) {
    @SuppressWarnings("unchecked")
    Type<T> type = (Type<T>) attr.getType();
    return getResolvedAttribute(attr.getName(), type);
  }

  /**
   * Variation of {@link #getAndValidate} that does not throw Exception. Instead, the method returns
   * the AttributeResolutionResult
   */
  public <T> AttributeResolutionResult<T> getResolvedAttribute(String attributeName, Type<T> type) {
    SelectorList<T> selectorList = getSelectorList(attributeName, type);
    if (selectorList == null) {
      // This is a normal attribute.
      return AttributeResolutionResult.ofSuccess(super.get(attributeName, type));
    }

    List<T> resolvedList = new ArrayList<>();
    for (Selector<T> selector : selectorList.getSelectors()) {
      SelectResolutionResult<T> resolvedPath = resolveSelector(attributeName, selector);
      if (resolvedPath.getType() == SelectResolutionResult.SelectResolutionResultType.FAILURE) {
        return AttributeResolutionResult.ofFailure(resolvedPath.getFailure());
      }
      if (!selector.isValueSet(resolvedPath.getSuccess().configKey)) {
        // Use the default. We don't have access to the rule here, so pass null to
        // Attribute.getValue(). This has the result of making attributes with condition
        // predicates ineligible for "None" values. But no user-facing attributes should
        // do that anyway, so that isn't a loss.
        Attribute attr = getAttributeDefinition(attributeName);
        if (attr.isMandatory()) {
          return AttributeResolutionResult.ofFailure(
              String.format(
                  "Mandatory attribute '%s' resolved to 'None' after evaluating 'select'"
                      + " expression",
                  attributeName));
        }
        @SuppressWarnings("unchecked")
        T defaultValue = (T) attr.getDefaultValue(rule);
        resolvedList.add(defaultValue);
      } else {
        resolvedList.add(resolvedPath.getSuccess().value);
      }
    }

    return AttributeResolutionResult.ofSuccess(
        resolvedList.size() == 1 ? resolvedList.get(0) : type.concat(resolvedList));
  }

  /** Representation of the config key and it's value. */
  public static class ConfigKeyAndValue<T> {
    final Label configKey;
    final T value;
    /** If null, this means the default condition (doesn't correspond to a config_setting). */
    @Nullable final ConfigMatchingProvider provider;

    ConfigKeyAndValue(Label key, T value, @Nullable ConfigMatchingProvider provider) {
      this.configKey = key;
      this.value = value;
      this.provider = provider;
    }
  }

  /**
   * AttributeResolutionResult combines all of the individual SelectResolutionResult instances if
   * there are multiple selects for the same attribute. AttributeResolutionResult is the instance of
   * either:
   *
   * <ol>
   *   <li>1. resolved value of an attribute as the result of successful attribute resolution
   *   <li>2. error string as the result of failed attribute resolution
   * </ol>
   */
  @AutoOneOf(AttributeResolutionResult.AttributeResolutionResultType.class)
  public abstract static class AttributeResolutionResult<T> {

    /** result type of attribute resolution */
    public enum AttributeResolutionResultType {
      SUCCESS,
      FAILURE
    }

    public abstract AttributeResolutionResultType getType();

    public abstract Optional<T> getSuccess();

    public abstract String getFailure();

    public static <T> AttributeResolutionResult<T> ofSuccess(@Nullable T value) {
      return AutoOneOf_ConfiguredAttributeMapper_AttributeResolutionResult.success(
          Optional.ofNullable(value));
    }

    public static <T> AttributeResolutionResult<T> ofFailure(String error) {
      return AutoOneOf_ConfiguredAttributeMapper_AttributeResolutionResult.failure(error);
    }
  }

  /**
   * SelectResolutionResult is the instance of either:
   *
   * <ol>
   *   <li>1. ConfigKeyAndValue as the result of successful select resolution
   *   <li>2. error string as the result of failed select resolution
   * </ol>
   */
  @AutoOneOf(SelectResolutionResult.SelectResolutionResultType.class)
  public abstract static class SelectResolutionResult<T> {

    /** result type of select resolution */
    public enum SelectResolutionResultType {
      SUCCESS,
      FAILURE
    }

    public abstract SelectResolutionResultType getType();

    public abstract ConfigKeyAndValue<T> getSuccess();

    public abstract String getFailure();

    public static <T> SelectResolutionResult<T> ofSuccess(ConfigKeyAndValue<T> value) {
      return AutoOneOf_ConfiguredAttributeMapper_SelectResolutionResult.success(value);
    }

    public static <T> SelectResolutionResult<T> ofFailure(String noMatchError) {
      return AutoOneOf_ConfiguredAttributeMapper_SelectResolutionResult.failure(noMatchError);
    }
  }

  private <T> SelectResolutionResult<T> resolveSelector(
      String attributeName, Selector<T> selector) {
    // Use a LinkedHashMap to guarantee a deterministic branch selection when multiple branches
    // matches but they
    // resolve to the same value.
    LinkedHashMap<Label, ConfigKeyAndValue<T>> matchingConditions = new LinkedHashMap<>();
    // Use a LinkedHashSet to guarantee deterministic error message ordering. We use a LinkedHashSet
    // vs. a more general SortedSet because the latter supports insertion-order, which should more
    // closely match how users see select() structures in BUILD files.
    LinkedHashSet<Label> conditionLabels = new LinkedHashSet<>();

    ArrayList<String> errors = new ArrayList<>();
    // Find the matching condition and record its value (checking for duplicates).
    selector.forEach(
        (selectorKey, value) -> {
          if (BuildType.Selector.isDefaultConditionLabel(selectorKey)) {
            return;
          }

          ConfigMatchingProvider curCondition = configConditions.get(selectorKey);
          if (curCondition == null) {
            // This can happen if the rule is in error
            return;
          }
          conditionLabels.add(selectorKey);

          ConfigMatchingProvider.MatchResult matchResult = curCondition.result();

          if (matchResult.getError() != null) {
            // Resolving selects so last chance to actually surface these errors.
            String message = matchResult.getError();
            errors.add("config_setting " + selectorKey + " is unresolvable because: " + message);
            // Defer the throw in order to collect all possible config_setting that are in error.
          } else if (matchResult.equals(ConfigMatchingProvider.MatchResult.MATCH)) {
            // We keep track of all matches which are more precise than any we have found so
            // far. Therefore, we remove any previous matches which are strictly less precise
            // than this one, and only add this one if none of the previous matches are more
            // precise. It is an error if we do not end up with only one most-precise match.
            boolean suppressed = false;
            Iterator<Map.Entry<Label, ConfigKeyAndValue<T>>> it =
                matchingConditions.entrySet().iterator();
            while (it.hasNext()) {
              ConfigMatchingProvider existingMatch = it.next().getValue().provider;
              if (curCondition.refines(existingMatch)) {
                it.remove();
              } else if (existingMatch.refines(curCondition)) {
                suppressed = true;
                break;
              }
            }
            if (!suppressed) {
              matchingConditions.put(
                  selectorKey, new ConfigKeyAndValue<>(selectorKey, value, curCondition));
            }
          }
        });
    if (!errors.isEmpty()) {
      return SelectResolutionResult.ofFailure(
          "Unresolvable config_settings for configurable attribute \""
              + attributeName
              + "\" in "
              + getLabel()
              + ":\n"
              + Joiner.on("\n").join(errors));
    }

    if (matchingConditions.values().stream().map(s -> s.value).distinct().count() > 1) {
      return SelectResolutionResult.ofFailure(
          "Illegal ambiguous match on configurable attribute \""
              + attributeName
              + "\" in "
              + getLabel()
              + ":\n"
              + Joiner.on("\n").join(matchingConditions.keySet())
              + "\nMultiple matches are not allowed unless one is unambiguously "
              + "more specialized or they resolve to the same value. "
              + "See https://bazel.build/reference/be/functions#select.");
    } else if (!matchingConditions.isEmpty()) {
      return SelectResolutionResult.ofSuccess(
          Iterables.getFirst(matchingConditions.values(), null));
    }

    // If nothing matched, choose the default condition.
    if (selector.hasDefault()) {
      return SelectResolutionResult.ofSuccess(
          new ConfigKeyAndValue<T>(Selector.DEFAULT_CONDITION_LABEL, selector.getDefault(), null));
    }

    // If we're in a debugging mode, set a fake default using the empty value for this select's
    // type.
    if (alwaysSucceed) {
      return SelectResolutionResult.ofSuccess(
          new ConfigKeyAndValue<>(
              Selector.DEFAULT_CONDITION_LABEL,
              selector.getOriginalType().getDefaultValue(),
              null));
    }

    return SelectResolutionResult.ofFailure(
        noMatchError(
            attributeName, selector.getNoMatchError(), conditionLabels, getLabel(), configHash));
  }

  /**
   * Constructs a <a href="https://bazel.build/designs/2016/05/23/beautiful-error-messages.html">
   * beautiful error</a> for when no conditions in a configurable attribute match.
   */
  private static String noMatchError(
      String attribute,
      String customNoMatchError,
      LinkedHashSet<Label> conditionLabels,
      Label targetLabel,
      String configHash) {
    String error =
        String.format(
            "configurable attribute \"%s\" in %s doesn't match this configuration",
            attribute, targetLabel);
    if (!customNoMatchError.isEmpty()) {
      error += String.format(": %s\n", customNoMatchError);
    } else {
      error +=
          ". Would a default condition help?\n\n"
              + "Conditions checked:\n "
              + Joiner.on("\n ").join(conditionLabels)
              + "\n\n"
              + "To see a condition's definition, run: bazel query --output=build "
              + "<condition label>.\n";
    }
    // See ConfiguredTargetQueryEnvironment#shortID for the substring rationale.
    String configShortHash = configHash.substring(0, 7);
    error +=
        String.format(
            "\nThis instance of %s has configuration identifier %s. "
                + "To inspect its configuration, run: bazel config %s.\n",
            targetLabel, configShortHash, configShortHash);
    error +=
        "\n"
            + "For more help, see"
            + " https://bazel.build/docs/configurable-attributes#faq-select-choose-condition.\n\n";
    return error;
  }

  @Override
  public <T> T get(String attributeName, Type<T> type) {
    try {
      return getAndValidate(attributeName, type);
    } catch (ValidationException e) {
      // Callers that reach this branch should explicitly validate the attribute through an
      // appropriate call (either {@link #validateAttributes} or {@link #getAndValidate}) and handle
      // the exception directly. This method assumes pre-validated attributes.
      throw new IllegalStateException(
          "lookup failed on attribute " + attributeName + ": " + e.getMessage());
    }
  }

  @Override
  public boolean isAttributeValueExplicitlySpecified(String attributeName) {
    SelectorList<?> selectorList = getSelectorList(attributeName, getAttributeType(attributeName));
    if (selectorList == null) {
      // This is a normal attribute.
      return super.isAttributeValueExplicitlySpecified(attributeName);
    }
    for (Selector<?> selector : selectorList.getSelectors()) {
      SelectResolutionResult<?> resolvedPath = resolveSelector(attributeName, selector);
      if (resolvedPath.getType() == SelectResolutionResult.SelectResolutionResultType.FAILURE) {
        return true;
      }

      if (selector.isValueSet(resolvedPath.getSuccess().configKey)) {
        return true;
      }
    }
    return false; // Every select() in this list chooses a path with value "None".
  }

  /** Returns the labels that appear multiple times in the same attribute value. */
  public Set<Label> checkForDuplicateLabels(Attribute attribute) {
    Type<List<Label>> attrType = BuildType.LABEL_LIST;
    checkArgument(attribute.getType() == attrType, "Not a label list type: %s", attribute);
    String attrName = attribute.getName();
    SelectorList<List<Label>> selectorList = getSelectorList(attrName, attrType);
    // already checked in RuleClass via AggregatingAttributeMapper.checkForDuplicateLabels
    if (selectorList == null || selectorList.getSelectors().size() == 1) {
      return ImmutableSet.of();
    }
    List<Label> labels = get(attrName, attrType);
    return CollectionUtils.duplicatedElementsOf(labels);
  }
}
