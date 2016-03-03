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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.NativeAspectClass.NativeAspectFactory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.ClassObject.SkylarkClassObject;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkCallbackFunction;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.syntax.Type.ConversionException;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.StringUtil;

import java.util.Arrays;
import java.util.Collection;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.concurrent.Immutable;

/**
 * Metadata of a rule attribute. Contains the attribute name and type, and an
 * default value to be used if none is provided in a rule declaration in a BUILD
 * file. Attributes are immutable, and may be shared by more than one rule (for
 * example, <code>foo_binary</code> and <code>foo_library</code> may share many
 * attributes in common).
 */
@Immutable
public final class Attribute implements Comparable<Attribute> {

  public static final Predicate<RuleClass> ANY_RULE = Predicates.alwaysTrue();

  public static final Predicate<RuleClass> NO_RULE = Predicates.alwaysFalse();

  private static final class RuleAspect {
    private final AspectClass aspectFactory;
    private final Function<Rule, AspectParameters> parametersExtractor;

    RuleAspect(AspectClass aspectFactory, Function<Rule, AspectParameters> parametersExtractor) {
      this.aspectFactory = aspectFactory;
      this.parametersExtractor = parametersExtractor;
    }

    AspectClass getAspectFactory() {
      return aspectFactory;
    }
    
    Function<Rule, AspectParameters> getParametersExtractor() {
      return parametersExtractor;
    }
  }

  /**
   * A configuration transition.
   */
  public interface Transition {
    /**
     * Usually, a non-existent entry in the configuration transition table indicates an error.
     * Unfortunately, that means that we need to always build the full table. This method allows a
     * transition to indicate that a non-existent entry indicates a self transition, i.e., that the
     * resulting configuration is the same as the current configuration. This can simplify the code
     * needed to set up the transition table.
     */
    boolean defaultsToSelf();
  }

  /**
   * A configuration split transition; this should be used to transition to multiple configurations
   * simultaneously. Note that the corresponding rule implementations must have special support to
   * handle this.
   */
  // TODO(bazel-team): Serializability constraints?
  public interface SplitTransition<T> extends Transition {
    /**
     * Return the list of {@code BuildOptions} after splitting; empty if not applicable.
     */
    List<T> split(T buildOptions);
  }

  /**
   * Declaration how the configuration should change when following a label or
   * label list attribute.
   */
  @SkylarkModule(name = "ConfigurationTransition", doc =
      "Declares how the configuration should change when following a dependency. "
    + "It can be either <a href=\"globals.html#DATA_CFG\">DATA_CFG</a> or "
    + "<a href=\"globals.html#HOST_CFG\">HOST_CFG</a>.")
  public enum ConfigurationTransition implements Transition {
    /** No transition, i.e., the same configuration as the current. */
    NONE,

    /** Transition to the host configuration. */
    HOST,

    /** Transition to a null configuration (applies to, e.g., input files). */
    NULL,

    /** Transition from the target configuration to the data configuration. */
    // TODO(bazel-team): Move this elsewhere.
    DATA;

    @Override
    public boolean defaultsToSelf() {
      return false;
    }
  }

  private enum PropertyFlag {
    MANDATORY,
    EXECUTABLE,
    UNDOCUMENTED,
    TAGGABLE,

    /**
     * Whether the list attribute is order-independent and can be sorted.
     */
    ORDER_INDEPENDENT,

    /**
     * Whether the allowedRuleClassesForLabels or allowedFileTypesForLabels are
     * set to custom values. If so, and the attribute is called "deps", the
     * legacy deps checking is skipped, and the new stricter checks are used
     * instead. For non-"deps" attributes, this allows skipping the check if it
     * would pass anyway, as the default setting allows any rule classes and
     * file types.
     */
    STRICT_LABEL_CHECKING,

    /**
     * Set for things that would cause the a compile or lint-like action to
     * be executed when the input changes.  Used by compile_one_dependency.
     * Set for attributes like hdrs and srcs on cc_ rules or srcs on java_
     * or py_rules.  Generally not set on data/resource attributes.
     */
    DIRECT_COMPILE_TIME_INPUT,

    /**
     * Whether the value of the list type attribute must not be an empty list.
     */
    NON_EMPTY,

    /**
     * Verifies that the referenced rule produces a single artifact. Note that this check happens
     * on a per label basis, i.e. the check happens separately for every label in a label list.
     */
    SINGLE_ARTIFACT,

    /**
     * Whether we perform silent ruleclass filtering of the dependencies of the label type
     * attribute according to their rule classes. I.e. elements of the list which don't match the
     * allowedRuleClasses predicate or not rules will be filtered out without throwing any errors.
     * This flag is introduced to handle plugins, do not use it in other cases.
     */
    SILENT_RULECLASS_FILTER,

    // TODO(bazel-team): This is a hack introduced because of the bad design of the original rules.
    // Depot cleanup would be too expensive, but don't migrate this to Skylark.
    /**
     * Whether to perform analysis time filetype check on this label-type attribute or not.
     * If the flag is set, we skip the check that applies the allowedFileTypes filter
     * to generated files. Do not use this if avoidable.
     */
    SKIP_ANALYSIS_TIME_FILETYPE_CHECK,

    /**
     * Whether the value of the attribute should come from a given set of values.
     */
    CHECK_ALLOWED_VALUES,

    /**
     * Whether this attribute is opted out of "configurability", i.e. the ability to determine
     * its value based on properties of the build configuration.
     */
    NONCONFIGURABLE,

    /**
     * Whether we should skip dependency validation checks done by
     * {@link com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.PrerequisiteValidator}
     * (for visibility, etc.).
     */
    SKIP_PREREQ_VALIDATOR_CHECKS,

    /**
     * Whether we should check constraints on dependencies under this attribute
     * (see {@link com.google.devtools.build.lib.analysis.constraints.ConstraintSemantics}). If set,
     * the attribute is constraint-enforced even if default enforcement policy would skip it.
     */
    CHECK_CONSTRAINTS,
  }

  // TODO(bazel-team): modify this interface to extend Predicate and have an extra error
  // message function like AllowedValues does
  /**
   * A predicate-like class that determines whether an edge between two rules is valid or not.
   */
  public interface ValidityPredicate {
    /**
     * This method should return null if the edge is valid, or a suitable error message
     * if it is not. Note that warnings are not supported.
     */
    String checkValid(Rule from, Rule to);
  }

  public static final ValidityPredicate ANY_EDGE =
      new ValidityPredicate() {
        @Override
        public String checkValid(Rule from, Rule to) {
          return null;
        }
      };

  /**
   * Using this callback function, rules can set the configuration of their dependencies during the
   * analysis phase.
   */
  public interface Configurator<TConfig, TRule> {
    TConfig apply(TRule fromRule, TConfig fromConfiguration, Attribute attribute, Target toTarget);
  }

  /**
   * A predicate class to check if the value of the attribute comes from a predefined set.
   */
  public static class AllowedValueSet implements PredicateWithMessage<Object> {

    private final Set<Object> allowedValues;

    public <T> AllowedValueSet(T... values) {
      this(Arrays.asList(values));
    }

    public AllowedValueSet(Iterable<?> values) {
      Preconditions.checkNotNull(values);
      Preconditions.checkArgument(!Iterables.isEmpty(values));
      // Do not remove <Object>: workaround for Java 7 type inference.
      allowedValues = ImmutableSet.<Object>copyOf(values);
    }

    @Override
    public boolean apply(Object input) {
      return allowedValues.contains(input);
    }

    @Override
    public String getErrorReason(Object value) {
      return String.format("has to be one of %s instead of '%s'",
          StringUtil.joinEnglishList(allowedValues, "or", "'"), value);
    }

    @VisibleForTesting
    public Collection<Object> getAllowedValues() {
      return allowedValues;
    }
  }

  /**
   * Creates a new attribute builder.
   *
   * @param name attribute name
   * @param type attribute type
   * @return attribute builder
   *
   * @param <TYPE> attribute type class
   */
  public static <TYPE> Attribute.Builder<TYPE> attr(String name, Type<TYPE> type) {
    return new Builder<>(name, type);
  }

  /**
   * A fluent builder for the {@code Attribute} instances.
   *
   * <p>All methods could be called only once per builder. The attribute
   * already undocumented based on its name cannot be marked as undocumented.
   */
  public static class Builder <TYPE> {
    private String name;
    private final Type<TYPE> type;
    private Transition configTransition = ConfigurationTransition.NONE;
    private Predicate<RuleClass> allowedRuleClassesForLabels = Predicates.alwaysTrue();
    private Predicate<RuleClass> allowedRuleClassesForLabelsWarning = Predicates.alwaysFalse();
    private Configurator<?, ?> configurator;
    private FileTypeSet allowedFileTypesForLabels;
    private ValidityPredicate validityPredicate = ANY_EDGE;
    private Object value;
    private boolean valueSet;
    private Predicate<AttributeMap> condition;
    private Set<PropertyFlag> propertyFlags = EnumSet.noneOf(PropertyFlag.class);
    private PredicateWithMessage<Object> allowedValues = null;
    private ImmutableList<ImmutableSet<String>> mandatoryProvidersList =
        ImmutableList.<ImmutableSet<String>>of();
    private Set<RuleAspect> aspects = new LinkedHashSet<>();

    /**
     * Creates an attribute builder with given name and type. This attribute is optional, uses
     * target configuration and has a default value the same as its type default value. This
     * attribute will be marked as undocumented if its name starts with the dollar sign ({@code $})
     * or colon ({@code :}).
     *
     * @param name attribute name
     * @param type attribute type
     */
    public Builder(String name, Type<TYPE> type) {
      this.name = Preconditions.checkNotNull(name);
      this.type = Preconditions.checkNotNull(type);
      if (isImplicit(name) || isLateBound(name)) {
        setPropertyFlag(PropertyFlag.UNDOCUMENTED, "undocumented");
      }
    }

    private Builder<TYPE> setPropertyFlag(PropertyFlag flag, String propertyName) {
      Preconditions.checkState(
          !propertyFlags.contains(flag), "%s flag is already set", propertyName);
      propertyFlags.add(flag);
      return this;
    }

    /**
     * Sets the property flag of the corresponding name if exists, otherwise throws an Exception.
     * Only meant to use from Skylark, do not use from Java.
     */
    public Builder<TYPE> setPropertyFlag(String propertyName) {
      PropertyFlag flag = null;
      try {
        flag = PropertyFlag.valueOf(propertyName);
      } catch (IllegalArgumentException e) {
        throw new IllegalArgumentException("unknown attribute flag " + propertyName);
      }
      setPropertyFlag(flag, propertyName);
      return this;
    }

    /**
     * Makes the built attribute mandatory.
     */
    public Builder<TYPE> mandatory() {
      return setPropertyFlag(PropertyFlag.MANDATORY, "mandatory");
    }

    /**
     * Makes the built attribute non empty, meaning the attribute cannot have an empty list value.
     * Only applicable for list type attributes.
     */
    public Builder<TYPE> nonEmpty() {
      Preconditions.checkNotNull(type.getListElementType(), "attribute '%s' must be a list", name);
      return setPropertyFlag(PropertyFlag.NON_EMPTY, "non_empty");
    }

    /**
     * Makes the built attribute producing a single artifact.
     */
    public Builder<TYPE> singleArtifact() {
      Preconditions.checkState((type == BuildType.LABEL) || (type == BuildType.LABEL_LIST),
          "attribute '%s' must be a label-valued type", name);
      return setPropertyFlag(PropertyFlag.SINGLE_ARTIFACT, "single_artifact");
    }

    /**
     * Forces silent ruleclass filtering on the label type attribute.
     * This flag is introduced to handle plugins, do not use it in other cases.
     */
    public Builder<TYPE> silentRuleClassFilter() {
      Preconditions.checkState((type == BuildType.LABEL) || (type == BuildType.LABEL_LIST),
          "must be a label-valued type");
      return setPropertyFlag(PropertyFlag.SILENT_RULECLASS_FILTER, "silent_ruleclass_filter");
    }

    /**
     * Skip analysis time filetype check. Don't use it if avoidable.
     */
    public Builder<TYPE> skipAnalysisTimeFileTypeCheck() {
      Preconditions.checkState((type == BuildType.LABEL) || (type == BuildType.LABEL_LIST),
          "must be a label-valued type");
      return setPropertyFlag(PropertyFlag.SKIP_ANALYSIS_TIME_FILETYPE_CHECK,
          "skip_analysis_time_filetype_check");
    }

    /**
     * Mark the built attribute as order-independent.
     */
    public Builder<TYPE> orderIndependent() {
      Preconditions.checkNotNull(type.getListElementType(), "attribute '%s' must be a list", name);
      return setPropertyFlag(PropertyFlag.ORDER_INDEPENDENT, "order-independent");
    }

    /**
     * Defines the configuration transition for this attribute. Defaults to
     * {@code NONE}.
     */
    public Builder<TYPE> cfg(Transition configTransition) {
      Preconditions.checkState(this.configTransition == ConfigurationTransition.NONE,
          "the configuration transition is already set");
      this.configTransition = configTransition;
      return this;
    }

    public Builder<TYPE> cfg(Configurator<?, ?> configurator) {
      this.configurator = configurator;
      return this;
    }

    /**
     * Requires the attribute target to be executable; only for label or label
     * list attributes. Defaults to {@code false}.
     */
    public Builder<TYPE> exec() {
      return setPropertyFlag(PropertyFlag.EXECUTABLE, "executable");
    }

    /**
     * Indicates that the attribute (like srcs or hdrs) should be used as an input when calculating
     * compile_one_dependency.
     */
    public Builder<TYPE> direct_compile_time_input() {
      return setPropertyFlag(PropertyFlag.DIRECT_COMPILE_TIME_INPUT,
                             "direct_compile_time_input");
    }

    /**
     * Makes the built attribute undocumented.
     *
     * @param reason explanation why the attribute is undocumented. This is not
     *        used but required for documentation
     */
    public Builder<TYPE> undocumented(String reason) {
      return setPropertyFlag(PropertyFlag.UNDOCUMENTED, "undocumented");
    }

    /**
     * Sets the attribute default value. The type of the default value must
     * match the type parameter. (e.g. list=[], integer=0, string="",
     * label=null). The {@code defaultValue} must be immutable.
     *
     * <p>If defaultValue is of type Label and is a target, that target will
     * become an implicit dependency of the Rule; we will load the target
     * (and its dependencies) if it encounters the Rule and build the target
     * if needs to apply the Rule.
     */
    public Builder<TYPE> value(TYPE defaultValue) {
      Preconditions.checkState(!valueSet, "the default value is already set");
      value = defaultValue;
      valueSet = true;
      return this;
    }

    /**
     * See value(TYPE) above. This method is only meant for Skylark usage.
     */
    public Builder<TYPE> defaultValue(Object defaultValue) throws ConversionException {
      Preconditions.checkState(!valueSet, "the default value is already set");
      value = type.convert(defaultValue, "attribute " + name);
      valueSet = true;
      return this;
    }

    public boolean isValueSet() {
      return valueSet;
    }

    /**
     * Sets the attribute default value to a computed default value - use
     * this when the default value is a function of other attributes of the
     * Rule. The type of the computed default value for a mandatory attribute
     * must match the type parameter: (e.g. list=[], integer=0, string="",
     * label=null). The {@code defaultValue} implementation must be immutable.
     *
     * <p>If computedDefault returns a Label that is a target, that target will
     * become an implicit dependency of this Rule; we will load the target
     * (and its dependencies) if it encounters the Rule and build the target if
     * needs to apply the Rule.
     */
    public Builder<TYPE> value(ComputedDefault defaultValue) {
      Preconditions.checkState(!valueSet, "the default value is already set");
      value = defaultValue;
      valueSet = true;
      return this;
    }

    /**
     * Sets the attribute default value to be late-bound, i.e., it is derived from the build
     * configuration.
     */
    public Builder<TYPE> value(LateBoundDefault<?> defaultValue) {
      Preconditions.checkState(!valueSet, "the default value is already set");
      Preconditions.checkState(name.isEmpty() || isLateBound(name));
      value = defaultValue;
      valueSet = true;
      return this;
    }

    /**
     * Returns true if a late-bound value has been set. Useful only for Skylark.
     */
    public boolean hasLateBoundValue() {
      return value instanceof LateBoundDefault;
    }

    /**
     * Sets a condition predicate. The default value of the attribute only applies if the condition
     * evaluates to true. If the value is explicitly provided, then this condition is ignored.
     *
     * <p>The condition is only evaluated if the attribute is not explicitly set, and after all
     * explicit attributes have been set. It can generally not access default values of other
     * attributes.
     */
    public Builder<TYPE> condition(Predicate<AttributeMap> condition) {
      Preconditions.checkState(this.condition == null, "the condition is already set");
      this.condition = condition;
      return this;
    }

    /**
     * Switches on the capability of an attribute to be published to the rule's
     * tag set.
     */
    public Builder<TYPE> taggable() {
      return setPropertyFlag(PropertyFlag.TAGGABLE, "taggable");
    }

    /**
     * Disables dependency checks done by
     * {@link com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.PrerequisiteValidator}.
     */
    public Builder<TYPE> skipPrereqValidatorCheck() {
      return setPropertyFlag(PropertyFlag.SKIP_PREREQ_VALIDATOR_CHECKS,
          "skip_prereq_validator_checks");
    }

    /**
     * Enforces constraint checking on dependencies under this attribute. Not calling this method
     * does <i>not</i> mean the attribute won't be enforced. This method simply overrides default
     * enforcement policy, so it's useful for special-case attributes that would otherwise be
     * skipped.
     *
     * <p>See {@link com.google.devtools.build.lib.analysis.constraints.ConstraintSemantics#getConstraintCheckedDependencies}
     * for default enforcement policy.
     */
    public Builder<TYPE> checkConstraints() {
      return setPropertyFlag(PropertyFlag.CHECK_CONSTRAINTS, "check_constraints");
    }

    /**
     * If this is a label or label-list attribute, then this sets the allowed
     * rule types for the labels occurring in the attribute. If the attribute
     * contains Labels of any other rule type, then an error is produced during
     * the analysis phase. Defaults to allow any types.
     *
     * <p>This only works on a per-target basis, not on a per-file basis; with
     * other words, it works for 'deps' attributes, but not 'srcs' attributes.
     */
    public Builder<TYPE> allowedRuleClasses(Iterable<String> allowedRuleClasses) {
      return allowedRuleClasses(
          new RuleClass.Builder.RuleClassNamePredicate(allowedRuleClasses));
    }

    /**
     * If this is a label or label-list attribute, then this sets the allowed
     * rule types for the labels occurring in the attribute. If the attribute
     * contains Labels of any other rule type, then an error is produced during
     * the analysis phase. Defaults to allow any types.
     *
     * <p>This only works on a per-target basis, not on a per-file basis; with
     * other words, it works for 'deps' attributes, but not 'srcs' attributes.
     */
    public Builder<TYPE> allowedRuleClasses(Predicate<RuleClass> allowedRuleClasses) {
      Preconditions.checkState((type == BuildType.LABEL) || (type == BuildType.LABEL_LIST),
          "must be a label-valued type");
      propertyFlags.add(PropertyFlag.STRICT_LABEL_CHECKING);
      allowedRuleClassesForLabels = allowedRuleClasses;
      return this;
    }

    /**
     * If this is a label or label-list attribute, then this sets the allowed
     * rule types for the labels occurring in the attribute. If the attribute
     * contains Labels of any other rule type, then an error is produced during
     * the analysis phase. Defaults to allow any types.
     *
     * <p>This only works on a per-target basis, not on a per-file basis; with
     * other words, it works for 'deps' attributes, but not 'srcs' attributes.
     */
    public Builder<TYPE> allowedRuleClasses(String... allowedRuleClasses) {
      return allowedRuleClasses(ImmutableSet.copyOf(allowedRuleClasses));
    }

    /**
     * If this is a label or label-list attribute, then this sets the allowed
     * file types for file labels occurring in the attribute. If the attribute
     * contains labels that correspond to files of any other type, then an error
     * is produced during the analysis phase.
     *
     * <p>This only works on a per-target basis, not on a per-file basis; with
     * other words, it works for 'deps' attributes, but not 'srcs' attributes.
     */
    public Builder<TYPE> allowedFileTypes(FileTypeSet allowedFileTypes) {
      Preconditions.checkState((type == BuildType.LABEL) || (type == BuildType.LABEL_LIST),
          "must be a label-valued type");
      propertyFlags.add(PropertyFlag.STRICT_LABEL_CHECKING);
      allowedFileTypesForLabels = Preconditions.checkNotNull(allowedFileTypes);
      return this;
    }

    /**
     * Allow all files for legacy compatibility. All uses of this method should be audited and then
     * removed. In some cases, it's correct to allow any file, but mostly the set of files should be
     * restricted to a reasonable set.
     */
    public Builder<TYPE> legacyAllowAnyFileType() {
      return allowedFileTypes(FileTypeSet.ANY_FILE);
    }

    /**
     * If this is a label or label-list attribute, then this sets the allowed
     * file types for file labels occurring in the attribute. If the attribute
     * contains labels that correspond to files of any other type, then an error
     * is produced during the analysis phase.
     *
     * <p>This only works on a per-target basis, not on a per-file basis; with
     * other words, it works for 'deps' attributes, but not 'srcs' attributes.
     */
    public Builder<TYPE> allowedFileTypes(FileType... allowedFileTypes) {
      return allowedFileTypes(FileTypeSet.of(allowedFileTypes));
    }

    /**
     * If this is a label or label-list attribute, then this sets the allowed
     * rule types with warning for the labels occurring in the attribute. If the attribute
     * contains Labels of any other rule type (other than this or those set in
     * allowedRuleClasses()), then a warning is produced during
     * the analysis phase. Defaults to deny any types.
     *
     * <p>This only works on a per-target basis, not on a per-file basis; with
     * other words, it works for 'deps' attributes, but not 'srcs' attributes.
     */
    public Builder<TYPE> allowedRuleClassesWithWarning(Collection<String> allowedRuleClasses) {
      return allowedRuleClassesWithWarning(
          new RuleClass.Builder.RuleClassNamePredicate(allowedRuleClasses));
    }

    /**
     * If this is a label or label-list attribute, then this sets the allowed
     * rule types for the labels occurring in the attribute. If the attribute
     * contains Labels of any other rule type (other than this or those set in
     * allowedRuleClasses()), then a warning is produced during
     * the analysis phase. Defaults to deny any types.
     *
     * <p>This only works on a per-target basis, not on a per-file basis; with
     * other words, it works for 'deps' attributes, but not 'srcs' attributes.
     */
    public Builder<TYPE> allowedRuleClassesWithWarning(Predicate<RuleClass> allowedRuleClasses) {
      Preconditions.checkState((type == BuildType.LABEL) || (type == BuildType.LABEL_LIST),
          "must be a label-valued type");
      propertyFlags.add(PropertyFlag.STRICT_LABEL_CHECKING);
      allowedRuleClassesForLabelsWarning = allowedRuleClasses;
      return this;
    }

    /**
     * If this is a label or label-list attribute, then this sets the allowed
     * rule types for the labels occurring in the attribute. If the attribute
     * contains Labels of any other rule type (other than this or those set in
     * allowedRuleClasses()), then a warning is produced during
     * the analysis phase. Defaults to deny any types.
     *
     * <p>This only works on a per-target basis, not on a per-file basis; with
     * other words, it works for 'deps' attributes, but not 'srcs' attributes.
     */
    public Builder<TYPE> allowedRuleClassesWithWarning(String... allowedRuleClasses) {
      return allowedRuleClassesWithWarning(ImmutableSet.copyOf(allowedRuleClasses));
    }

    /**
     * Sets a list of sets of mandatory Skylark providers. Every configured target occurring in
     * this label type attribute has to provide all the providers from one of those sets,
     * otherwise an error is produces during the analysis phase.
     */
    public Builder<TYPE> mandatoryProvidersList(Iterable<? extends Iterable<String>> providersList){
      Preconditions.checkState((type == BuildType.LABEL) || (type == BuildType.LABEL_LIST),
          "must be a label-valued type");
      ImmutableList.Builder<ImmutableSet<String>> listBuilder = ImmutableList.builder();
      for (Iterable<String> providers : providersList) {
        listBuilder.add(ImmutableSet.copyOf(providers));
      }
      this.mandatoryProvidersList = listBuilder.build();
      return this;
    }

    public Builder<TYPE> mandatoryProviders(Iterable<String> providers) {
      if (providers.iterator().hasNext()) {
        mandatoryProvidersList(ImmutableList.of(providers));
      }
      return this;
    }

    /**
     * Asserts that a particular aspect probably needs to be computed for all direct dependencies
     * through this attribute.
     */
    public <T extends NativeAspectFactory> Builder<TYPE> aspect(Class<T> aspect) {
      Function<Rule, AspectParameters> noParameters = new Function<Rule, AspectParameters>() {
        @Override
        public AspectParameters apply(Rule input) {
          return AspectParameters.EMPTY;
        }
      };
      return this.aspect(aspect, noParameters);
    }

    /**
     * Asserts that a particular parameterized aspect probably needs to be computed for all direct
     * dependencies through this attribute.
     *
     * @param evaluator function that extracts aspect parameters from rule.
     */
    public <T extends NativeAspectFactory> Builder<TYPE> aspect(
        Class<T> aspect, Function<Rule, AspectParameters> evaluator) {
      return this.aspect(new NativeAspectClass<T>(aspect), evaluator);
    }

    /**
     * Asserts that a particular parameterized aspect probably needs to be computed for all direct
     * dependencies through this attribute.
     *
     * @param evaluator function that extracts aspect parameters from rule.
     */
    public Builder<TYPE> aspect(AspectClass aspect, Function<Rule, AspectParameters> evaluator) {
      this.aspects.add(new RuleAspect(aspect, evaluator));
      return this;
    }

    /**
     * Asserts that a particular parameterized aspect probably needs to be computed for all direct
     * dependencies through this attribute.
     */
    public Builder<TYPE> aspect(AspectClass aspect) {
      Function<Rule, AspectParameters> noParameters =
          new Function<Rule, AspectParameters>() {
            @Override
            public AspectParameters apply(Rule input) {
              return AspectParameters.EMPTY;
            }
          };
      return this.aspect(aspect, noParameters);
    }

    /**
     * Sets the predicate-like edge validity checker.
     */
    public Builder<TYPE> validityPredicate(ValidityPredicate validityPredicate) {
      propertyFlags.add(PropertyFlag.STRICT_LABEL_CHECKING);
      this.validityPredicate = validityPredicate;
      return this;
    }

    /**
     * The value of the attribute must be one of allowedValues.
     */
    public Builder<TYPE> allowedValues(PredicateWithMessage<Object> allowedValues) {
      this.allowedValues = allowedValues;
      propertyFlags.add(PropertyFlag.CHECK_ALLOWED_VALUES);
      return this;
    }

    /**
     * Makes the built attribute "non-configurable", i.e. its value cannot be influenced by
     * the build configuration. Attributes are "configurable" unless explicitly opted out here.
     *
     * <p>Non-configurability indicates an exceptional state: there exists Blaze logic that needs
     * the attribute's value, has no access to configurations, and can't apply a workaround
     * through an appropriate {@link AbstractAttributeMapper} implementation. Scenarios like
     * this should be as uncommon as possible, so it's important we maintain clear documentation
     * on what causes them and why users consequently can't configure certain attributes.
     *
     * @param reason why this attribute can't be configurable. This isn't used by Blaze - it's
     *    solely a documentation mechanism.
     */
    public Builder<TYPE> nonconfigurable(String reason) {
      Preconditions.checkState(!reason.isEmpty());
      return setPropertyFlag(PropertyFlag.NONCONFIGURABLE, "nonconfigurable");
    }

    /**
     * Creates the attribute. Uses name, type, optionality, configuration type
     * and the default value configured by the builder.
     */
    public Attribute build() {
      return build(this.name);
    }

    /**
     * Creates the attribute. Uses type, optionality, configuration type
     * and the default value configured by the builder. Use the name
     * passed as an argument. This function is used by Skylark where the
     * name is provided only when we build. We don't want to modify the
     * builder, as it is shared in a multithreaded environment.
     */
    public Attribute build(String name) {
      Preconditions.checkState(!name.isEmpty(), "name has not been set");
      // TODO(bazel-team): Set the default to be no file type, then remove this check, and also
      // remove all allowedFileTypes() calls without parameters.
      if ((type == BuildType.LABEL) || (type == BuildType.LABEL_LIST)) {
        if ((name.startsWith("$") || name.startsWith(":")) && allowedFileTypesForLabels == null) {
          allowedFileTypesForLabels = FileTypeSet.ANY_FILE;
        }
        if (allowedFileTypesForLabels == null) {
          throw new IllegalStateException(name);
        }
      } else if ((type == BuildType.OUTPUT) || (type == BuildType.OUTPUT_LIST)) {
        // TODO(bazel-team): Set the default to no file type and make explicit calls instead.
        if (allowedFileTypesForLabels == null) {
          allowedFileTypesForLabels = FileTypeSet.ANY_FILE;
        }
      }
      return new Attribute(
          name,
          type,
          Sets.immutableEnumSet(propertyFlags),
          valueSet ? value : type.getDefaultValue(),
          configTransition,
          configurator,
          allowedRuleClassesForLabels,
          allowedRuleClassesForLabelsWarning,
          allowedFileTypesForLabels,
          validityPredicate,
          condition,
          allowedValues,
          mandatoryProvidersList,
          ImmutableSet.copyOf(aspects));
    }
  }

  /**
   * A computed default is a default value for a Rule attribute that is a
   * function of other attributes of the rule.
   *
   * <p>Attributes whose defaults are computed are first initialized to the default
   * for their type, and then the computed defaults are evaluated after all
   * non-computed defaults have been initialized. There is no defined order
   * among computed defaults, so they must not depend on each other.
   *
   * <p>If a computed default reads the value of another attribute, at least one of
   * the following must be true:
   *
   * <ol>
   *   <li>The other attribute must be declared in the computed default's constructor</li>
   *   <li>The other attribute must be non-configurable ({@link Builder#nonconfigurable}</li>
   * </ol>
   *
   * <p>The reason for enforced declarations is that, since attribute values might be
   * configurable, a computed default that depends on them may itself take multiple
   * values. Since we have no access to a target's configuration at the time these values
   * are computed, we need the ability to probe the default's *complete* dependency space.
   * Declared dependencies allow us to do so sanely. Non-configurable attributes don't have
   * this problem because their value is fixed and known even without configuration information.
   *
   * <p>Implementations of this interface must be immutable.
   */
  public abstract static class ComputedDefault {
    private final List<String> dependencies;
    List<String> dependencies() { return dependencies; }

    /**
     * Create a computed default that can read all non-configurable attribute values and no
     * configurable attribute values.
     */
    public ComputedDefault() {
      dependencies = ImmutableList.of();
    }

    /**
     * Create a computed default that can read all non-configurable attributes values and one
     * explicitly specified configurable attribute value
     */
    public ComputedDefault(String depAttribute) {
      dependencies = ImmutableList.of(depAttribute);
    }

    /**
     * Create a computed default that can read all non-configurable attributes values and two
     * explicitly specified configurable attribute values.
     */
    public ComputedDefault(String depAttribute1, String depAttribute2) {
      dependencies = ImmutableList.of(depAttribute1, depAttribute2);
    }

    public abstract Object getDefault(AttributeMap rule);
  }

  /**
   * Marker interface for late-bound values. Unfortunately, we can't refer to BuildConfiguration
   * right now, since that is in a separate compilation unit.
   *
   * <p>Implementations of this interface must be immutable.
   *
   * <p>Use sparingly - having different values for attributes during loading and analysis can
   * confuse users.
   */
  public interface LateBoundDefault<T> {
    /**
     * Whether to look up the label in the host configuration. This is only here for the host JDK -
     * we usually need to look up labels in the target configuration.
     */
    boolean useHostConfiguration();

    /**
     * Returns the set of required configuration fragments, i.e., fragments that will be accessed by
     * the code.
     */
    Set<Class<?>> getRequiredConfigurationFragments();

    /**
     * The default value for the attribute that is set during the loading phase.
     */
    Object getDefault();

    /**
     * The actual value for the attribute for the analysis phase, which depends on the build
     * configuration. Note that configurations transitions are applied after the late-bound
     * attribute was evaluated.
     *
     * @param rule the rule being evaluated
     * @param attributes interface for retrieving the values of the rule's other attributes
     * @param o the configuration to evaluate with
     */
    Object getDefault(Rule rule, AttributeMap attributes, T o)
        throws EvalException, InterruptedException;
  }

  /**
   * Abstract super class for label-typed {@link LateBoundDefault} implementations that simplifies
   * the client code a little and makes it a bit more type-safe.
   */
  public abstract static class LateBoundLabel<T> implements LateBoundDefault<T> {
    private final Label label;
    private final ImmutableSet<Class<?>> requiredConfigurationFragments;

    public LateBoundLabel() {
      this((Label) null);
    }

    public LateBoundLabel(Class<?>... requiredConfigurationFragments) {
      this((Label) null, requiredConfigurationFragments);
    }

    public LateBoundLabel(Label label) {
      this.label = label;
      this.requiredConfigurationFragments = ImmutableSet.of();
    }

    public LateBoundLabel(Label label, Class<?>... requiredConfigurationFragments) {
      this.label = label;
      this.requiredConfigurationFragments = ImmutableSet.copyOf(requiredConfigurationFragments);
    }

    public LateBoundLabel(String label) {
      this(Label.parseAbsoluteUnchecked(label));
    }

    public LateBoundLabel(String label, Class<?>... requiredConfigurationFragments) {
      this(Label.parseAbsoluteUnchecked(label), requiredConfigurationFragments);
    }

    @Override
    public boolean useHostConfiguration() {
      return false;
    }

    @Override
    public ImmutableSet<Class<?>> getRequiredConfigurationFragments() {
      return requiredConfigurationFragments;
    }

    @Override
    public final Label getDefault() {
      return label;
    }

    @Override
    public abstract Label getDefault(Rule rule, AttributeMap attributes, T configuration);
  }

  /**
   * Abstract super class for label-list-typed {@link LateBoundDefault} implementations that
   * simplifies the client code a little and makes it a bit more type-safe.
   */
  public abstract static class LateBoundLabelList<T> implements LateBoundDefault<T> {
    private final ImmutableList<Label> labels;

    public LateBoundLabelList() {
      this.labels = ImmutableList.of();
    }

    public LateBoundLabelList(List<Label> labels) {
      this.labels = ImmutableList.copyOf(labels);
    }

    @Override
    public boolean useHostConfiguration() {
      return false;
    }

    @Override
    public ImmutableSet<Class<?>> getRequiredConfigurationFragments() {
      return ImmutableSet.of();
    }

    @Override
    public final List<Label> getDefault() {
      return labels;
    }

    @Override
    public abstract List<Label> getDefault(Rule rule, AttributeMap attributes, T configuration);
  }

  /**
   * A class for late bound attributes defined in Skylark.
   */
  public static final class SkylarkLateBound implements LateBoundDefault<Object> {

    private final SkylarkCallbackFunction callback;

    public SkylarkLateBound(SkylarkCallbackFunction callback) {
      this.callback = callback;
    }

    @Override
    public boolean useHostConfiguration() {
      return false;
    }

    @Override
    public ImmutableSet<Class<?>> getRequiredConfigurationFragments() {
      return ImmutableSet.of();
    }

    @Override
    public Object getDefault() {
      return null;
    }

    @Override
    public Object getDefault(Rule rule, AttributeMap attributes, Object o)
        throws EvalException, InterruptedException {
      Map<String, Object> attrValues = new HashMap<>();
      for (Attribute attr : rule.getAttributes()) {
        if (!attr.isLateBound()) {
          Object value = attributes.get(attr.getName(), attr.getType());
          if (value != null) {
            attrValues.put(attr.getName(), value);
          }
        }
      }
      ClassObject attrs = new SkylarkClassObject(attrValues,
          "No such regular (non late-bound) attribute '%s'.");
      return callback.call(attrs, o);
    }
  }

  private final String name;

  private final Type<?> type;

  private final Set<PropertyFlag> propertyFlags;

  // Exactly one of these conditions is true:
  // 1. defaultValue == null.
  // 2. defaultValue instanceof ComputedDefault &&
  //    type.isValid(defaultValue.getDefault())
  // 3. type.isValid(defaultValue).
  // 4. defaultValue instanceof LateBoundDefault &&
  //    type.isValid(defaultValue.getDefault(configuration))
  // (We assume a hypothetical Type.isValid(Object) predicate.)
  private final Object defaultValue;

  private final Transition configTransition;

  private final Configurator<?, ?> configurator;

  /**
   * For label or label-list attributes, this predicate returns which rule
   * classes are allowed for the targets in the attribute.
   */
  private final Predicate<RuleClass> allowedRuleClassesForLabels;

  /**
   * For label or label-list attributes, this predicate returns which rule
   * classes are allowed for the targets in the attribute with warning.
   */
  private final Predicate<RuleClass> allowedRuleClassesForLabelsWarning;

  /**
   * For label or label-list attributes, this predicate returns which file
   * types are allowed for targets in the attribute that happen to be file
   * targets (rather than rules).
   */
  private final FileTypeSet allowedFileTypesForLabels;

  /**
   * This predicate-like object checks
   * if the edge between two rules using this attribute is valid
   * in the dependency graph. Returns null if valid, otherwise an error message.
   */
  private final ValidityPredicate validityPredicate;

  private final Predicate<AttributeMap> condition;

  private final PredicateWithMessage<Object> allowedValues;

  private final ImmutableList<ImmutableSet<String>> mandatoryProvidersList;

  private final ImmutableSet<RuleAspect> aspects;

  /**
   * Constructs a rule attribute with the specified name, type and default
   * value.
   *
   * @param name the name of the attribute
   * @param type the type of the attribute
   * @param defaultValue the default value to use for this attribute if none is
   *        specified in rule declaration in the BUILD file. Must be null, or of
   *        type "type". May be an instance of ComputedDefault, in which case
   *        its getDefault() method must return an instance of "type", or null.
   *        Must be immutable.
   * @param configTransition the configuration transition for this attribute
   *        (which must be of type LABEL, LABEL_LIST, NODEP_LABEL or
   *        NODEP_LABEL_LIST).
   */
  private Attribute(
      String name,
      Type<?> type,
      Set<PropertyFlag> propertyFlags,
      Object defaultValue,
      Transition configTransition,
      Configurator<?, ?> configurator,
      Predicate<RuleClass> allowedRuleClassesForLabels,
      Predicate<RuleClass> allowedRuleClassesForLabelsWarning,
      FileTypeSet allowedFileTypesForLabels,
      ValidityPredicate validityPredicate,
      Predicate<AttributeMap> condition,
      PredicateWithMessage<Object> allowedValues,
      ImmutableList<ImmutableSet<String>> mandatoryProvidersList,
      ImmutableSet<RuleAspect> aspects) {
    Preconditions.checkNotNull(configTransition);
    Preconditions.checkArgument(
        (configTransition == ConfigurationTransition.NONE && configurator == null)
        || type == BuildType.LABEL || type == BuildType.LABEL_LIST
        || type == BuildType.NODEP_LABEL || type == BuildType.NODEP_LABEL_LIST,
        "Configuration transitions can only be specified for label or label list attributes");
    Preconditions.checkArgument(
        isLateBound(name) == (defaultValue instanceof LateBoundDefault),
        "late bound attributes require a default value that is late bound (and vice versa): %s",
        name);
    if (isLateBound(name)) {
      LateBoundDefault<?> lateBoundDefault = (LateBoundDefault<?>) defaultValue;
      Preconditions.checkArgument((configurator == null),
          "a late bound attribute cannot specify a configurator");
      Preconditions.checkArgument(!lateBoundDefault.useHostConfiguration()
          || (configTransition == ConfigurationTransition.HOST),
          "a late bound default value using the host configuration must use the host transition");
    }

    this.name = name;
    this.type = type;
    this.propertyFlags = propertyFlags;
    this.defaultValue = defaultValue;
    this.configTransition = configTransition;
    this.configurator = configurator;
    this.allowedRuleClassesForLabels = allowedRuleClassesForLabels;
    this.allowedRuleClassesForLabelsWarning = allowedRuleClassesForLabelsWarning;
    this.allowedFileTypesForLabels = allowedFileTypesForLabels;
    this.validityPredicate = validityPredicate;
    this.condition = condition;
    this.allowedValues = allowedValues;
    this.mandatoryProvidersList = mandatoryProvidersList;
    this.aspects = aspects;
  }

  /**
   * Returns the name of this attribute.
   */
  public String getName() {
    return name;
  }

  /**
   * Returns the public name of this attribute. This is the name we use in Skylark code
   * and we can use it to display to the end-user.
   * Implicit and late-bound attributes start with '_' (instead of '$' or ':').
   */
  public String getPublicName() {
    String name = getName();
    // latebound and implicit attributes have a one-character prefix we want to drop
    if (isLateBound() || isImplicit()) {
      return "_" + name.substring(1);
    }
    return name;
  }

  /**
   * Returns the logical type of this attribute. (May differ from the actual
   * representation as a value in the build interpreter; for example, an
   * attribute may logically be a list of labels, but be represented as a list
   * of strings.)
   */
  public Type<?> getType() {
    return type;
  }

  private boolean getPropertyFlag(PropertyFlag flag) {
    return propertyFlags.contains(flag);
  }

  /**
   *  Returns true if this parameter is mandatory.
   */
  public boolean isMandatory() {
    return getPropertyFlag(PropertyFlag.MANDATORY);
  }

  /**
   *  Returns true if this list parameter cannot have an empty list as a value.
   */
  public boolean isNonEmpty() {
    return getPropertyFlag(PropertyFlag.NON_EMPTY);
  }

  /**
   *  Returns true if this label parameter must produce a single artifact.
   */
  public boolean isSingleArtifact() {
    return getPropertyFlag(PropertyFlag.SINGLE_ARTIFACT);
  }

  /**
   *  Returns true if this label type parameter is checked by silent ruleclass filtering.
   */
  public boolean isSilentRuleClassFilter() {
    return getPropertyFlag(PropertyFlag.SILENT_RULECLASS_FILTER);
  }

  /**
   *  Returns true if this label type parameter skips the analysis time filetype check.
   */
  public boolean isSkipAnalysisTimeFileTypeCheck() {
    return getPropertyFlag(PropertyFlag.SKIP_ANALYSIS_TIME_FILETYPE_CHECK);
  }

  /**
   *  Returns true if this parameter is order-independent.
   */
  public boolean isOrderIndependent() {
    return getPropertyFlag(PropertyFlag.ORDER_INDEPENDENT);
  }

  /**
   * Returns the configuration transition for this attribute for label or label
   * list attributes. For other attributes it will always return {@code NONE}.
   */
  public Transition getConfigurationTransition() {
    return configTransition;
  }

  /**
   * Returns the configurator instance for this attribute for label or label list attributes.
   * For other attributes it will always return {@code null}.
   */
  public Configurator<?, ?> getConfigurator() {
    return configurator;
  }

  /**
   * Returns whether the target is required to be executable for label or label
   * list attributes. For other attributes it always returns {@code false}.
   */
  public boolean isExecutable() {
    return getPropertyFlag(PropertyFlag.EXECUTABLE);
  }

  /**
   * Returns {@code true} iff the rule is a direct input for an action.
   */
  public boolean isDirectCompileTimeInput() {
    return getPropertyFlag(PropertyFlag.DIRECT_COMPILE_TIME_INPUT);
  }

  /**
   * Returns {@code true} iff this attribute requires documentation.
   */
  public boolean isDocumented() {
    return !getPropertyFlag(PropertyFlag.UNDOCUMENTED);
  }

  /**
   * Returns {@code true} iff this attribute should be published to the rule's
   * tag set. Note that not all Type classes support tag conversion.
   */
  public boolean isTaggable() {
    return getPropertyFlag(PropertyFlag.TAGGABLE);
  }

  public boolean isStrictLabelCheckingEnabled() {
    return getPropertyFlag(PropertyFlag.STRICT_LABEL_CHECKING);
  }

  /**
   * Returns true if the value of this attribute should be a part of a given set.
   */
  public boolean checkAllowedValues() {
    return getPropertyFlag(PropertyFlag.CHECK_ALLOWED_VALUES);
  }

  public boolean performPrereqValidatorCheck() {
    return !getPropertyFlag(PropertyFlag.SKIP_PREREQ_VALIDATOR_CHECKS);
  }

  public boolean checkConstraintsOverride() {
    return getPropertyFlag(PropertyFlag.CHECK_CONSTRAINTS);
  }

  /**
   * Returns true if this attribute's value can be influenced by the build configuration.
   */
  public boolean isConfigurable() {
    return !(type == BuildType.OUTPUT      // Excluded because of Rule#populateExplicitOutputFiles.
        || type == BuildType.OUTPUT_LIST
        || getPropertyFlag(PropertyFlag.NONCONFIGURABLE));
  }

  /**
   * Returns a predicate that evaluates to true for rule classes that are
   * allowed labels in this attribute. If this is not a label or label-list
   * attribute, the returned predicate always evaluates to true.
   */
  public Predicate<RuleClass> getAllowedRuleClassesPredicate() {
    return allowedRuleClassesForLabels;
  }

  /**
   * Returns a predicate that evaluates to true for rule classes that are
   * allowed labels in this attribute with warning. If this is not a label or label-list
   * attribute, the returned predicate always evaluates to true.
   */
  public Predicate<RuleClass> getAllowedRuleClassesWarningPredicate() {
    return allowedRuleClassesForLabelsWarning;
  }

  /**
   * Returns the list of sets of mandatory Skylark providers.
   */
  public ImmutableList<ImmutableSet<String>> getMandatoryProvidersList() {
    return mandatoryProvidersList;
  }

  public FileTypeSet getAllowedFileTypesPredicate() {
    return allowedFileTypesForLabels;
  }

  public ValidityPredicate getValidityPredicate() {
    return validityPredicate;
  }

  public Predicate<AttributeMap> getCondition() {
    return condition == null ? Predicates.<AttributeMap>alwaysTrue() : condition;
  }

  public PredicateWithMessage<Object> getAllowedValues() {
    return allowedValues;
  }

  /**
   * Returns the list of aspects required for dependencies through this attribute.
   */
  public ImmutableList<Aspect> getAspects(Rule rule) {
    ImmutableList.Builder<Aspect> builder = ImmutableList.builder();
    for (RuleAspect aspect : aspects) {
      builder.add(
          new Aspect(aspect.getAspectFactory(), aspect.getParametersExtractor().apply(rule)));
    }
    return builder.build();
  }

  /**
   * Returns the default value of this attribute in the context of the
   * specified Rule.  For attributes with a computed default, i.e. {@code
   * hasComputedDefault()}, {@code rule} must be non-null since the result may
   * depend on the values of its other attributes.
   *
   * <p>The result may be null (although this is not a value in the build
   * language).
   *
   * <p>During population of the rule's attribute dictionary, all non-computed
   * defaults must be set before all computed ones.
   *
   * @param rule the rule to which this attribute belongs; non-null if
   *   {@code hasComputedDefault()}; ignored otherwise.
   */
  public Object getDefaultValue(Rule rule) {
    if (!getCondition().apply(rule == null ? null : NonconfigurableAttributeMapper.of(rule))) {
      return null;
    } else if (defaultValue instanceof LateBoundDefault<?>) {
      return ((LateBoundDefault<?>) defaultValue).getDefault();
    } else {
      return defaultValue;
    }
  }

  /**
   * Returns the default value of this attribute, even if it has a condition, is a computed default,
   * or a late-bound default.
   */
  @VisibleForTesting
  public Object getDefaultValueForTesting() {
    return defaultValue;
  }

  public LateBoundDefault<?> getLateBoundDefault() {
    Preconditions.checkState(isLateBound());
    return (LateBoundDefault<?>) defaultValue;
  }

  /**
   * Returns true iff this attribute has a computed default or a condition.
   *
   * @see #getDefaultValue(Rule)
   */
  boolean hasComputedDefault() {
    return (defaultValue instanceof ComputedDefault) || (condition != null);
  }

  /**
   * Returns if this attribute is an implicit dependency according to the naming policy that
   * designates implicit attributes.
   */
  public boolean isImplicit() {
    return isImplicit(getName());
  }

  /**
   * Returns if an attribute with the given name is an implicit dependency according to the
   * naming policy that designates implicit attributes.
   */
  public static boolean isImplicit(String name) {
    return name.startsWith("$");
  }

  /**
   * Returns if this attribute is late-bound according to the naming policy that designates
   * late-bound attributes.
   */
  public boolean isLateBound() {
    return isLateBound(getName());
  }

  /**
   * Returns if an attribute with the given name is late-bound according to the naming policy
   * that designates late-bound attributes.
   */
  public static boolean isLateBound(String name) {
    return name.startsWith(":");
  }

  @Override
  public String toString() {
    return "Attribute(" + name + ", " + type + ")";
  }

  @Override
  public int compareTo(Attribute other) {
    return name.compareTo(other.name);
  }

  /**
   * Returns a replica builder of this Attribute.
   */
  public Attribute.Builder<?> cloneBuilder() {
    Builder<?> builder = new Builder<>(name, this.type);
    builder.allowedFileTypesForLabels = allowedFileTypesForLabels;
    builder.allowedRuleClassesForLabels = allowedRuleClassesForLabels;
    builder.allowedRuleClassesForLabelsWarning = allowedRuleClassesForLabelsWarning;
    builder.validityPredicate = validityPredicate;
    builder.condition = condition;
    builder.configTransition = configTransition;
    builder.propertyFlags = propertyFlags.isEmpty() ?
        EnumSet.noneOf(PropertyFlag.class) : EnumSet.copyOf(propertyFlags);
    builder.value = defaultValue;
    builder.valueSet = false;
    builder.allowedValues = allowedValues;

    return builder;
  }
}
