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

import static com.google.common.collect.Sets.newEnumSet;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassNamePredicate;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.packages.Type.LabelClass;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.StringUtil;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

/**
 * Metadata of a rule attribute. Contains the attribute name and type, and an default value to be
 * used if none is provided in a rule declaration in a BUILD file. Attributes are immutable, and may
 * be shared by more than one rule (for example, <code>foo_binary</code> and <code>foo_library
 * </code> may share many attributes in common).
 */
@Immutable
public final class Attribute implements Comparable<Attribute> {

  public static final RuleClassNamePredicate ANY_RULE = RuleClassNamePredicate.unspecified();

  private static final RuleClassNamePredicate NO_RULE = RuleClassNamePredicate.only();

  /** Wraps the information necessary to construct an Aspect. */
  @VisibleForSerialization
  abstract static class RuleAspect<C extends AspectClass> {
    protected final C aspectClass;
    protected final Function<Rule, AspectParameters> parametersExtractor;

    private RuleAspect(C aspectClass, Function<Rule, AspectParameters> parametersExtractor) {
      this.aspectClass = aspectClass;
      this.parametersExtractor = parametersExtractor;
    }

    public String getName() {
      return this.aspectClass.getName();
    }

    public ImmutableSet<String> getRequiredParameters() {
      return ImmutableSet.<String>of();
    }

    public abstract Aspect getAspect(Rule rule);

    public C getAspectClass() {
      return aspectClass;
    }
  }

  private static class NativeRuleAspect extends RuleAspect<NativeAspectClass> {
    NativeRuleAspect(
        NativeAspectClass aspectClass, Function<Rule, AspectParameters> parametersExtractor) {
      super(aspectClass, parametersExtractor);
    }

    @Override
    public Aspect getAspect(Rule rule) {
      AspectParameters params = parametersExtractor.apply(rule);
      return params == null ? null : Aspect.forNative(aspectClass, params);
    }
  }

  @VisibleForSerialization
  @AutoCodec
  static class SkylarkRuleAspect extends RuleAspect<SkylarkAspectClass> {
    private final SkylarkDefinedAspect aspect;

    @VisibleForSerialization
    SkylarkRuleAspect(SkylarkDefinedAspect aspect) {
      super(aspect.getAspectClass(), aspect.getDefaultParametersExtractor());
      this.aspect = aspect;
    }

    @Override
    public ImmutableSet<String> getRequiredParameters() {
      return aspect.getParamAttributes();
    }

    @Override
    public Aspect getAspect(Rule rule) {
      AspectParameters parameters = parametersExtractor.apply(rule);
      return Aspect.forSkylark(aspectClass, aspect.getDefinition(parameters), parameters);
    }
  }

  /** A RuleAspect that just wraps a pre-existing Aspect that doesn't vary with the Rule. */
  private static class PredefinedRuleAspect extends RuleAspect<AspectClass> {
    private final Aspect aspect;

    PredefinedRuleAspect(Aspect aspect) {
      super(aspect.getAspectClass(), null);
      this.aspect = aspect;
    }

    @Override
    public Aspect getAspect(Rule rule) {
      return aspect;
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
     * Whether we should check constraints on this attribute even if default enforcement policy
     * would skip it. See
     * {@link com.google.devtools.build.lib.analysis.constraints.ConstraintSemantics} for more on
     * constraints.
     */
    CHECK_CONSTRAINTS_OVERRIDE,

    /**
     * Whether we should skip constraints checking on this attribute even if default enforcement
     * policy would check it.
     */
    SKIP_CONSTRAINTS_OVERRIDE,

    /**
     * Whether we should use output_licenses to check the licences on this attribute.
     */
    OUTPUT_LICENSES,

    /**
     * Has a Starlark-defined configuration transition. Transitions for analysis testing are tracked
     * separately: see {@link #HAS_ANALYSIS_TEST_TRANSITION}.
     */
    HAS_STARLARK_DEFINED_TRANSITION,

    /**
     * Has a Starlark-defined configuration transition designed specifically for rules which
     * run analysis tests.
     */
    HAS_ANALYSIS_TEST_TRANSITION,
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

  @AutoCodec public static final ValidityPredicate ANY_EDGE = (from, to) -> null;

  /** A predicate class to check if the value of the attribute comes from a predefined set. */
  public static class AllowedValueSet implements PredicateWithMessage<Object> {

    private final Set<Object> allowedValues;

    public AllowedValueSet(Object... values) {
      this(Arrays.asList(values));
    }

    public AllowedValueSet(Iterable<?> values) {
      Preconditions.checkNotNull(values);
      Preconditions.checkArgument(!Iterables.isEmpty(values));
      allowedValues = ImmutableSet.copyOf(values);
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

  public ImmutableMap<String, ImmutableSet<String>> getRequiredAspectParameters() {
    ImmutableMap.Builder<String, ImmutableSet<String>> paramBuilder = ImmutableMap.builder();
    for (RuleAspect<?> aspect : aspects) {
      paramBuilder.put(aspect.getName(), aspect.getRequiredParameters());
    }
    return paramBuilder.build();
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

  /** A factory to generate {@link Attribute} instances. */
  public static class ImmutableAttributeFactory {
    private final Type<?> type;
    private final String doc;
    private final TransitionFactory<AttributeTransitionData> transitionFactory;
    private final RuleClassNamePredicate allowedRuleClassesForLabels;
    private final RuleClassNamePredicate allowedRuleClassesForLabelsWarning;
    private final FileTypeSet allowedFileTypesForLabels;
    private final ValidityPredicate validityPredicate;
    private final Object value;
    private final AttributeValueSource valueSource;
    private final boolean valueSet;
    private final Predicate<AttributeMap> condition;
    private final ImmutableSet<PropertyFlag> propertyFlags;
    private final PredicateWithMessage<Object> allowedValues;
    private final RequiredProviders requiredProviders;
    private final ImmutableList<RuleAspect<?>> aspects;

    private ImmutableAttributeFactory(
        Type<?> type,
        String doc,
        ImmutableSet<PropertyFlag> propertyFlags,
        Object value,
        TransitionFactory<AttributeTransitionData> transitionFactory,
        RuleClassNamePredicate allowedRuleClassesForLabels,
        RuleClassNamePredicate allowedRuleClassesForLabelsWarning,
        FileTypeSet allowedFileTypesForLabels,
        ValidityPredicate validityPredicate,
        AttributeValueSource valueSource,
        boolean valueSet,
        Predicate<AttributeMap> condition,
        PredicateWithMessage<Object> allowedValues,
        RequiredProviders requiredProviders,
        ImmutableList<RuleAspect<?>> aspects) {
      this.type = type;
      this.doc = doc;
      this.transitionFactory = transitionFactory;
      this.allowedRuleClassesForLabels = allowedRuleClassesForLabels;
      this.allowedRuleClassesForLabelsWarning = allowedRuleClassesForLabelsWarning;
      this.allowedFileTypesForLabels = allowedFileTypesForLabels;
      this.validityPredicate = validityPredicate;
      this.value = value;
      this.valueSource = valueSource;
      this.valueSet = valueSet;
      this.condition = condition;
      this.propertyFlags = propertyFlags;
      this.allowedValues = allowedValues;
      this.requiredProviders = requiredProviders;
      this.aspects = aspects;
    }

    public AttributeValueSource getValueSource() {
      return valueSource;
    }

    public boolean isValueSet() {
      return valueSet;
    }

    public Attribute build(String name) {
      Preconditions.checkState(!name.isEmpty(), "name has not been set");
      if (valueSource == AttributeValueSource.LATE_BOUND) {
        Preconditions.checkState(isLateBound(name));
        Preconditions.checkState(!transitionFactory.isSplit());
      }
      // TODO(bazel-team): Set the default to be no file type, then remove this check, and also
      // remove all allowedFileTypes() calls without parameters.

      // do not modify this.allowedFileTypesForLabels, instead create a copy.
      FileTypeSet allowedFileTypesForLabels = this.allowedFileTypesForLabels;
      if (type.getLabelClass() == LabelClass.DEPENDENCY) {
        if (isPrivateAttribute(name) && allowedFileTypesForLabels == null) {
          allowedFileTypesForLabels = FileTypeSet.ANY_FILE;
        }
        Preconditions.checkNotNull(
            allowedFileTypesForLabels, "allowedFileTypesForLabels not set for %s", name);
      } else if (type.getLabelClass() == LabelClass.OUTPUT) {
        // TODO(bazel-team): Set the default to no file type and make explicit calls instead.
        if (allowedFileTypesForLabels == null) {
          allowedFileTypesForLabels = FileTypeSet.ANY_FILE;
        }
      }

      return new Attribute(
          name,
          doc,
          type,
          propertyFlags,
          value,
          transitionFactory,
          allowedRuleClassesForLabels,
          allowedRuleClassesForLabelsWarning,
          allowedFileTypesForLabels,
          validityPredicate,
          condition,
          allowedValues,
          requiredProviders,
          aspects);
    }
  }

  /**
   * A fluent builder for the {@code Attribute} instances.
   *
   * <p>All methods could be called only once per builder. The attribute
   * already undocumented based on its name cannot be marked as undocumented.
   */
  public static class Builder <TYPE> {
    private final String name;
    private final Type<TYPE> type;
    private TransitionFactory<AttributeTransitionData> transitionFactory =
        NoTransition.createFactory();
    private RuleClassNamePredicate allowedRuleClassesForLabels = ANY_RULE;
    private RuleClassNamePredicate allowedRuleClassesForLabelsWarning = NO_RULE;
    private FileTypeSet allowedFileTypesForLabels;
    private ValidityPredicate validityPredicate = ANY_EDGE;
    private Object value;
    private String doc;
    private AttributeValueSource valueSource = AttributeValueSource.DIRECT;
    private boolean valueSet;
    private Predicate<AttributeMap> condition;
    private Set<PropertyFlag> propertyFlags = EnumSet.noneOf(PropertyFlag.class);
    private PredicateWithMessage<Object> allowedValues = null;
    private RequiredProviders.Builder requiredProvidersBuilder =
        RequiredProviders.acceptAnyBuilder();
    private HashMap<String, RuleAspect<?>> aspects = new LinkedHashMap<>();

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
      Preconditions.checkState(type.getLabelClass() == LabelClass.DEPENDENCY,
          "attribute '%s' must be a label-valued type", name);
      return setPropertyFlag(PropertyFlag.SINGLE_ARTIFACT, "single_artifact");
    }

    /**
     * Forces silent ruleclass filtering on the label type attribute.
     * This flag is introduced to handle plugins, do not use it in other cases.
     */
    public Builder<TYPE> silentRuleClassFilter() {
      Preconditions.checkState(type.getLabelClass() == LabelClass.DEPENDENCY,
          "must be a label-valued type");
      return setPropertyFlag(PropertyFlag.SILENT_RULECLASS_FILTER, "silent_ruleclass_filter");
    }

    /**
     * Skip analysis time filetype check. Don't use it if avoidable.
     */
    public Builder<TYPE> skipAnalysisTimeFileTypeCheck() {
      Preconditions.checkState(type.getLabelClass() == LabelClass.DEPENDENCY,
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
     * Mark the built attribute as to use output_licenses for license checking.
     */
    public Builder<TYPE> useOutputLicenses() {
      Preconditions.checkState(BuildType.isLabelType(type), "must be a label type");
      return setPropertyFlag(PropertyFlag.OUTPUT_LICENSES, "output_license");
    }

    /**
     * Indicate the attribute uses uses a starlark-defined (non-analysis-test) configuration
     * transition. Transitions for analysis testing are tracked separately: see
     * {@link #hasAnalysisTestTransition()}.
     */
    public Builder<TYPE> hasStarlarkDefinedTransition() {
      return setPropertyFlag(PropertyFlag.HAS_STARLARK_DEFINED_TRANSITION,
          "starlark-defined split transition");
    }

    /**
     * Indicate the attribute uses uses a starlark-defined analysis-test configuration transition.
     * Such a configuration transition may only be applied on rules with {@code analysis_test=true}.
     */
    public Builder<TYPE> hasAnalysisTestTransition() {
      return setPropertyFlag(PropertyFlag.HAS_ANALYSIS_TEST_TRANSITION,
          "analysis-test split transition");
    }

    /** Defines the configuration transition for this attribute. */
    public Builder<TYPE> cfg(TransitionFactory<AttributeTransitionData> transitionFactory) {
      Preconditions.checkNotNull(transitionFactory);
      Preconditions.checkState(
          NoTransition.isInstance(this.transitionFactory),
          "the configuration transition is already set");
      this.transitionFactory = transitionFactory;
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
     * Set the doc string for the attribute.
     *
     * @param doc The doc string for this attribute.
     */
    public Builder<TYPE> setDoc(String doc) {
      this.doc = doc;
      return this;
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
     *
     * <p>The parameter {@code context} is relevant iff the default value is a Label string. In this
     * case, {@code context} must point to the parent Label in order to be able to convert the
     * default value string to a proper Label.
     *
     * @param parameterName The name of the attribute to use in error messages
     */
    public Builder<TYPE> defaultValue(
        Object defaultValue, Object context, @Nullable String parameterName)
        throws ConversionException {
      Preconditions.checkState(!valueSet, "the default value is already set");
      value =
          type.convert(
              defaultValue,
              ((parameterName == null) ? "" : String.format("parameter '%s' of ", parameterName))
                  + String.format("attribute '%s'", name),
              context);
      valueSet = true;
      return this;
    }

    /** See value(TYPE) above. This method is only meant for Skylark usage. */
    public Builder<TYPE> defaultValue(Object defaultValue) throws ConversionException {
      return defaultValue(defaultValue, null, null);
    }

    /**
     * Sets the attribute default value to a computed default value - use this when the default
     * value is a function of other attributes of the Rule. The type of the computed default value
     * for a mandatory attribute must match the type parameter: (e.g. list=[], integer=0, string="",
     * label=null). The {@code defaultValue} implementation must be immutable.
     *
     * <p>If the computed default returns a Label that is a target, that target will become an
     * implicit dependency of this Rule; we will load the target (and its dependencies) if it
     * encounters the Rule and build the target if needs to apply the Rule.
     */
    public Builder<TYPE> value(ComputedDefault defaultValue) {
      Preconditions.checkState(!valueSet, "the default value is already set");
      value = defaultValue;
      valueSource = AttributeValueSource.COMPUTED_DEFAULT;
      valueSet = true;
      return this;
    }

    /**
     * Sets the attribute default value to a Skylark computed default template. Like a native
     * Computed Default, this allows a Skylark-defined Rule Class to specify that the default value
     * of an attribute is a function of other attributes of the Rule.
     *
     * <p>During the loading phase, the computed default template will be specialized for each rule
     * it applies to. Those rules' attribute values will not be references to {@link
     * SkylarkComputedDefaultTemplate}s, but instead will be references to {@link
     * SkylarkComputedDefault}s.
     *
     * <p>If the computed default returns a Label that is a target, that target will become an
     * implicit dependency of this Rule; we will load the target (and its dependencies) if it
     * encounters the Rule and build the target if needs to apply the Rule.
     */
    public Builder<TYPE> value(SkylarkComputedDefaultTemplate skylarkComputedDefaultTemplate) {
      Preconditions.checkState(!valueSet, "the default value is already set");
      value = skylarkComputedDefaultTemplate;
      valueSource = AttributeValueSource.COMPUTED_DEFAULT;
      valueSet = true;
      return this;
    }

    /**
     * Sets the attribute default value to be late-bound, i.e., it is derived from the build
     * configuration and/or the rule's configured attributes.
     */
    public Builder<TYPE> value(LateBoundDefault<?, ? extends TYPE> defaultValue) {
      Preconditions.checkState(!valueSet, "the default value is already set");
      value = defaultValue;
      valueSource = AttributeValueSource.LATE_BOUND;
      valueSet = true;
      return this;
    }

    /**
     * Returns where the value of this attribute comes from. Useful only for Skylark.
     */
    public AttributeValueSource getValueSource() {
      return valueSource;
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
     * Enforces constraint checking on this attribute even if default enforcement policy would skip
     * it. If default policy checks the attribute, this is a no-op.
     *
     * <p>Most attributes are enforced by default, so in the common case this call is unnecessary.
     *
     * <p>See {@link com.google.devtools.build.lib.analysis.constraints.ConstraintSemantics#getConstraintCheckedDependencies}
     * for enforcement policy details.
     */
    public Builder<TYPE> checkConstraints() {
      Verify.verify(!propertyFlags.contains(PropertyFlag.SKIP_CONSTRAINTS_OVERRIDE),
          "constraint checking is already overridden to be skipped");
      return setPropertyFlag(PropertyFlag.CHECK_CONSTRAINTS_OVERRIDE, "check_constraints");
    }

    /**
     * Skips constraint checking on this attribute even if default enforcement policy would check
     * it. If default policy skips the attribute, this is a no-op.
     *
     * <p>See {@link com.google.devtools.build.lib.analysis.constraints.ConstraintSemantics#getConstraintCheckedDependencies}
     * for enforcement policy details.
     */
    public Builder<TYPE> dontCheckConstraints() {
      Verify.verify(!propertyFlags.contains(PropertyFlag.CHECK_CONSTRAINTS_OVERRIDE),
          "constraint checking is already overridden to be checked");
      return setPropertyFlag(PropertyFlag.SKIP_CONSTRAINTS_OVERRIDE, "dont_check_constraints");
    }

    /**
     * If this is a label or label-list attribute, then this sets the allowed rule types for the
     * labels occurring in the attribute.
     *
     * <p>If the attribute contains Labels of any other rule type, then if they're in {@link
     * #allowedRuleClassesForLabelsWarning}, the build continues with a warning. Else if they
     * fulfill {@link #mandatoryNativeProvidersList}, the build continues without error. Else the
     * build fails during analysis.
     *
     * <p>If neither this nor {@link #allowedRuleClassesForLabelsWarning} is set, only rules that
     * fulfill {@link #mandatoryNativeProvidersList} build without error.
     *
     * <p>This only works on a per-target basis, not on a per-file basis; with other words, it works
     * for 'deps' attributes, but not 'srcs' attributes.
     */
    public Builder<TYPE> allowedRuleClasses(Iterable<String> allowedRuleClasses) {
      return allowedRuleClasses(
          RuleClassNamePredicate.only(allowedRuleClasses));
    }

    /**
     * If this is a label or label-list attribute, then this sets the allowed rule types for the
     * labels occurring in the attribute.
     *
     * <p>If the attribute contains Labels of any other rule type, then if they're in {@link
     * #allowedRuleClassesForLabelsWarning}, the build continues with a warning. Else if they
     * fulfill {@link #mandatoryNativeProvidersList}, the build continues without error. Else the
     * build fails during analysis.
     *
     * <p>If neither this nor {@link #allowedRuleClassesForLabelsWarning} is set, only rules that
     * fulfill {@link #mandatoryNativeProvidersList} build without error.
     *
     * <p>This only works on a per-target basis, not on a per-file basis; with other words, it works
     * for 'deps' attributes, but not 'srcs' attributes.
     */
    public Builder<TYPE> allowedRuleClasses(RuleClassNamePredicate allowedRuleClasses) {
      Preconditions.checkState(type.getLabelClass() == LabelClass.DEPENDENCY,
          "must be a label-valued type");
      propertyFlags.add(PropertyFlag.STRICT_LABEL_CHECKING);
      allowedRuleClassesForLabels = allowedRuleClasses;
      return this;
    }

    /**
     * If this is a label or label-list attribute, then this sets the allowed rule types for the
     * labels occurring in the attribute.
     *
     * <p>If the attribute contains Labels of any other rule type, then if they're in {@link
     * #allowedRuleClassesForLabelsWarning}, the build continues with a warning. Else if they
     * fulfill {@link #mandatoryNativeProvidersList}, the build continues without error. Else the
     * build fails during analysis.
     *
     * <p>If neither this nor {@link #allowedRuleClassesForLabelsWarning} is set, only rules that
     * fulfill {@link #mandatoryNativeProvidersList} build without error.
     *
     * <p>This only works on a per-target basis, not on a per-file basis; with other words, it works
     * for 'deps' attributes, but not 'srcs' attributes.
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
      Preconditions.checkState(type.getLabelClass() == LabelClass.DEPENDENCY,
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
     * If this is a label or label-list attribute, then this sets the allowed rule types with
     * warning for the labels occurring in the attribute. This must be a disjoint set from
     * {@link #allowedRuleClasses}.
     *
     * <p>If the attribute contains Labels of any other rule type (other than this or those set in
     * allowedRuleClasses()) and they fulfill {@link #getMandatoryNativeProvidersList()}}, the build
     * continues without error. Else the build fails during analysis.
     *
     * <p>If neither this nor {@link #allowedRuleClassesForLabels} is set, only rules that
     * fulfill {@link #getMandatoryNativeProvidersList()} build without error.
     *
     * <p>This only works on a per-target basis, not on a per-file basis; with other words, it
     * works for 'deps' attributes, but not 'srcs' attributes.
     */
    public Builder<TYPE> allowedRuleClassesWithWarning(Collection<String> allowedRuleClasses) {
      return allowedRuleClassesWithWarning(
          RuleClassNamePredicate.only(allowedRuleClasses));
    }

    /**
     * If this is a label or label-list attribute, then this sets the allowed rule types with
     * warning for the labels occurring in the attribute. This must be a disjoint set from
     * {@link #allowedRuleClasses}.
     *
     * <p>If the attribute contains Labels of any other rule type (other than this or those set in
     * allowedRuleClasses()) and they fulfill {@link #getMandatoryNativeProvidersList()}}, the build
     * continues without error. Else the build fails during analysis.
     *
     * <p>If neither this nor {@link #allowedRuleClassesForLabels} is set, only rules that
     * fulfill {@link #getMandatoryNativeProvidersList()} build without error.
     *
     * <p>This only works on a per-target basis, not on a per-file basis; with other words, it
     * works for 'deps' attributes, but not 'srcs' attributes.
     */
    public Builder<TYPE> allowedRuleClassesWithWarning(RuleClassNamePredicate allowedRuleClasses) {
      Preconditions.checkState(type.getLabelClass() == LabelClass.DEPENDENCY,
          "must be a label-valued type");
      propertyFlags.add(PropertyFlag.STRICT_LABEL_CHECKING);
      allowedRuleClassesForLabelsWarning = allowedRuleClasses;
      return this;
    }

    /**
     * If this is a label or label-list attribute, then this sets the allowed rule types with
     * warning for the labels occurring in the attribute. This must be a disjoint set from {@link
     * #allowedRuleClasses}.
     *
     * <p>If the attribute contains Labels of any other rule type (other than this or those set in
     * allowedRuleClasses()) and they fulfill {@link #getRequiredProviders()}}, the build continues
     * without error. Else the build fails during analysis.
     *
     * <p>If neither this nor {@link #allowedRuleClassesForLabels} is set, only rules that fulfill
     * {@link #getRequiredProviders()} build without error.
     *
     * <p>This only works on a per-target basis, not on a per-file basis; with other words, it works
     * for 'deps' attributes, but not 'srcs' attributes.
     */
    public Builder<TYPE> allowedRuleClassesWithWarning(String... allowedRuleClasses) {
      return allowedRuleClassesWithWarning(ImmutableSet.copyOf(allowedRuleClasses));
    }

    /**
     * Sets a list of lists of mandatory native providers. Every configured target occurring in this
     * label type attribute has to provide all the providers from one of those lists, otherwise an
     * error is produced during the analysis phase.
     */
    public final Builder<TYPE> mandatoryNativeProvidersList(
        Iterable<? extends Iterable<Class<? extends TransitiveInfoProvider>>> providersList) {
      Preconditions.checkState(type.getLabelClass() == LabelClass.DEPENDENCY,
          "must be a label-valued type");

      for (Iterable<Class<? extends TransitiveInfoProvider>> providers : providersList) {
        this.requiredProvidersBuilder.addNativeSet(ImmutableSet.copyOf(providers));
      }
      return this;
    }

    public Builder<TYPE> mandatoryNativeProviders(
        Iterable<Class<? extends TransitiveInfoProvider>> providers) {
      if (providers.iterator().hasNext()) {
        mandatoryNativeProvidersList(ImmutableList.of(providers));
      }
      return this;
    }

    /**
     * Sets a list of sets of mandatory Skylark providers. Every configured target occurring in
     * this label type attribute has to provide all the providers from one of those sets,
     * or be one of {@link #allowedRuleClasses}, otherwise an error is produced during
     * the analysis phase.
     */
    public Builder<TYPE> mandatoryProvidersList(
        Iterable<? extends Iterable<SkylarkProviderIdentifier>> providersList){
      Preconditions.checkState(type.getLabelClass() == LabelClass.DEPENDENCY,
          "must be a label-valued type");
      for (Iterable<SkylarkProviderIdentifier> providers : providersList) {
        this.requiredProvidersBuilder.addSkylarkSet(ImmutableSet.copyOf(providers));
      }
      return this;
    }

    public Builder<TYPE> legacyMandatoryProviders(String... ids) {
      return mandatoryProviders(
          Iterables.transform(
              Arrays.asList(ids),
              s -> {
                Preconditions.checkNotNull(s);
                return SkylarkProviderIdentifier.forLegacy(s);
              }));
    }

    public Builder<TYPE> mandatoryProviders(Iterable<SkylarkProviderIdentifier> providers) {
      if (providers.iterator().hasNext()) {
        mandatoryProvidersList(ImmutableList.of(providers));
      }
      return this;
    }

    public Builder<TYPE> mandatoryProviders(SkylarkProviderIdentifier... providers) {
      mandatoryProviders(Arrays.asList(providers));
      return this;
    }

    /**
     * Asserts that a particular parameterized aspect probably needs to be computed for all direct
     * dependencies through this attribute.
     *
     * @param evaluator function that extracts aspect parameters from rule. If it returns null,
     * then the aspect will not be attached.
     */
    public Builder<TYPE> aspect(
        NativeAspectClass aspect, Function<Rule, AspectParameters> evaluator) {
      NativeRuleAspect nativeRuleAspect = new NativeRuleAspect(aspect, evaluator);
      RuleAspect<?> oldAspect = this.aspects.put(nativeRuleAspect.getName(), nativeRuleAspect);
      if (oldAspect != null) {
        throw new AssertionError(
            String.format("Aspect %s has already been added", oldAspect.getName()));
      }
      return this;
    }

    /**
     * Asserts that a particular parameterized aspect probably needs to be computed for all direct
     * dependencies through this attribute.
     */
    public Builder<TYPE> aspect(NativeAspectClass aspect) {
      return this.aspect(aspect, EMPTY_FUNCTION);
    }

    @AutoCodec @AutoCodec.VisibleForSerialization
    static final Function<Rule, AspectParameters> EMPTY_FUNCTION = input -> AspectParameters.EMPTY;

    public Builder<TYPE> aspect(SkylarkDefinedAspect skylarkAspect, Location location)
        throws EvalException {
      SkylarkRuleAspect skylarkRuleAspect = new SkylarkRuleAspect(skylarkAspect);
      RuleAspect<?> oldAspect = this.aspects.put(skylarkAspect.getName(), skylarkRuleAspect);
      if (oldAspect != null) {
        throw new EvalException(
            location, String.format("aspect %s added more than once", skylarkAspect.getName()));
      }
      return this;
    }

    /**
     * Should only be used for deserialization.
     */
    public Builder<TYPE> aspect(final Aspect aspect) {
      PredefinedRuleAspect predefinedRuleAspect = new PredefinedRuleAspect(aspect);
      RuleAspect<?> oldAspect =
          this.aspects.put(predefinedRuleAspect.getName(), predefinedRuleAspect);
      if (oldAspect != null) {
        throw new AssertionError(
            String.format("Aspect %s has already been added", oldAspect.getName()));
      }
      return this;
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

    /** Returns an {@link ImmutableAttributeFactory} that can be invoked to create attributes. */
    public ImmutableAttributeFactory buildPartial() {
      Preconditions.checkState(
          !allowedRuleClassesForLabels.consideredOverlapping(allowedRuleClassesForLabelsWarning),
          "allowedRuleClasses %s and allowedRuleClassesWithWarning %s "
              + "may not contain the same rule classes",
          allowedRuleClassesForLabels,
          allowedRuleClassesForLabelsWarning);

      return new ImmutableAttributeFactory(
          type,
          doc,
          Sets.immutableEnumSet(propertyFlags),
          valueSet ? value : type.getDefaultValue(),
          transitionFactory,
          allowedRuleClassesForLabels,
          allowedRuleClassesForLabelsWarning,
          allowedFileTypesForLabels,
          validityPredicate,
          valueSource,
          valueSet,
          condition,
          allowedValues,
          requiredProvidersBuilder.build(),
          ImmutableList.copyOf(aspects.values()));
    }

    /**
     * Creates the attribute. Uses name, type, optionality, configuration type and the default value
     * configured by the builder.
     */
    public Attribute build() {
      return build(this.name);
    }

    /**
     * Creates the attribute. Uses type, optionality, configuration type and the default value
     * configured by the builder. Use the name passed as an argument. This function is used by
     * Skylark where the name is provided only when we build. We don't want to modify the builder,
     * as it is shared in a multithreaded environment.
     */
    public Attribute build(String name) {
      return buildPartial().build(name);
    }
  }

  /**
   * A strategy for dealing with too many computations, used when creating lookup tables for {@link
   * ComputedDefault}s.
   *
   * @param <TException> The type of exception this strategy throws if too many computations are
   *     attempted.
   */
  interface ComputationLimiter<TException extends Exception> {
    void onComputationCount(int count) throws TException;
  }

  /**
   * An implementation of {@link ComputationLimiter} that never throws. For use with
   * natively-defined {@link ComputedDefault}s, which are limited in the number of configurable
   * attributes they depend on, not on the number of different combinations of possible inputs.
   */
  private static final ComputationLimiter<RuntimeException> NULL_COMPUTATION_LIMITER =
      new ComputationLimiter<RuntimeException>() {
        @Override
        public void onComputationCount(int count) throws RuntimeException {}
      };

  /** Exception for computed default attributes that depend on too many configurable attributes. */
  private static class TooManyConfigurableAttributesException extends Exception {
    TooManyConfigurableAttributesException(int max) {
      super(
          String.format(
              "Too many configurable attributes to compute all possible values: "
                  + "Found more than %d possible values.",
              max));
    }
  }

  private static class FixedComputationLimiter
      implements ComputationLimiter<TooManyConfigurableAttributesException> {

    /** Upper bound of the number of combinations of values for a computed default attribute. */
    private static final int COMPUTED_DEFAULT_MAX_COMBINATIONS = 64;

    private static final FixedComputationLimiter INSTANCE = new FixedComputationLimiter();

    @Override
    public void onComputationCount(int count) throws TooManyConfigurableAttributesException {
      if (count > COMPUTED_DEFAULT_MAX_COMBINATIONS) {
        throw new TooManyConfigurableAttributesException(COMPUTED_DEFAULT_MAX_COMBINATIONS);
      }
    }
  }

  /**
   * Specifies how values of {@link ComputedDefault} attributes are computed based on the values of
   * other attributes.
   *
   * <p>The {@code TComputeException} type parameter allows the two specializations of this class to
   * describe whether and how their computations throw. For natively defined computed defaults,
   * computation does not throw, but for Skylark-defined computed defaults, computation may throw
   * {@link InterruptedException}.
   */
  private abstract static class ComputationStrategy<TComputeException extends Exception> {
    abstract Object compute(AttributeMap map) throws TComputeException;

    /**
     * Returns a lookup table mapping from:
     *
     * <ul>
     * <li>tuples of values that may be assigned by {@code rule} to attributes with names in {@code
     *     dependencies} (note that there may be more than one such tuple for any given rule, if any
     *     of the dependencies are configurable)
     * </ul>
     *
     * <p>to:
     *
     * <ul>
     * <li>the value {@link #compute(AttributeMap)} evaluates to when the provided {@link
     *     AttributeMap} contains the values specified by that assignment, or {@code null} if the
     *     {@link ComputationStrategy} failed to evaluate.
     * </ul>
     *
     * <p>The lookup table contains a tuple for each possible assignment to the {@code dependencies}
     * attributes. The meaning of each tuple is well-defined because {@code dependencies} is
     * ordered.
     *
     * <p>This is useful because configurable attributes may have many possible values. During the
     * loading phase a configurable attribute can't be resolved to a single value. Configuration
     * information, needed to resolve such an attribute, is only available during analysis. However,
     * any labels that a ComputedDefault attribute may evaluate to must be loaded during the loading
     * phase.
     */
    <T, TLimitException extends Exception> Map<List<Object>, T> computeValuesForAllCombinations(
        List<String> dependencies,
        Type<T> type,
        Rule rule,
        ComputationLimiter<TLimitException> limiter)
        throws TComputeException, TLimitException {
      AggregatingAttributeMapper mapper = AggregatingAttributeMapper.of(rule);
      // This will hold every (value1, value2, ..) combination of the declared dependencies.
      // Collect those combinations.
      List<Map<String, Object>> depMaps = mapper.visitAttributes(dependencies, limiter);
      // For each combination, call compute() on a specialized AttributeMap providing those
      // values.
      Map<List<Object>, T> valueMap = Maps.newHashMapWithExpectedSize(depMaps.size());
      for (Map<String, Object> depMap : depMaps) {
        AttributeMap attrMap = mapper.createMapBackedAttributeMap(depMap);
        Object value = compute(attrMap);
        List<Object> key = createDependencyAssignmentTuple(dependencies, attrMap);
        valueMap.put(key, type.cast(value));
      }
      return valueMap;
    }

    /**
     * Given an {@link AttributeMap}, containing an assignment to each attribute in {@code
     * dependencies}, this returns a list of the assigned values, ordered as {@code dependencies} is
     * ordered.
     */
    static List<Object> createDependencyAssignmentTuple(
        List<String> dependencies, AttributeMap attrMap) {
      ArrayList<Object> tuple = new ArrayList<>(dependencies.size());
      for (String attrName : dependencies) {
        Type<?> attrType = attrMap.getAttributeType(attrName);
        tuple.add(attrMap.get(attrName, attrType));
      }
      return tuple;
    }
  }

  /**
   * A computed default is a default value for a Rule attribute that is a function of other
   * attributes of the rule.
   *
   * <p>Attributes whose defaults are computed are first initialized to the default for their type,
   * and then the computed defaults are evaluated after all non-computed defaults have been
   * initialized. There is no defined order among computed defaults, so they must not depend on each
   * other.
   *
   * <p>If a computed default reads the value of another attribute, at least one of the following
   * must be true:
   *
   * <ol>
   *   <li>The other attribute must be declared in the computed default's constructor
   *   <li>The other attribute must be non-configurable ({@link Builder#nonconfigurable}
   * </ol>
   *
   * <p>The reason for enforced declarations is that, since attribute values might be configurable,
   * a computed default that depends on them may itself take multiple values. Since we have no
   * access to a target's configuration at the time these values are computed, we need the ability
   * to probe the default's *complete* dependency space. Declared dependencies allow us to do so
   * sanely. Non-configurable attributes don't have this problem because their value is fixed and
   * known even without configuration information.
   *
   * <p>Implementations of this interface must be immutable.
   */
  public abstract static class ComputedDefault implements StarlarkValue {
    private final ImmutableList<String> dependencies;

    /**
     * Create a computed default that can read all non-configurable attribute values and no
     * configurable attribute values.
     */
    public ComputedDefault() {
      this(ImmutableList.<String>of());
    }

    /**
     * Create a computed default that can read all non-configurable attributes values and one
     * explicitly specified configurable attribute value
     */
    public ComputedDefault(String depAttribute) {
      this(ImmutableList.of(depAttribute));
    }

    /**
     * Create a computed default that can read all non-configurable attributes values and two
     * explicitly specified configurable attribute values.
     */
    public ComputedDefault(String depAttribute1, String depAttribute2) {
      this(ImmutableList.of(depAttribute1, depAttribute2));
    }

    /**
     * Creates a computed default that can read all non-configurable attributes and some explicitly
     * specified configurable attribute values.
     *
     * <p>This constructor should not be used by native {@link ComputedDefault} functions. The limit
     * of at-most-two depended-on configurable attributes is intended, to limit the exponential
     * growth of possible values. {@link SkylarkComputedDefault} uses this, but is limited by {@link
     * FixedComputationLimiter#COMPUTED_DEFAULT_MAX_COMBINATIONS}.
     */
    protected ComputedDefault(ImmutableList<String> dependencies) {
      // Order is important for #createDependencyAssignmentTuple.
      this.dependencies = Ordering.natural().immutableSortedCopy(dependencies);
    }

    <T> Iterable<T> getPossibleValues(Type<T> type, Rule rule) {
      final ComputedDefault owner = ComputedDefault.this;
      if (dependencies.isEmpty()) {
        AggregatingAttributeMapper mapper = AggregatingAttributeMapper.of(rule);
        Object value = owner.getDefault(mapper.createMapBackedAttributeMap(ImmutableMap.of()));
        return Lists.newArrayList(type.cast(value));
      }
      ComputationStrategy<RuntimeException> strategy =
          new ComputationStrategy<RuntimeException>() {
            @Override
            public Object compute(AttributeMap map) {
              return owner.getDefault(map);
            }
          };
      // Note that this uses ArrayList instead of something like ImmutableList because some
      // values may be null.
      return new ArrayList<>(
          strategy
              .computeValuesForAllCombinations(dependencies, type, rule, NULL_COMPUTATION_LIMITER)
              .values());
    }

    /** The list of configurable attributes this ComputedDefault declares it may read. */
    public ImmutableList<String> dependencies() {
      return dependencies;
    }

    /**
     * Returns the value this {@link ComputedDefault} evaluates to, given the inputs contained in
     * {@code rule}.
     */
    public abstract Object getDefault(AttributeMap rule);
  }

  /**
   * A Skylark-defined computed default, which can be precomputed for a specific {@link Rule} by
   * calling {@link #computePossibleValues}, which returns a {@link SkylarkComputedDefault} that
   * contains a lookup table.
   */
  public static final class SkylarkComputedDefaultTemplate {
    private final Type<?> type;
    private final StarlarkCallbackHelper callback;
    private final Location location;
    private final ImmutableList<String> dependencies;

    /**
     * Creates a new SkylarkComputedDefaultTemplate that allows the computation of attribute values
     * via a callback function during loading phase.
     *
     * @param type The type of the value of this attribute.
     * @param dependencies A list of all names of other attributes that are accessed by this
     *     attribute.
     * @param callback A function to compute the actual attribute value.
     * @param location The location of the Skylark function.
     */
    public SkylarkComputedDefaultTemplate(
        Type<?> type,
        ImmutableList<String> dependencies,
        StarlarkCallbackHelper callback,
        Location location) {
      this.type = Preconditions.checkNotNull(type);
      // Order is important for #createDependencyAssignmentTuple.
      this.dependencies =
          Ordering.natural().immutableSortedCopy(Preconditions.checkNotNull(dependencies));
      this.callback = Preconditions.checkNotNull(callback);
      this.location = Preconditions.checkNotNull(location);
    }

    /**
     * Returns a {@link SkylarkComputedDefault} containing a lookup table specifying the output of
     * this {@link SkylarkComputedDefaultTemplate}'s callback given each possible assignment {@code
     * rule} might make to the attributes specified by {@link #dependencies}.
     *
     * <p>If the rule is missing an attribute specified by {@link #dependencies}, or if there are
     * too many possible assignments, or if any evaluation fails, this throws {@link
     * CannotPrecomputeDefaultsException}.
     *
     * <p>May only be called after all non-{@link ComputedDefault} attributes have been set on the
     * {@code rule}.
     */
    SkylarkComputedDefault computePossibleValues(
        Attribute attr, final Rule rule, final EventHandler eventHandler)
        throws InterruptedException, CannotPrecomputeDefaultsException {

      final SkylarkComputedDefaultTemplate owner = SkylarkComputedDefaultTemplate.this;
      final String msg =
          String.format(
              "Cannot compute default value of attribute '%s' in rule '%s': ",
              attr.getPublicName(), rule.getLabel());
      final AtomicReference<EvalException> caughtEvalExceptionIfAny = new AtomicReference<>();
      ComputationStrategy<InterruptedException> strategy =
          new ComputationStrategy<InterruptedException>() {
            @Override
            public Object compute(AttributeMap map) throws InterruptedException {
              try {
                return owner.computeValue(eventHandler, map);
              } catch (EvalException ex) {
                caughtEvalExceptionIfAny.compareAndSet(null, ex);
                return null;
              }
            }
          };

      ImmutableList.Builder<Type<?>> dependencyTypesBuilder = ImmutableList.builder();
      Map<List<Object>, Object> lookupTable = new HashMap<>();
      try {
        for (String dependency : dependencies) {
          Attribute attribute = rule.getRuleClassObject().getAttributeByNameMaybe(dependency);
          if (attribute == null) {
            throw new AttributeNotFoundException(
                String.format("No such attribute %s in rule %s", dependency, rule.getLabel()));
          }
          dependencyTypesBuilder.add(attribute.getType());
        }
        lookupTable.putAll(
            strategy.computeValuesForAllCombinations(
                dependencies, attr.getType(), rule, FixedComputationLimiter.INSTANCE));
        if (caughtEvalExceptionIfAny.get() != null) {
          throw caughtEvalExceptionIfAny.get();
        }
      } catch (AttributeNotFoundException
          | TooManyConfigurableAttributesException
          | EvalException ex) {
        String error = msg + ex.getMessage();
        rule.reportError(error, eventHandler);
        throw new CannotPrecomputeDefaultsException(error);
      }
      return new SkylarkComputedDefault(dependencies, dependencyTypesBuilder.build(), lookupTable);
    }

    private Object computeValue(EventHandler eventHandler, AttributeMap rule)
        throws EvalException, InterruptedException {
      Map<String, Object> attrValues = new HashMap<>();
      for (String attrName : rule.getAttributeNames()) {
        Attribute attr = rule.getAttributeDefinition(attrName);
        if (!attr.hasComputedDefault()) {
          Object value = rule.get(attrName, attr.getType());
          if (!EvalUtils.isNullOrNone(value)) {
            // Some attribute values are not valid Starlark values:
            // visibility is an ImmutableList, for example.
            attrValues.put(attr.getName(), Starlark.fromJava(value, /*mutability=*/ null));
          }
        }
      }
      return invokeCallback(eventHandler, attrValues);
    }

    private Object invokeCallback(EventHandler eventHandler, Map<String, Object> attrValues)
        throws EvalException, InterruptedException {
      ClassObject attrs =
          StructProvider.STRUCT.create(
              attrValues, "No such regular (non computed) attribute '%s'.");
      Object result = callback.call(eventHandler, attrs);
      try {
        return type.cast((result == Starlark.NONE) ? type.getDefaultValue() : result);
      } catch (ClassCastException ex) {
        throw new EvalException(
            location,
            String.format(
                "expected '%s', but got '%s'", type, EvalUtils.getDataTypeName(result, true)));
      }
    }

    private static class AttributeNotFoundException extends Exception {
      private AttributeNotFoundException(String message) {
        super(message);
      }
    }

    static class CannotPrecomputeDefaultsException extends Exception {
      private CannotPrecomputeDefaultsException(String message) {
        super(message);
      }
    }
  }

  /**
   * A class for computed attributes defined in Skylark.
   *
   * <p>Unlike {@link ComputedDefault}, instances of this class contain a pre-computed table of all
   * possible assignments of depended-on attributes and what the Skylark function evaluates to, and
   * {@link #getPossibleValues(Type, Rule)} and {@link #getDefault(AttributeMap)} do lookups in that
   * table.
   */
  static final class SkylarkComputedDefault extends ComputedDefault {

    private final List<Type<?>> dependencyTypes;
    private final Map<List<Object>, Object> lookupTable;

    /**
     * Creates a new SkylarkComputedDefault containing a lookup table.
     *
     * @param dependencies A list of all names of other attributes that are accessed by this
     *     attribute.
     * @param dependencyTypes A list of requiredAttributes' types.
     * @param lookupTable An exhaustive mapping from requiredAttributes assignments to values this
     *     computed default evaluates to.
     */
    SkylarkComputedDefault(
        ImmutableList<String> dependencies,
        ImmutableList<Type<?>> dependencyTypes,
        Map<List<Object>, Object> lookupTable) {
      super(Preconditions.checkNotNull(dependencies));
      this.dependencyTypes = Preconditions.checkNotNull(dependencyTypes);
      this.lookupTable = Preconditions.checkNotNull(lookupTable);
    }

    List<Type<?>> getDependencyTypes() {
      return dependencyTypes;
    }

    Map<List<Object>, Object> getLookupTable() {
      return lookupTable;
    }

    @Override
    public Object getDefault(AttributeMap rule) {
      List<Object> key = ComputationStrategy.createDependencyAssignmentTuple(dependencies(), rule);
      Preconditions.checkState(
          lookupTable.containsKey(key),
          "Error in rule '%s': precomputed value missing for dependencies: %s. Available keys: %s.",
          rule.getLabel(),
          Iterables.toString(key),
          Iterables.toString(lookupTable.keySet()));
      return lookupTable.get(key);
    }

    @Override
    <T> Iterable<T> getPossibleValues(Type<T> type, Rule rule) {
      List<T> result = new ArrayList<>(lookupTable.size());
      for (Object obj : lookupTable.values()) {
        result.add(type.cast(obj));
      }
      return result;
    }
  }

  static class SimpleLateBoundDefault<FragmentT, ValueT>
      extends LateBoundDefault<FragmentT, ValueT> {
    private final Resolver<FragmentT, ValueT> resolver;

    private SimpleLateBoundDefault(boolean useHostConfiguration,
        Class<FragmentT> fragmentClass,
        ValueT defaultValue, Resolver<FragmentT, ValueT> resolver) {
      super(useHostConfiguration, fragmentClass, defaultValue);

      this.resolver = resolver;
    }

    @Override
    public ValueT resolve(Rule rule, AttributeMap attributes, FragmentT input) {
      return resolver.resolve(rule, attributes, input);
    }
  }

  // TODO(b/65746853): Remove documentation about accepting BuildConfiguration when uses are cleaned
  // up.
  /**
   * Provider of values for late-bound attributes. See {@link Attribute#value(LateBoundDefault<?, ?
   * extends TYPE> value)}.
   *
   * <p>Use sparingly - having different values for attributes during loading and analysis can
   * confuse users.
   *
   * @param <FragmentT> The type of value that is used to compute this value. This is usually a
   *     subclass of BuildConfiguration.Fragment. It may also be Void to receive null, or
   *     BuildConfiguration itself to receive the entire configuration.
   * @param <ValueT> The type of value returned by this class. Must be either {@link Void}, a {@link
   *     Label}, or a {@link List} of {@link Label} objects.
   */
  @Immutable
  public abstract static class LateBoundDefault<FragmentT, ValueT> implements StarlarkValue {
    /**
     * Functional interface for computing the value of a late-bound attribute.
     *
     * <p>Implementations of this interface must be immutable.
     */
    @FunctionalInterface
    public interface Resolver<FragmentT, ValueT> {
      ValueT resolve(Rule rule, AttributeMap attributeMap, FragmentT input);
    }

    private final boolean useHostConfiguration;
    private final ValueT defaultValue;
    private final Class<FragmentT> fragmentClass;

    /**
     * Creates a new LateBoundDefault which always returns the given value.
     *
     * <p>This is used primarily for matching names with late-bound attributes on other rules and
     * for testing. Use normal default values if the name does not matter.
     */
    @VisibleForTesting
    public static LabelLateBoundDefault<Void> fromConstantForTesting(Label defaultValue) {
      return new LabelLateBoundDefault<Void>(
          false,
          Void.class,
          Preconditions.checkNotNull(defaultValue),
          (rule, attributes, unused) -> defaultValue) {};
    }

    /**
     * Creates a new LateBoundDefault which always returns null.
     *
     * <p>This is used primarily for matching names with late-bound attributes on other rules and
     * for testing. Use normal default values if the name does not matter.
     */
    @SuppressWarnings("unchecked") // bivariant implementation
    public static <ValueT> LateBoundDefault<Void, ValueT> alwaysNull() {
      return (LateBoundDefault<Void, ValueT>) AlwaysNullLateBoundDefault.INSTANCE;
    }

    LateBoundDefault(
        boolean useHostConfiguration, Class<FragmentT> fragmentClass, ValueT defaultValue) {
      this.useHostConfiguration = useHostConfiguration;
      this.defaultValue = defaultValue;
      this.fragmentClass = fragmentClass;
    }

    /**
     * Whether to look up the label in the host configuration. This is only here for host
     * compilation tools - we usually need to look up labels in the target configuration.
     */
    public final boolean useHostConfiguration() {
      return useHostConfiguration;
    }

    /**
     * Returns the input type that the attribute expects. This is almost always a configuration
     * fragment to be retrieved from the target's configuration (or the host configuration).
     *
     * <p>It may also be {@link Void} to receive null. This is rarely necessary, but can be used,
     * e.g., if the attribute is named to match an attribute in another rule which is late-bound.
     *
     * <p>It may also be BuildConfiguration to receive the entire configuration. This is deprecated,
     * and only necessary when the default is computed from methods of BuildConfiguration itself.
     */
    public final Class<FragmentT> getFragmentClass() {
      return fragmentClass;
    }

    /** The default value for the attribute that is set during the loading phase. */
    public final ValueT getDefault() {
      return defaultValue;
    }

    /**
     * The actual value for the attribute for the analysis phase, which depends on the build
     * configuration. Note that configurations transitions are applied after the late-bound
     * attribute was evaluated.
     *
     * @param rule the rule being evaluated
     * @param attributes interface for retrieving the values of the rule's other attributes
     * @param input the configuration fragment to evaluate with
     */
    public abstract ValueT resolve(Rule rule, AttributeMap attributes, FragmentT input);
  }

  /**
   * An abstract {@link LateBoundDefault} class so that {@code SkylarkLateBoundDefault} can derive
   * from {@link LateBoundDefault} without compromising the type-safety of the second generic
   * parameter to {@link LateBoundDefault}.
   */
  public abstract static class AbstractLabelLateBoundDefault<FragmentT>
      extends LateBoundDefault<FragmentT, Label> {
    protected AbstractLabelLateBoundDefault(
        boolean useHostConfiguration, Class<FragmentT> fragmentClass, Label defaultValue) {
      super(useHostConfiguration, fragmentClass, defaultValue);
    }
  }

  @AutoCodec.VisibleForSerialization
  static class AlwaysNullLateBoundDefault extends SimpleLateBoundDefault<Void, Void> {
    @AutoCodec @AutoCodec.VisibleForSerialization
    static final AlwaysNullLateBoundDefault INSTANCE = new AlwaysNullLateBoundDefault();

    private AlwaysNullLateBoundDefault() {
      super(false, Void.class, null, (rule, attributes, unused) -> null);
    }
  }

  /** A {@link LateBoundDefault} for a {@link Label}. */
  public static class LabelLateBoundDefault<FragmentT>
      extends SimpleLateBoundDefault<FragmentT, Label> {
    @VisibleForTesting
    protected LabelLateBoundDefault(
        boolean useHostConfiguration,
        Class<FragmentT> fragmentClass,
        Label defaultValue,
        Resolver<FragmentT, Label> resolver) {
      super(useHostConfiguration, fragmentClass, defaultValue, resolver);
    }

    /**
     * Creates a new LabelLateBoundDefault which uses the rule, its configured attributes, and a
     * fragment of the target configuration.
     *
     * <p>Note that the configuration fragment here does not take into account any transitions that
     * are on the attribute with this LabelLateBoundDefault as its value. The configuration will be
     * the same as the configuration given to the target bearing the attribute.
     *
     * <p>Nearly all LateBoundDefaults should use this constructor or {@link
     * LabelListLateBoundDefault#fromTargetConfiguration}. There are few situations where it isn't
     * the appropriate option.
     *
     * <p>If you want a late-bound dependency which is configured in the host configuration, just
     * use this method with {@link com.google.devtools.build.lib.analysis.config.HostTransition}. If
     * you also need to decide the label of the dependency with information gained from the host
     * configuration - and it's very unlikely that you do - you can use {@link
     * LabelLateBoundDefault#fromHostConfiguration} as well.
     *
     * <p>If you want to decide an attribute's value based on the value of its other attributes, use
     * a subclass of {@link ComputedDefault}. The only time you should need {@link
     * LabelListLateBoundDefault#fromRuleAndAttributesOnly} is if you need access to three or more
     * configurable attributes, or if you need to match names with a late-bound attribute on another
     * rule.
     *
     * <p>If you have a constant-valued attribute, but you need it to have the same name as an
     * attribute on another rule which is late-bound, use {@link #alwaysNull}.
     *
     * @param fragmentClass The fragment to receive from the target configuration. May also be
     *     BuildConfiguration.class to receive the entire configuration (deprecated) - in this case,
     *     you must only use methods of BuildConfiguration itself, and not use any fragments.
     * @param defaultValue The default {@link Label} to return at loading time, when the
     *     configuration is not available.
     * @param resolver A function which will compute the actual value with the configuration.
     */
    public static <FragmentT> LabelLateBoundDefault<FragmentT> fromTargetConfiguration(
        Class<FragmentT> fragmentClass, Label defaultValue, Resolver<FragmentT, Label> resolver) {
      Preconditions.checkArgument(
          !fragmentClass.equals(Void.class),
          "Use fromRuleAndAttributesOnly to specify a LateBoundDefault which does not use "
              + "configuration.");
      return new LabelLateBoundDefault<>(false, fragmentClass, defaultValue, resolver);
    }

    /**
     * Creates a new LateBoundDefault which uses the rule, its configured attributes, and a fragment
     * of the host configuration.
     *
     * <p>This should only be necessary in very specialized cases. In almost all cases, you don't
     * need this method, just {@link #fromTargetConfiguration} and {@link
     * com.google.devtools.build.lib.analysis.config.HostTransition}.
     *
     * <p>This method only affects the configuration fragment passed to {@link #resolve}. You must
     * also use {@link com.google.devtools.build.lib.analysis.config.HostTransition}, so that the
     * dependency will be analyzed in the host configuration.
     *
     * @param fragmentClass The fragment to receive from the host configuration. May also be
     *     BuildConfiguration.class to receive the entire configuration (deprecated) - in this case,
     *     you must only use methods of BuildConfiguration itself, and not use any fragments. It is
     *     very rare that a LateBoundDefault should need a host configuration fragment; use {@link
     *     #fromTargetConfiguration} in most cases.
     * @param defaultValue The default {@link Label} to return at loading time, when the
     *     configuration is not available.
     * @param resolver A function which will compute the actual value with the configuration.
     */
    public static <FragmentT> LabelLateBoundDefault<FragmentT> fromHostConfiguration(
        Class<FragmentT> fragmentClass, Label defaultValue, Resolver<FragmentT, Label> resolver) {
      Preconditions.checkArgument(
          !fragmentClass.equals(Void.class),
          "Use fromRuleAndAttributesOnly to specify a LateBoundDefault which does not use "
              + "configuration.");
      return new LabelLateBoundDefault<>(true, fragmentClass, defaultValue, resolver);
    }
  }

  /** A {@link LateBoundDefault} for a {@link List} of {@link Label} objects. */
  public static class LabelListLateBoundDefault<FragmentT>
      extends SimpleLateBoundDefault<FragmentT, List<Label>> {
    private LabelListLateBoundDefault(
        boolean useHostConfiguration,
        Class<FragmentT> fragmentClass,
        Resolver<FragmentT, List<Label>> resolver) {
      super(useHostConfiguration, fragmentClass, ImmutableList.of(), resolver);
    }

    public static <FragmentT> LabelListLateBoundDefault<FragmentT> fromTargetConfiguration(
        Class<FragmentT> fragmentClass, Resolver<FragmentT, List<Label>> resolver) {
      Preconditions.checkArgument(
          !fragmentClass.equals(Void.class),
          "Use fromRuleAndAttributesOnly to specify a LateBoundDefault which does not use "
              + "configuration.");
      return new LabelListLateBoundDefault<>(false, fragmentClass, resolver);
    }

    /**
     * Creates a new LabelListLateBoundDefault which uses only the rule and its configured
     * attributes.
     *
     * <p>This should only be necessary in very specialized cases. In almost all cases, you don't
     * need this method, just use {@link ComputedDefault}.
     *
     * <p>This is used primarily for computing values based on three or more configurable attributes
     * and/or matching names with late-bound attributes on other rules.
     *
     * @param resolver A function which will compute the actual value with the configuration.
     */
    public static LabelListLateBoundDefault<Void> fromRuleAndAttributesOnly(
        Resolver<Void, List<Label>> resolver) {
      return new LabelListLateBoundDefault<>(false, Void.class, resolver);
    }
  }

  private final String name;

  private final String doc;

  private final Type<?> type;

  private final Set<PropertyFlag> propertyFlags;

  // Exactly one of these conditions is true:
  // 1. defaultValue == null.
  // 2. defaultValue instanceof ComputedDefault &&
  //    type.isValid(defaultValue.getDefault())
  // 3. defaultValue instanceof SkylarkComputedDefaultTemplate &&
  //    type.isValid(defaultValue.computePossibleValues().getDefault())
  // 4. type.isValid(defaultValue).
  // 5. defaultValue instanceof LateBoundDefault &&
  //    type.isValid(defaultValue.getDefault(configuration))
  // (We assume a hypothetical Type.isValid(Object) predicate.)
  private final Object defaultValue;

  private final TransitionFactory<AttributeTransitionData> transitionFactory;

  /**
   * For label or label-list attributes, this predicate returns which rule
   * classes are allowed for the targets in the attribute.
   */
  private final RuleClassNamePredicate allowedRuleClassesForLabels;

  /**
   * For label or label-list attributes, this predicate returns which rule
   * classes are allowed for the targets in the attribute with warning.
   */
  private final RuleClassNamePredicate allowedRuleClassesForLabelsWarning;

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

  private final RequiredProviders requiredProviders;

  private final ImmutableList<RuleAspect<?>> aspects;

  private final int hashCode;

  /**
   * Constructs a rule attribute with the specified name, type and default value.
   *
   * @param name the name of the attribute
   * @param type the type of the attribute
   * @param defaultValue the default value to use for this attribute if none is specified in rule
   *     declaration in the BUILD file. Must be null, or of type "type". May be an instance of
   *     ComputedDefault, in which case its getDefault() method must return an instance of "type",
   *     or null. Must be immutable.
   * @param transitionFactory the configuration transition for this attribute (which must be of type
   *     LABEL, LABEL_LIST, NODEP_LABEL or NODEP_LABEL_LIST).
   */
  Attribute(
      String name,
      String doc,
      Type<?> type,
      Set<PropertyFlag> propertyFlags,
      Object defaultValue,
      TransitionFactory<AttributeTransitionData> transitionFactory,
      RuleClassNamePredicate allowedRuleClassesForLabels,
      RuleClassNamePredicate allowedRuleClassesForLabelsWarning,
      FileTypeSet allowedFileTypesForLabels,
      ValidityPredicate validityPredicate,
      Predicate<AttributeMap> condition,
      PredicateWithMessage<Object> allowedValues,
      RequiredProviders requiredProviders,
      ImmutableList<RuleAspect<?>> aspects) {
    Preconditions.checkArgument(
        (NoTransition.isInstance(transitionFactory))
            || type.getLabelClass() == LabelClass.DEPENDENCY
            || type.getLabelClass() == LabelClass.NONDEP_REFERENCE,
        "Configuration transitions can only be specified for label or label list attributes");
    Preconditions.checkArgument(
        isLateBound(name) == (defaultValue instanceof LateBoundDefault),
        "late bound attributes require a default value that is late bound (and vice versa): %s",
        name);
    if (isLateBound(name)) {
      LateBoundDefault<?, ?> lateBoundDefault = (LateBoundDefault<?, ?>) defaultValue;
      Preconditions.checkArgument(
          !lateBoundDefault.useHostConfiguration() || transitionFactory.isHost(),
          "a late bound default value using the host configuration must use the host transition");
    }

    this.name = name;
    this.doc = doc;
    this.type = type;
    this.propertyFlags = propertyFlags;
    this.defaultValue = defaultValue;
    this.transitionFactory = transitionFactory;
    this.allowedRuleClassesForLabels = allowedRuleClassesForLabels;
    this.allowedRuleClassesForLabelsWarning = allowedRuleClassesForLabelsWarning;
    this.allowedFileTypesForLabels = allowedFileTypesForLabels;
    this.validityPredicate = validityPredicate;
    this.condition = condition;
    this.allowedValues = allowedValues;
    this.requiredProviders = requiredProviders;
    this.aspects = aspects;
    this.hashCode =
        Objects.hash(
            name,
            doc,
            type,
            propertyFlags,
            defaultValue,
            transitionFactory,
            allowedRuleClassesForLabels,
            allowedRuleClassesForLabelsWarning,
            allowedFileTypesForLabels,
            validityPredicate,
            condition,
            allowedValues,
            requiredProviders,
            aspects);
  }

  /**
   * Returns the name of this attribute.
   */
  public String getName() {
    return name;
  }

  /** Returns the doc string for that attribute, if any. */
  public String getDoc() {
    return doc;
  }

  /**
   * Returns the public name of this attribute. This is the name we use in Skylark code
   * and we can use it to display to the end-user.
   * Implicit and late-bound attributes start with '_' (instead of '$' or ':').
   */
  public String getPublicName() {
    return getSkylarkName(getName());
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
   *  Returns true if output_licenses should be used for checking licensing.
   */
  public boolean useOutputLicenses() {
    return getPropertyFlag(PropertyFlag.OUTPUT_LICENSES);
  }

  /**
   * Returns true if this attribute uses a starlark-defined, non analysis-test configuration
   * transition. Starlark-defined analysis-test configuration transitions are handled separately.
   * See {@link #hasAnalysisTestTransition}.
   */
  public boolean hasStarlarkDefinedTransition() {
    return getPropertyFlag(PropertyFlag.HAS_STARLARK_DEFINED_TRANSITION);
  }

  /**
   * Returns true if this attributes uses Starlark-defined configuration transition designed
   * specifically for rules which run analysis tests.
   */
  public boolean hasAnalysisTestTransition() {
    return getPropertyFlag(PropertyFlag.HAS_ANALYSIS_TEST_TRANSITION);
  }

  /**
   * Returns the configuration transition factory for this attribute for label or label list
   * attributes. For other attributes it will always return {@code NONE}.
   */
  public TransitionFactory<AttributeTransitionData> getTransitionFactory() {
    return transitionFactory;
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
    return getPropertyFlag(PropertyFlag.CHECK_CONSTRAINTS_OVERRIDE);
  }

  public boolean skipConstraintsOverride() {
    return getPropertyFlag(PropertyFlag.SKIP_CONSTRAINTS_OVERRIDE);
  }

  /**
   * Returns true if this attribute's value can be influenced by the build configuration.
   */
  public boolean isConfigurable() {
    // Output types are excluded because of Rule#populateExplicitOutputFiles.
    return !(type.getLabelClass() == LabelClass.OUTPUT
        || getPropertyFlag(PropertyFlag.NONCONFIGURABLE));
  }

  /**
   * Returns a predicate that evaluates to true for rule classes that are allowed labels in this
   * attribute. If this is not a label or label-list attribute, the returned predicate always
   * evaluates to true.
   *
   * <p>NOTE: This may return Predicates.<RuleClass>alwaysTrue() as a sentinel meaning "do the right
   * thing", rather than actually allowing all rule classes in that attribute. Others parts of bazel
   * code check for that specific instance.
   */
  public Predicate<RuleClass> getAllowedRuleClassesPredicate() {
    return allowedRuleClassesForLabels.asPredicateOfRuleClass();
  }

  /**
   * Returns a predicate that evaluates to true for rule classes that are
   * allowed labels in this attribute with warning. If this is not a label or label-list
   * attribute, the returned predicate always evaluates to true.
   */
  public Predicate<RuleClass> getAllowedRuleClassesWarningPredicate() {
    return allowedRuleClassesForLabelsWarning.asPredicateOfRuleClass();
  }

  public RequiredProviders getRequiredProviders() {
    return requiredProviders;
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
    ImmutableList.Builder<Aspect> builder = null;
    for (RuleAspect<?> aspect : aspects) {
      Aspect a = aspect.getAspect(rule);
      if (a != null) {
        if (builder == null) {
          builder = ImmutableList.builder();
        }
        builder.add(a);
      }
    }
    return builder == null ? ImmutableList.of() : builder.build();
  }

  public ImmutableList<AspectClass> getAspectClasses() {
    ImmutableList.Builder<AspectClass> result = ImmutableList.builder();
    for (RuleAspect<?> aspect : aspects) {
      result.add(aspect.getAspectClass());
    }
    return result.build();
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
    } else if (defaultValue instanceof LateBoundDefault<?, ?>) {
      return ((LateBoundDefault<?, ?>) defaultValue).getDefault();
    } else {
      return defaultValue;
    }
  }

  /**
   * Returns the default value of this attribute, even if it has a condition, is a computed default,
   * or a late-bound default.
   */
  @VisibleForTesting
  public Object getDefaultValueUnchecked() {
    return defaultValue;
  }

  public LateBoundDefault<?, ?> getLateBoundDefault() {
    Preconditions.checkState(isLateBound());
    return (LateBoundDefault<?, ?>) defaultValue;
  }

  /**
   * Returns true iff this attribute has a computed default or a condition.
   *
   * @see #getDefaultValue(Rule)
   */
  boolean hasComputedDefault() {
    return (defaultValue instanceof ComputedDefault)
        || (defaultValue instanceof SkylarkComputedDefaultTemplate)
        || (condition != null);
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

  /** Returns whether this attribute is considered private in Skylark. */
  private static boolean isPrivateAttribute(String nativeAttrName) {
    return isLateBound(nativeAttrName) || isImplicit(nativeAttrName);
  }

  /**
   * Returns the Skylark-usable name of this attribute.
   *
   * Implicit and late-bound attributes start with '_' (instead of '$' or ':').
   */
  public static String getSkylarkName(String nativeAttrName) {
    if (isPrivateAttribute(nativeAttrName)) {
      return "_" + nativeAttrName.substring(1);
    }
    return nativeAttrName;
  }

  @Override
  public String toString() {
    return "Attribute(" + name + ", " + type + ")";
  }

  @Override
  public int compareTo(Attribute other) {
    return name.compareTo(other.name);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    Attribute attribute = (Attribute) o;
    return Objects.equals(hashCode, attribute.hashCode)
        && Objects.equals(name, attribute.name)
        && Objects.equals(doc, attribute.doc)
        && Objects.equals(type, attribute.type)
        && Objects.equals(propertyFlags, attribute.propertyFlags)
        && Objects.equals(defaultValue, attribute.defaultValue)
        && Objects.equals(transitionFactory, attribute.transitionFactory)
        && Objects.equals(allowedRuleClassesForLabels, attribute.allowedRuleClassesForLabels)
        && Objects.equals(
            allowedRuleClassesForLabelsWarning, attribute.allowedRuleClassesForLabelsWarning)
        && Objects.equals(allowedFileTypesForLabels, attribute.allowedFileTypesForLabels)
        && Objects.equals(validityPredicate, attribute.validityPredicate)
        && Objects.equals(condition, attribute.condition)
        && Objects.equals(allowedValues, attribute.allowedValues)
        && Objects.equals(requiredProviders, attribute.requiredProviders)
        && Objects.equals(aspects, attribute.aspects);
  }

  @Override
  public int hashCode() {
    return hashCode;
  }

  /**
   * Returns a replica builder of this Attribute.
   */
  public <TYPE> Attribute.Builder<TYPE> cloneBuilder(Type<TYPE> tp) {
    Preconditions.checkArgument(tp == this.type);
    Builder<TYPE> builder = new Builder<>(name, tp);
    builder.doc = doc;
    builder.allowedFileTypesForLabels = allowedFileTypesForLabels;
    builder.allowedRuleClassesForLabels = allowedRuleClassesForLabels;
    builder.allowedRuleClassesForLabelsWarning = allowedRuleClassesForLabelsWarning;
    builder.requiredProvidersBuilder = requiredProviders.copyAsBuilder();
    builder.validityPredicate = validityPredicate;
    builder.condition = condition;
    builder.transitionFactory = transitionFactory;
    builder.propertyFlags = newEnumSet(propertyFlags, PropertyFlag.class);
    builder.value = defaultValue;
    builder.valueSet = false;
    builder.allowedValues = allowedValues;
    builder.aspects = new LinkedHashMap<>();
    for (RuleAspect<?> aspect : aspects) {
      builder.aspects.put(aspect.getName(), aspect);
    }

    return builder;
  }

  public Attribute.Builder<?> cloneBuilder() {
    return cloneBuilder(this.type);
  }
}
