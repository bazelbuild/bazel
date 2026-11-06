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
package com.google.devtools.build.lib.analysis;

import static com.google.devtools.build.lib.analysis.DependencyKind.OUTPUT_FILE_RULE_DEPENDENCY;
import static com.google.devtools.build.lib.analysis.DependencyKind.TRANSITIVE_VISIBILITY_DEPENDENCY;
import static com.google.devtools.build.lib.analysis.DependencyKind.VISIBILITY_DEPENDENCY;

import com.google.auto.value.AutoOneOf;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.DependencyKind.AttributeDependencyKind;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.Attribute.LateBoundDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.AttributeProvider;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.DeclaredExecGroup;
import com.google.devtools.build.lib.packages.EnvironmentGroup;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Type.LabelClass;
import com.google.devtools.build.lib.skyframe.toolchains.UnloadedToolchainContext;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.syntax.Location;

/**
 * Helpers for resolution for dependencies between configured targets.
 *
 * <p>Includes logic to determine all attribute dependencies and their associated labels.
 */
public final class DependencyResolutionHelpers {

  private DependencyResolutionHelpers() {}

  /** The tuple {@link #computeDependencyLabels} outputs. */
  public static final class DependencyLabels {
    private final OrderedSetMultimap<DependencyKind, Label> labels;
    @Nullable private final ConfiguredAttributeMapper attributeMap;

    private DependencyLabels(
        OrderedSetMultimap<DependencyKind, Label> labels,
        @Nullable ConfiguredAttributeMapper attributeMap) {
      this.labels = labels;
      this.attributeMap = attributeMap;
    }

    public OrderedSetMultimap<DependencyKind, Label> labels() {
      return labels;
    }

    @Nullable // Non-null for rules and output files when there are aspects that apply to files.
    public ConfiguredAttributeMapper attributeMap() {
      return attributeMap;
    }
  }

  public static DependencyLabels computeDependencyLabels(
      TargetAndConfiguration node,
      ImmutableList<Aspect> aspects,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      @Nullable ToolchainCollection<ToolchainContext> toolchainContexts,
      @Nullable ToolchainCollection<UnloadedToolchainContext> baseTargetUnloadedToolchainContexts)
      throws Failure, InterruptedException {
    Target target = node.getTarget();
    BuildConfigurationValue config = node.getConfiguration();
    OrderedSetMultimap<DependencyKind, Label> outgoingLabels = OrderedSetMultimap.create();

    // TODO(bazel-team): Figure out a way to implement the below using LabelVisitationUtils.
    Rule fromRule;
    ConfiguredAttributeMapper attributeMap = null;
    if (target instanceof OutputFile) {
      Preconditions.checkNotNull(config);
      addVisibilityDepLabels(target.getVisibilityDependencyLabels(), outgoingLabels);
      addTransitiveVisibilityDepLabel(
          target.getPackageDeclarations().getPackageArgs().transitiveVisibility(), outgoingLabels);
      Rule rule = ((OutputFile) target).getGeneratingRule();
      outgoingLabels.put(OUTPUT_FILE_RULE_DEPENDENCY, rule.getLabel());
      if (Iterables.any(aspects, a -> a.getDefinition().applyToFiles())) {
        attributeMap = ConfiguredAttributeMapper.of(rule, configConditions, config);
        resolveAttributes(
            /* includeRuleAttributes= */ false,
            aspects,
            outgoingLabels,
            rule,
            attributeMap,
            config);
      }
      addToolchainDeps(toolchainContexts, outgoingLabels);
    } else if (target instanceof InputFile) {
      addVisibilityDepLabels(target.getVisibilityDependencyLabels(), outgoingLabels);
      addTransitiveVisibilityDepLabel(
          target.getPackageDeclarations().getPackageArgs().transitiveVisibility(), outgoingLabels);
    } else if (target instanceof EnvironmentGroup) {
      addVisibilityDepLabels(target.getVisibilityDependencyLabels(), outgoingLabels);
    } else if (target instanceof Rule rule) {
      fromRule = rule;
      attributeMap = ConfiguredAttributeMapper.of(fromRule, configConditions, config);
      addTransitiveVisibilityDepLabel(
          fromRule.getPackageDeclarations().getPackageArgs().transitiveVisibility(),
          outgoingLabels);
      visitRule(
          node,
          aspects,
          attributeMap,
          toolchainContexts,
          baseTargetUnloadedToolchainContexts,
          outgoingLabels);
    } else if (target instanceof PackageGroup packageGroup) {
      outgoingLabels.putAll(VISIBILITY_DEPENDENCY, packageGroup.getIncludes());
    } else {
      throw new IllegalStateException(target.getLabel().toString());
    }
    return new DependencyLabels(outgoingLabels, attributeMap);
  }

  /** The results of {@link #getExecutionPlatformLabel} as a tagged union. */
  @AutoOneOf(DependencyResolutionHelpers.ExecutionPlatformResult.Kind.class)
  public abstract static class ExecutionPlatformResult {
    /** Tags for the possible results. */
    public enum Kind {
      /** A label was successfully determined. */
      LABEL,
      /**
       * A label was successfully determined to be null.
       *
       * <p>{@link AutoOneOf} does not permit {@code @Nullable} so this is distinct from {@link
       * #LABEL}.
       */
      NULL_LABEL,
      /**
       * The dependency should be skipped.
       *
       * <p>See comments in {@link #getExecutionPlatformLabel} for details.
       */
      SKIP,
      /** An error message. */
      ERROR
    }

    public abstract Kind kind();

    public abstract Label label();

    abstract void nullLabel();

    abstract void skip();

    public abstract String error();

    private static ExecutionPlatformResult ofLabel(Label label) {
      return AutoOneOf_DependencyResolutionHelpers_ExecutionPlatformResult.label(label);
    }

    private static ExecutionPlatformResult ofNullLabel() {
      return AutoOneOf_DependencyResolutionHelpers_ExecutionPlatformResult.nullLabel();
    }

    private static ExecutionPlatformResult ofSkip() {
      return AutoOneOf_DependencyResolutionHelpers_ExecutionPlatformResult.skip();
    }

    private static ExecutionPlatformResult ofError(String message) {
      return AutoOneOf_DependencyResolutionHelpers_ExecutionPlatformResult.error(message);
    }
  }

  public static ExecutionPlatformResult getExecutionPlatformLabel(
      AttributeDependencyKind kind,
      @Nullable ToolchainCollection<ToolchainContext> toolchainContexts,
      @Nullable ToolchainCollection<UnloadedToolchainContext> baseTargetUnloadedToolchainContexts,
      ImmutableList<Aspect> aspectsList) {
    if (aspectsList.isEmpty() || isMainAspect(aspectsList, kind.getOwningAspect())) {
      return getExecutionPlatformLabel(kind, toolchainContexts);
    } else if (kind.getOwningAspect() == null) {
      // During aspect evaluation, use {@code baseTargetUnloadedToolchainContexts} for the base
      // target's dependencies.
      return getExecutionPlatformLabel(kind, baseTargetUnloadedToolchainContexts);
    } else {
      ExecutionPlatformResult executionPlatformResult =
          getExecutionPlatformLabel(kind, toolchainContexts);
      if (executionPlatformResult.kind() == ExecutionPlatformResult.Kind.ERROR) {
        // TODO(b/373963347): Make the toolchain contexts of base aspects available to be used with
        // their corresponding dependencies.
        // Currently dependencies of the base aspects are resolved with the toolchain context of the
        // main aspect, skip errors as actual errors would be reported during the base aspect
        // evaluation.
        return ExecutionPlatformResult.ofSkip();
      } else {
        return executionPlatformResult;
      }
    }
  }

  private static ExecutionPlatformResult getExecutionPlatformLabel(
      AttributeDependencyKind kind,
      @Nullable ToolchainCollection<? extends ToolchainContext> toolchainContexts) {
    if (toolchainContexts == null) {
      return ExecutionPlatformResult.ofNullLabel();
    }

    // The exec transition may be composed with other transitions (via transition.and_then), so
    // look for it anywhere in the (possibly composed) transition factory.
    ExecutionTransitionFactory execTransitionFactory =
        findExecutionTransitionFactory(kind.getAttribute().getTransitionFactory());
    if (execTransitionFactory == null) {
      return ExecutionPlatformResult.ofLabel(
          toolchainContexts
              .getToolchainContext(DeclaredExecGroup.DEFAULT_EXEC_GROUP_NAME)
              .executionPlatform()
              .label());
    }

    String execGroup = execTransitionFactory.getExecGroup();
    if (toolchainContexts.hasToolchainContext(execGroup)) {
      PlatformInfo platform = toolchainContexts.getToolchainContext(execGroup).executionPlatform();
      return platform == null
          ? ExecutionPlatformResult.ofNullLabel()
          : ExecutionPlatformResult.ofLabel(platform.label());
    }

    return ExecutionPlatformResult.ofError(
        String.format(
            "Attr '%s' declares a transition for non-existent exec group '%s'",
            kind.getAttribute().getName(), execGroup));
  }

  /**
   * Returns the {@link ExecutionTransitionFactory} within the given transition factory, or {@code
   * null} if there is none. A composition contains at most one native transition, so there is at
   * most one match.
   */
  @Nullable
  private static ExecutionTransitionFactory findExecutionTransitionFactory(
      TransitionFactory<AttributeTransitionData> transitionFactory) {
    var found = new ExecutionTransitionFactory[1];
    transitionFactory.visit(
        factory -> {
          if (factory instanceof ExecutionTransitionFactory execFactory) {
            found[0] = execFactory;
          }
        });
    return found[0];
  }

  /** True if {@code owningAspect} is the main aspect, the last one in {@code aspectsList}. */
  private static boolean isMainAspect(
      ImmutableList<Aspect> aspectsList, @Nullable AspectClass owningAspect) {
    return Iterables.getLast(aspectsList).getAspectClass().equals(owningAspect);
  }

  /** Indicates a failure during dependency resolution. */
  public static class Failure extends Exception {
    @Nullable private final Location location;

    private Failure(Location location, String message) {
      super(message);
      this.location = location;
    }

    /** Returns the location of the error, if known. */
    @Nullable
    public Location getLocation() {
      return location;
    }
  }

  private static void visitRule(
      TargetAndConfiguration node,
      ImmutableList<Aspect> aspects,
      ConfiguredAttributeMapper attributeMap,
      @Nullable ToolchainCollection<ToolchainContext> toolchainContexts,
      @Nullable ToolchainCollection<UnloadedToolchainContext> baseTargetUnloadedToolchainContexts,
      OrderedSetMultimap<DependencyKind, Label> outgoingLabels)
      throws Failure, InterruptedException {
    Preconditions.checkArgument(node.getTarget() instanceof Rule, node);
    BuildConfigurationValue ruleConfig = Preconditions.checkNotNull(node.getConfiguration(), node);
    Rule rule = (Rule) node.getTarget();

    try {
      attributeMap.validateAttributes();
    } catch (ConfiguredAttributeMapper.ValidationException ex) {
      throw new Failure(rule.getLocation(), ex.getMessage());
    }

    Iterable<Label> visibilityDepLabels = rule.getVisibilityDependencyLabels();
    addVisibilityDepLabels(visibilityDepLabels, outgoingLabels);
    resolveAttributes(
        /* includeRuleAttributes= */ true, aspects, outgoingLabels, rule, attributeMap, ruleConfig);

    // Add the rule's visibility labels (which may come from the rule or from package defaults).
    addExplicitDeps(outgoingLabels, rule, "visibility", visibilityDepLabels);

    // Add package default constraints when the rule doesn't explicitly declare them.
    //
    // Note that this can have subtle implications for constraint semantics. For example: say that
    // package defaults declare compatibility with ':foo' and rule R declares compatibility with
    // ':bar'. Does that mean that R is compatible with [':foo', ':bar'] or just [':bar']? In other
    // words, did R's author intend to add additional compatibility to the package defaults or to
    // override them? More severely, what if package defaults "restrict" support to just [':baz']?
    // Should R's declaration signify [':baz'] + ['bar'], [ORIGINAL_DEFAULTS] + ['bar'], or
    // something else?
    //
    // Rather than try to answer these questions with possibly confusing logic, we take the
    // simple approach of assigning the rule's "restriction" attribute to the rule-declared value if
    // it exists, else the package defaults value (and likewise for "compatibility"). This may not
    // always provide what users want, but it makes it easy for them to understand how rule
    // declarations and package defaults intermix (and how to refactor them to get what they want).
    //
    // An alternative model would be to apply the "rule declaration" / "rule class defaults"
    // relationship, i.e. the rule class' "compatibility" and "restriction" declarations are merged
    // to generate a set of default environments, then the rule's declarations are independently
    // processed on top of that. This protects against obscure coupling behavior between
    // declarations from wildly different places (e.g. it offers clear answers to the examples posed
    // above). But within the scope of a single package it seems better to keep the model simple and
    // make the user responsible for resolving ambiguities.
    if (!rule.isAttributeValueExplicitlySpecified(RuleClass.COMPATIBLE_ENVIRONMENT_ATTR)) {
      addExplicitDeps(
          outgoingLabels,
          rule,
          RuleClass.COMPATIBLE_ENVIRONMENT_ATTR,
          rule.getPackageDeclarations().getPackageArgs().defaultCompatibleWith());
    }
    if (!rule.isAttributeValueExplicitlySpecified(RuleClass.RESTRICTED_ENVIRONMENT_ATTR)) {
      addExplicitDeps(
          outgoingLabels,
          rule,
          RuleClass.RESTRICTED_ENVIRONMENT_ATTR,
          rule.getPackageDeclarations().getPackageArgs().defaultRestrictedTo());
    }

    addToolchainDeps(toolchainContexts, outgoingLabels);
    addBaseTargetToolchainDeps(baseTargetUnloadedToolchainContexts, outgoingLabels);
  }

  private static void addToolchainDeps(
      ToolchainCollection<ToolchainContext> toolchainContexts,
      OrderedSetMultimap<DependencyKind, Label> outgoingLabels) {
    if (toolchainContexts != null) {
      for (Map.Entry<String, ToolchainContext> entry : toolchainContexts.contextMap().entrySet()) {
        outgoingLabels.putAll(
            DependencyKind.forExecGroup(entry.getKey()),
            entry.getValue().resolvedToolchainLabels());
      }
    }
  }

  private static void addBaseTargetToolchainDeps(
      @Nullable ToolchainCollection<UnloadedToolchainContext> toolchainContexts,
      OrderedSetMultimap<DependencyKind, Label> outgoingLabels) {
    if (toolchainContexts == null) {
      return;
    }
    for (Map.Entry<String, UnloadedToolchainContext> execGroup :
        toolchainContexts.contextMap().entrySet()) {
      for (var toolchainTypeToResolved :
          execGroup.getValue().toolchainTypeToResolved().asMap().entrySet()) {
        // map entries from (exec group, toolchain type) to resolved toolchain labels. We need to
        // distinguish the resolved toolchains per type because aspects propagate on toolchains
        // based on the types specified in `toolchains_aspects`. So even if 2 types resolved to the
        // same toolchain target, their CT will be different if an aspect propagates to one type but
        // not the other.
        outgoingLabels.putAll(
            DependencyKind.forBaseTargetExecGroup(
                execGroup.getKey(), toolchainTypeToResolved.getKey().typeLabel()),
            toolchainTypeToResolved.getValue());
      }
    }
  }

  /**
   * Resolves the dependencies through the attributes that should be visited for this rule/aspect
   * combination.
   *
   * <p>For aspects evaluation, all attributes of the main aspect (last aspect in {@code aspects})
   * are visited, even if they have the same name as an attribute in the rule or a base aspect,
   * because main aspect attributes are separated and retrieved from `ctx.attr`.
   *
   * <p>Attributes of the underlying rule and base aspects are merged and retrieved from
   * `ctx.rule.attr` with rule attributes taking precedence then aspects' attributes based on the
   * aspect order in the aspects path (lowest order to highest).
   *
   * <p>This deliberately iterates the attributes directly instead of materializing a list of {@link
   * AttributeDependencyKind}s, since it runs for every configured target and aspect and most
   * attributes don't give rise to dependencies.
   *
   * @param includeRuleAttributes whether to visit the attributes of {@code rule} itself. If false,
   *     only the aspects' attributes are visited (and base aspect attributes are not deduplicated
   *     against the rule's attributes).
   */
  private static void resolveAttributes(
      boolean includeRuleAttributes,
      ImmutableList<Aspect> aspects,
      OrderedSetMultimap<DependencyKind, Label> outgoingLabels,
      Rule rule,
      ConfiguredAttributeMapper attributeMap,
      BuildConfigurationValue ruleConfig)
      throws InterruptedException {
    AttributeProvider ruleAttributeProvider = rule.getRuleClassObject().getAttributeProvider();
    if (includeRuleAttributes) {
      for (Attribute attribute : ruleAttributeProvider.getAttributes()) {
        maybeResolveAttribute(
            attribute, /* owningAspect= */ null, outgoingLabels, rule, attributeMap, ruleConfig);
      }
    }

    if (aspects.isEmpty()) {
      return;
    }

    // Add all the main aspect's attributes
    Aspect mainAspect = Iterables.getLast(aspects);
    for (Attribute attribute : mainAspect.getDefinition().getAttributes().values()) {
      maybeResolveAttribute(
          attribute, mainAspect.getAspectClass(), outgoingLabels, rule, attributeMap, ruleConfig);
    }

    if (aspects.size() == 1) {
      return;
    }

    // For base aspects, if multiple attributes have the same name, take the first encountered in
    // the aspects path.
    Set<String> processedBaseAspectAttributes = new HashSet<>();
    for (Aspect aspect : aspects.subList(0, aspects.size() - 1)) {
      for (Attribute attribute : aspect.getDefinition().getAttributes().values()) {
        String name = attribute.getName();
        if (includeRuleAttributes && ruleAttributeProvider.getAttributeIndex(name) != null) {
          continue;
        }
        if (processedBaseAspectAttributes.add(name)) {
          maybeResolveAttribute(
              attribute, aspect.getAspectClass(), outgoingLabels, rule, attributeMap, ruleConfig);
        }
      }
    }
  }

  private static void maybeResolveAttribute(
      Attribute attribute,
      @Nullable AspectClass owningAspect,
      OrderedSetMultimap<DependencyKind, Label> outgoingLabels,
      Rule rule,
      ConfiguredAttributeMapper attributeMap,
      BuildConfigurationValue ruleConfig)
      throws InterruptedException {
    // Not only is resolving CONFIG_SETTING_DEPS_ATTRIBUTE deps here wasteful, since the only
    // place they're used is in ConfiguredTargetFunction.getConfigConditions, but it actually
    // breaks trimming as shown by
    // FeatureFlagManualTrimmingTest#featureFlagInUnusedSelectBranchButNotInTransitiveConfigs_DoesNotError
    // because it resolves a dep that trimming (correctly) doesn't account for because it's part
    // of an unchosen select() branch.
    if (attribute.getName().equals(RuleClass.CONFIG_SETTING_DEPS_ATTRIBUTE)) {
      return;
    }
    Type<?> type = attribute.getType();
    if (type.getLabelClass() == LabelClass.NONE && !attribute.isMaterializing()) {
      return;
    }
    if (type.equals(BuildType.OUTPUT)
        || type.equals(BuildType.OUTPUT_LIST)
        || type.equals(BuildType.NODEP_LABEL)
        || type.equals(BuildType.NODEP_LABEL_LIST)
        || type.equals(BuildType.DORMANT_LABEL)
        || type.equals(BuildType.DORMANT_LABEL_LIST)
        || type.equals(BuildType.GENQUERY_SCOPE_TYPE)
        || type.equals(BuildType.GENQUERY_SCOPE_TYPE_LIST)) {
      // These types invoke visitLabels() so that they are reported in "bazel query" but do not
      // create a dependency. Maybe it's better to remove that, but then the labels() query
      // function would need to be rethought.
      return;
    }

    resolveAttribute(attribute, type, owningAspect, outgoingLabels, rule, attributeMap, ruleConfig);
  }

  private static <T> void resolveAttribute(
      Attribute attribute,
      Type<T> type,
      @Nullable AspectClass owningAspect,
      OrderedSetMultimap<DependencyKind, Label> outgoingLabels,
      Rule rule,
      ConfiguredAttributeMapper attributeMap,
      BuildConfigurationValue ruleConfig)
      throws InterruptedException {
    T attributeValue = null;
    if (attribute.isImplicit()) {
      // Since the attributes that come from aspects do not appear in attributeMap, we have to get
      // their values from somewhere else. This incidentally means that aspects attributes are not
      // configurable. It would be nice if that wasn't the case, but we'd have to revamp how
      // attribute mapping works, which is a large chunk of work.
      if (owningAspect == null) {
        attributeValue = attributeMap.get(attribute.getName(), type);
      } else {
        Object defaultValue = attribute.getDefaultValue(rule);
        attributeValue =
            type.cast(
                defaultValue instanceof ComputedDefault computedDefault
                    ? computedDefault.getDefault(attributeMap)
                    : defaultValue);
      }
    } else if (attribute.isMaterializing()) {
      // These attributes are resolved by calling the materializer function in
      // DependencyMapProducer. The reason is that they need the analyzed versions some direct
      // dependencies and we can't do that here.
      outgoingLabels.put(createDependencyKind(attribute, owningAspect), null);
    } else if (attribute.isLateBound()) {
      attributeValue =
          type.cast(resolveLateBoundDefault(rule, attributeMap, attribute, ruleConfig));
    } else if (owningAspect == null && attributeMap.has(attribute.getName())) {
      // This condition is false for aspect attributes that do not give rise to dependencies because
      // attributes that come from aspects do not appear in attributeMap (see the comment in the
      // case that handles implicit attributes).
      attributeValue = attributeMap.get(attribute.getName(), type);
    }

    if (attributeValue == null) {
      return;
    }

    AttributeDependencyKind dependencyKind = createDependencyKind(attribute, owningAspect);
    type.visitLabels(
        (depLabel, ctx) -> outgoingLabels.put(dependencyKind, depLabel),
        attributeValue,
        /*context=*/ null);
  }

  private static AttributeDependencyKind createDependencyKind(
      Attribute attribute, @Nullable AspectClass owningAspect) {
    return owningAspect == null
        ? AttributeDependencyKind.forRule(attribute)
        : AttributeDependencyKind.forAspect(attribute, owningAspect);
  }

  @Nullable
  @VisibleForTesting(/* used to test LateBoundDefaults' default values */ )
  static <FragmentT> Object resolveLateBoundDefault(
      Rule rule, AttributeMap attributeMap, Attribute attribute, BuildConfigurationValue ruleConfig)
      throws InterruptedException {
    Preconditions.checkState(!attribute.getTransitionFactory().isSplit());
    @SuppressWarnings("unchecked")
    LateBoundDefault<FragmentT, ?> lateBoundDefault =
        (LateBoundDefault<FragmentT, ?>) attribute.getLateBoundDefault();

    Class<FragmentT> fragmentClass = lateBoundDefault.getFragmentClass();
    try {
      // TODO(b/65746853): remove this when nothing uses it anymore
      if (BuildConfigurationValue.class.equals(fragmentClass)
          // noconfig targets can't meaningfully parse late-bound defaults. See NoConfigTransition.
          && !ruleConfig.getOptions().hasNoConfig()) {
        return lateBoundDefault.resolve(rule, attributeMap, fragmentClass.cast(ruleConfig));
      }
      if (Void.class.equals(fragmentClass)) {
        return lateBoundDefault.resolve(
            rule, attributeMap, /* input= */ null
            /* analysisContext= */
            /* eventHandler= */ );
      }
      @SuppressWarnings("unchecked")
      FragmentT fragment =
          fragmentClass.cast(ruleConfig.getFragment((Class<? extends Fragment>) fragmentClass));
      if (fragment == null) {
        return null;
      }
      return lateBoundDefault.resolve(
          rule, attributeMap, fragment
          /* analysisContext= */
          /* eventHandler= */ );
    } catch (EvalException e) {
      // Materializers should not be called here and those are the only kind of late-bound defaults
      // that can throw these exceptions.
      throw new IllegalStateException(e);
    }
  }

  /**
   * Adds new dependencies to the given rule under the given attribute name
   *
   * @param attrName the name of the attribute to add dependency labels to
   * @param labels the dependencies to add
   */
  private static void addExplicitDeps(
      OrderedSetMultimap<DependencyKind, Label> outgoingLabels,
      Rule rule,
      String attrName,
      Iterable<Label> labels) {
    if (!rule.isAttrDefined(attrName, BuildType.LABEL_LIST)
        && !rule.isAttrDefined(attrName, BuildType.NODEP_LABEL_LIST)) {
      return;
    }
    Attribute attribute =
        rule.getRuleClassObject().getAttributeProvider().getAttributeByName(attrName);
    outgoingLabels.putAll(AttributeDependencyKind.forRule(attribute), labels);
  }

  private static void addVisibilityDepLabels(
      Iterable<Label> labels, OrderedSetMultimap<DependencyKind, Label> outgoingLabels) {
    outgoingLabels.putAll(VISIBILITY_DEPENDENCY, labels);
  }

  private static void addTransitiveVisibilityDepLabel(
      Label label, OrderedSetMultimap<DependencyKind, Label> outgoingLabels) {
    if (label != null) {
      outgoingLabels.put(TRANSITIVE_VISIBILITY_DEPENDENCY, label);
    }
  }
}
