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

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.AspectCollection.AspectCycleOnPathException;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.config.TransitionResolver;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NullTransition;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.LateBoundDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.EnvironmentGroup;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleTransitionFactory;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * Resolver for dependencies between configured targets.
 *
 * <p>Includes logic to derive the right configurations depending on transition type.
 */
public abstract class DependencyResolver {

  /**
   * A kind of dependency.
   *
   * <p>Usually an attribute, but other special-cased kinds exist, for example, for visibility or
   * toolchains.
   */
  private interface DependencyKind {

    /**
     * The attribute through which a dependency arises.
     *
     * <p>Should only be called for dependency kinds representing an attribute.
     */
    @Nullable
    Attribute getAttribute();

    /**
     * The aspect owning the attribute through which the dependency arises.
     *
     * <p>Should only be called for dependency kinds representing an attribute.
     */
    @Nullable
    AspectClass getOwningAspect();
  }

  /** A dependency caused by something that's not an attribute. Special cases enumerated below. */
  private static final class NonAttributeDependencyKind implements DependencyKind {
    private NonAttributeDependencyKind() {}

    @Override
    public Attribute getAttribute() {
      throw new IllegalStateException();
    }

    @Nullable
    @Override
    public AspectClass getOwningAspect() {
      throw new IllegalStateException();
    }
  }

  /** A dependency for visibility. */
  private static final DependencyKind VISIBILITY_DEPENDENCY = new NonAttributeDependencyKind();

  /** The dependency on the rule that creates a given output file. */
  private static final DependencyKind OUTPUT_FILE_RULE_DEPENDENCY =
      new NonAttributeDependencyKind();

  /** A dependency on a resolved toolchain. */
  private static final DependencyKind TOOLCHAIN_DEPENDENCY = new NonAttributeDependencyKind();

  /** A dependency through an attribute, either that of an aspect or the rule itself. */
  @AutoValue
  abstract static class AttributeDependencyKind implements DependencyKind {
    @Override
    public abstract Attribute getAttribute();

    @Override
    @Nullable
    public abstract AspectClass getOwningAspect();

    private static AttributeDependencyKind forRule(Attribute attribute) {
      return new AutoValue_DependencyResolver_AttributeDependencyKind(attribute, null);
    }

    public static AttributeDependencyKind forAspect(Attribute attribute, AspectClass owningAspect) {
      return new AutoValue_DependencyResolver_AttributeDependencyKind(
          attribute, Preconditions.checkNotNull(owningAspect));
    }
  }

  /**
   * Returns ids for dependent nodes of a given node, sorted by attribute. Note that some
   * dependencies do not have a corresponding attribute here, and we use the null attribute to
   * represent those edges.
   *
   * <p>If {@code aspect} is null, returns the dependent nodes of the configured target node
   * representing the given target and configuration, otherwise that of the aspect node accompanying
   * the aforementioned configured target node for the specified aspect.
   *
   * <p>The values are not simply labels because this also implements the first step of applying
   * configuration transitions, namely, split transitions. This needs to be done before the labels
   * are resolved because late bound attributes depend on the configuration. A good example for this
   * is @{code :cc_toolchain}.
   *
   * <p>The long-term goal is that most configuration transitions be applied here. However, in order
   * to do that, we first have to eliminate transitions that depend on the rule class of the
   * dependency.
   *
   * @param node the target/configuration being evaluated
   * @param hostConfig the configuration this target would use if it was evaluated as a host tool.
   *     This is needed to support {@link LateBoundDefault#useHostConfiguration()}.
   * @param aspect the aspect applied to this target (if any)
   * @param configConditions resolver for config_setting labels
   * @param toolchainLabels required toolchain labels
   * @param trimmingTransitionFactory the transition factory used to trim rules (note: this is a
   *     temporary feature; see the corresponding methods in ConfiguredRuleClassProvider)
   * @return a mapping of each attribute in this rule or aspects to its dependent nodes
   */
  public final OrderedSetMultimap<Attribute, Dependency> dependentNodeMap(
      TargetAndConfiguration node,
      BuildConfiguration hostConfig,
      @Nullable Aspect aspect,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      ImmutableSet<Label> toolchainLabels,
      @Nullable RuleTransitionFactory trimmingTransitionFactory)
      throws EvalException, InterruptedException, InconsistentAspectOrderException {
    NestedSetBuilder<Cause> rootCauses = NestedSetBuilder.stableOrder();
    OrderedSetMultimap<Attribute, Dependency> outgoingEdges =
        dependentNodeMap(
            node,
            hostConfig,
            aspect != null ? ImmutableList.of(aspect) : ImmutableList.<Aspect>of(),
            configConditions,
            toolchainLabels,
            rootCauses,
            trimmingTransitionFactory);
    if (!rootCauses.isEmpty()) {
      throw new IllegalStateException(rootCauses.build().iterator().next().toString());
    }
    return outgoingEdges;
  }

  /**
   * Returns ids for dependent nodes of a given node, sorted by attribute. Note that some
   * dependencies do not have a corresponding attribute here, and we use the null attribute to
   * represent those edges.
   *
   * <p>If {@code aspects} is empty, returns the dependent nodes of the configured target node
   * representing the given target and configuration.
   *
   * <p>Otherwise {@code aspects} represents an aspect path. The function returns dependent nodes of
   * the entire path applied to given target and configuration. These are the depenent nodes of the
   * last aspect in the path.
   *
   * <p>This also implements the first step of applying configuration transitions, namely, split
   * transitions. This needs to be done before the labels are resolved because late bound attributes
   * depend on the configuration. A good example for this is @{code :cc_toolchain}.
   *
   * <p>The long-term goal is that most configuration transitions be applied here. However, in order
   * to do that, we first have to eliminate transitions that depend on the rule class of the
   * dependency.
   *
   * @param node the target/configuration being evaluated
   * @param hostConfig the configuration this target would use if it was evaluated as a host tool.
   *     This is needed to support {@link LateBoundDefault#useHostConfiguration()}.
   * @param aspects the aspects applied to this target (if any)
   * @param configConditions resolver for config_setting labels
   * @param toolchainLabels required toolchain labels
   * @param trimmingTransitionFactory the transition factory used to trim rules (note: this is a
   *     temporary feature; see the corresponding methods in ConfiguredRuleClassProvider)
   * @param rootCauses collector for dep labels that can't be (loading phase) loaded
   * @return a mapping of each attribute in this rule or aspects to its dependent nodes
   */
  public final OrderedSetMultimap<Attribute, Dependency> dependentNodeMap(
      TargetAndConfiguration node,
      BuildConfiguration hostConfig,
      Iterable<Aspect> aspects,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      ImmutableSet<Label> toolchainLabels,
      NestedSetBuilder<Cause> rootCauses,
      @Nullable RuleTransitionFactory trimmingTransitionFactory)
      throws EvalException, InterruptedException, InconsistentAspectOrderException {
    Target target = node.getTarget();
    BuildConfiguration config = node.getConfiguration();
    OrderedSetMultimap<DependencyKind, Label> outgoingLabels = OrderedSetMultimap.create();

    if (target instanceof OutputFile) {
      Preconditions.checkNotNull(config);
      visitTargetVisibility(node, outgoingLabels);
      Rule rule = ((OutputFile) target).getGeneratingRule();
      outgoingLabels.put(OUTPUT_FILE_RULE_DEPENDENCY, rule.getLabel());
    } else if (target instanceof InputFile) {
      visitTargetVisibility(node, outgoingLabels);
    } else if (target instanceof EnvironmentGroup) {
      visitTargetVisibility(node, outgoingLabels);
    } else if (target instanceof Rule) {
      visitRule(node, hostConfig, aspects, configConditions, toolchainLabels, outgoingLabels);
    } else if (target instanceof PackageGroup) {
      outgoingLabels.putAll(VISIBILITY_DEPENDENCY, ((PackageGroup) target).getIncludes());
    } else {
      throw new IllegalStateException(target.getLabel().toString());
    }

    OrderedSetMultimap<Attribute, Dependency> outgoingEdges = OrderedSetMultimap.create();

    List<Label> dependencyLabels =
        outgoingLabels.entries().stream()
            // Toolchains are resolved separately, so we don't need to depend on their packages.
            // It doesn't cause diminished functionality (after all, we depend on a package that
            // must have been loaded), but this makse the error message reporting a missing
            // toolchain a bit better.
            .filter(e -> e.getKey() != TOOLCHAIN_DEPENDENCY)
            .map(e -> e.getValue())
            .distinct()
            .collect(Collectors.toList());

    Map<Label, Target> targetMap = getTargets(dependencyLabels, target, rootCauses);
    if (targetMap == null) {
      // Dependencies could not be resolved. Try again when they are loaded by Skyframe.
      return outgoingEdges;
    }

    Rule fromRule = target instanceof Rule ? (Rule) target : null;
    ConfiguredAttributeMapper attributeMap =
        fromRule == null ? null : ConfiguredAttributeMapper.of(fromRule, configConditions);

    for (Map.Entry<DependencyKind, Label> entry : outgoingLabels.entries()) {
      Label toLabel = entry.getValue();

      if (entry.getKey() == TOOLCHAIN_DEPENDENCY) {
        // This dependency is a toolchain. Its package has not been loaded and therefore we can't
        // determine which aspects and which rule configuration transition we should use, so just
        // use sensible defaults. Not depending on their package makes the error message reporting
        // a missing toolchain a bit better.
        // TODO(lberki): This special-casing is weird. Find a better way to depend on toolchains.
        Attribute toolchainsAttribute =
            attributeMap.getAttributeDefinition(PlatformSemantics.RESOLVED_TOOLCHAINS_ATTR);
        outgoingEdges.put(
            toolchainsAttribute,
            Dependency.withTransitionAndAspects(
                toLabel,
                // TODO(jcater): Replace this with a proper transition for the execution platform.
                HostTransition.INSTANCE,
                AspectCollection.EMPTY));
        continue;
      }

      Target toTarget = targetMap.get(toLabel);

      if (toTarget == null) {
        // Dependency pointing to non-existent target. This error was reported above, so we can just
        // ignore this dependency. Toolchain dependencies always have toTarget == null since we do
        // not depend on their package.
        continue;
      }

      if (entry.getKey() == VISIBILITY_DEPENDENCY) {
        if (toTarget instanceof PackageGroup) {
          outgoingEdges.put(null, Dependency.withNullConfiguration(toLabel));
        } else {
          // Note that this error could also be caught in
          // AbstractConfiguredTarget.convertVisibility(), but we have an
          // opportunity here to avoid dependency cycles that result from
          // the visibility attribute of a rule referring to a rule that
          // depends on it (instead of its package)
          invalidPackageGroupReferenceHook(node, toLabel);
        }

        continue;
      }

      if (entry.getKey() == OUTPUT_FILE_RULE_DEPENDENCY) {
        outgoingEdges.put(
            null,
            Dependency.withTransitionAndAspects(
                toLabel, NoTransition.INSTANCE, AspectCollection.EMPTY));
        continue;
      }

      Attribute attribute = entry.getKey().getAttribute();
      AspectClass ownerAspect = entry.getKey().getOwningAspect();

      ConfigurationTransition attributeTransition =
          attribute.hasSplitConfigurationTransition()
              ? attribute.getSplitTransition(attributeMap)
              : attribute.getConfigurationTransition();

      ConfigurationTransition transition =
          TransitionResolver.evaluateTransition(
              node.getConfiguration(), attributeTransition, toTarget, trimmingTransitionFactory);
      AspectCollection requiredAspects =
          requiredAspects(fromRule, aspects, attribute, ownerAspect, toTarget);
      outgoingEdges.put(
          attribute,
          transition == NullTransition.INSTANCE
              ? Dependency.withNullConfiguration(toLabel)
              : Dependency.withTransitionAndAspects(entry.getValue(), transition, requiredAspects));
    }

    return outgoingEdges;
  }

  private void visitRule(
      TargetAndConfiguration node,
      BuildConfiguration hostConfig,
      Iterable<Aspect> aspects,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      ImmutableSet<Label> toolchainLabels,
      OrderedSetMultimap<DependencyKind, Label> outgoingLabels)
      throws EvalException {
    Preconditions.checkArgument(node.getTarget() instanceof Rule, node);
    BuildConfiguration ruleConfig = Preconditions.checkNotNull(node.getConfiguration(), node);
    Rule rule = (Rule) node.getTarget();

    ConfiguredAttributeMapper attributeMap = ConfiguredAttributeMapper.of(rule, configConditions);
    attributeMap.validateAttributes();

    visitTargetVisibility(node, outgoingLabels);
    resolveAttributes(outgoingLabels, rule, attributeMap, aspects, ruleConfig, hostConfig);

    // Add the rule's visibility labels (which may come from the rule or from package defaults).
    addExplicitDeps(outgoingLabels, rule, "visibility", rule.getVisibility().getDependencyLabels());

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
          rule.getPackage().getDefaultCompatibleWith());
    }
    if (!rule.isAttributeValueExplicitlySpecified(RuleClass.RESTRICTED_ENVIRONMENT_ATTR)) {
      addExplicitDeps(
          outgoingLabels,
          rule,
          RuleClass.RESTRICTED_ENVIRONMENT_ATTR,
          rule.getPackage().getDefaultRestrictedTo());
    }

    outgoingLabels.putAll(TOOLCHAIN_DEPENDENCY, toolchainLabels);
  }

  private void resolveAttributes(
      OrderedSetMultimap<DependencyKind, Label> outgoingLabels,
      Rule rule,
      ConfiguredAttributeMapper attributeMap,
      Iterable<Aspect> aspects,
      BuildConfiguration ruleConfig,
      BuildConfiguration hostConfig) {
    Label ruleLabel = rule.getLabel();
    ImmutableSet<String> mappedAttributes = ImmutableSet.copyOf(attributeMap.getAttributeNames());
    for (AttributeDependencyKind dependencyKind : getAttributes(rule, aspects)) {
      Attribute attribute = dependencyKind.getAttribute();
      if (!attribute.getCondition().apply(attributeMap)) {
        continue;
      }

      if (attribute.getType() == BuildType.OUTPUT
          || attribute.getType() == BuildType.OUTPUT_LIST
          || attribute.getType() == BuildType.NODEP_LABEL
          || attribute.getType() == BuildType.NODEP_LABEL_LIST) {
        // These types invoke visitLabels() so that they are reported in "bazel query" but do not
        // create a dependency. Maybe it's better to remove that, but then the labels() query
        // function would need to be rethought.
        continue;
      }

      Object attributeValue;
      if (attribute.isImplicit()) {
        // Since the attributes that come from aspects do not appear in attributeMap, we have to
        // get their values from somewhere else. This incidentally means that aspects attributes
        // are not configurable. It would be nice if that wasn't the case, but we'd have to revamp
        // how attribute mapping works, which is a large chunk of work.
        attributeValue =
            mappedAttributes.contains(attribute.getName())
                ? attributeMap.get(attribute.getName(), attribute.getType())
                : attribute.getDefaultValue(rule);
      } else if (attribute.isLateBound()) {
        attributeValue =
            resolveLateBoundDefault(rule, attributeMap, attribute, ruleConfig, hostConfig);
      } else if (attributeMap.has(attribute.getName())) {
        // This condition is false for aspect attributes that do not give rise to dependencies
        // because attributes that come from aspects do not appear in attributeMap (see the
        // comment in the case that handles implicit attributes)
        attributeValue = attributeMap.get(attribute.getName(), attribute.getType());
      } else {
        continue;
      }

      if (attributeValue == null) {
        continue;
      }

      List<Label> labels = new ArrayList<>();
      attribute
          .getType()
          .visitLabels(
              (depLabel, ctx) -> {
                labels.add(ruleLabel.resolveRepositoryRelative(depLabel));
              },
              attributeValue,
              null);

      outgoingLabels.putAll(dependencyKind, labels);
    }
  }

  @VisibleForTesting(/* used to test LateBoundDefaults' default values */ )
  public static <FragmentT> Object resolveLateBoundDefault(
      Rule rule,
      AttributeMap attributeMap,
      Attribute attribute,
      BuildConfiguration ruleConfig,
      BuildConfiguration hostConfig) {
    Preconditions.checkState(!attribute.hasSplitConfigurationTransition());
    @SuppressWarnings("unchecked")
    LateBoundDefault<FragmentT, ?> lateBoundDefault =
        (LateBoundDefault<FragmentT, ?>) attribute.getLateBoundDefault();
    BuildConfiguration attributeConfig =
        lateBoundDefault.useHostConfiguration() ? hostConfig : ruleConfig;

    Class<FragmentT> fragmentClass = lateBoundDefault.getFragmentClass();
    // TODO(b/65746853): remove this when nothing uses it anymore
    if (BuildConfiguration.class.equals(fragmentClass)) {
      return lateBoundDefault.resolve(rule, attributeMap, fragmentClass.cast(attributeConfig));
    }
    if (Void.class.equals(fragmentClass)) {
      return lateBoundDefault.resolve(rule, attributeMap, null);

    }
    @SuppressWarnings("unchecked")
    FragmentT fragment =
        fragmentClass.cast(
            attributeConfig.getFragment(
                (Class<? extends BuildConfiguration.Fragment>) fragmentClass));
    if (fragment == null) {
      return null;
    }
    return lateBoundDefault.resolve(rule, attributeMap, fragment);
  }

  /**
   * Adds new dependencies to the given rule under the given attribute name
   *
   * @param attrName the name of the attribute to add dependency labels to
   * @param labels the dependencies to add
   */
  private void addExplicitDeps(
      OrderedSetMultimap<DependencyKind, Label> outgoingLabels,
      Rule rule,
      String attrName,
      Collection<Label> labels) {
    if (!rule.isAttrDefined(attrName, BuildType.LABEL_LIST)
        && !rule.isAttrDefined(attrName, BuildType.NODEP_LABEL_LIST)) {
      return;
    }
    Attribute attribute = rule.getRuleClassObject().getAttributeByName(attrName);
    outgoingLabels.putAll(AttributeDependencyKind.forRule(attribute), labels);
  }

  /**
   * Collects into {@code filteredAspectPath} aspects from {@code aspectPath} that propagate along
   * {@code attributeAndOwner} and apply to a given {@code target}.
   *
   * <p>The last aspect in {@code aspectPath} is (potentially) visible and recorded in {@code
   * visibleAspects}.
   */
  private static void collectPropagatingAspects(
      Iterable<Aspect> aspectPath,
      Attribute attribute,
      @Nullable AspectClass aspectOwningAttribute,
      Rule target,
      ImmutableList.Builder<Aspect> filteredAspectPath,
      ImmutableSet.Builder<AspectDescriptor> visibleAspects) {

    Aspect lastAspect = null;
    for (Aspect aspect : aspectPath) {
      if (aspect.getAspectClass().equals(aspectOwningAttribute)) {
        // Do not propagate over the aspect's own attributes.
        continue;
      }
      lastAspect = aspect;
      if (aspect.getDefinition().propagateAlong(attribute)
          && aspect
              .getDefinition()
              .getRequiredProviders()
              .isSatisfiedBy(target.getRuleClassObject().getAdvertisedProviders())) {
        filteredAspectPath.add(aspect);
      } else {
        lastAspect = null;
      }
    }

    if (lastAspect != null) {
      visibleAspects.add(lastAspect.getDescriptor());
    }
  }

  /**
   * Collect all aspects that originate on {@code attribute} of {@code originalRule}
   * and are applicable to a {@code target}
   *
   * They are appended to {@code filteredAspectPath} and registered in {@code visibleAspects} set.
   */
  private static void collectOriginatingAspects(
      Rule originalRule, Attribute attribute, Rule target,
      ImmutableList.Builder<Aspect> filteredAspectPath,
      ImmutableSet.Builder<AspectDescriptor> visibleAspects) {
    ImmutableList<Aspect> baseAspects = attribute.getAspects(originalRule);
    RuleClass ruleClass = target.getRuleClassObject();
    for (Aspect baseAspect : baseAspects) {
      if (baseAspect.getDefinition().getRequiredProviders()
          .isSatisfiedBy(ruleClass.getAdvertisedProviders())) {
        filteredAspectPath.add(baseAspect);
        visibleAspects.add(baseAspect.getDescriptor());
      }
    }
  }

  /** Returns the attributes that should be visited for this rule/aspect combination. */
  private List<AttributeDependencyKind> getAttributes(Rule rule, Iterable<Aspect> aspects) {
    ImmutableList.Builder<AttributeDependencyKind> result = ImmutableList.builder();
    List<Attribute> ruleDefs = rule.getRuleClassObject().getAttributes();
    for (Attribute attribute : ruleDefs) {
      result.add(AttributeDependencyKind.forRule(attribute));
    }
    for (Aspect aspect : aspects) {
      for (Attribute attribute : aspect.getDefinition().getAttributes().values()) {
        result.add(AttributeDependencyKind.forAspect(attribute, aspect.getAspectClass()));
      }
    }
    return result.build();
  }

  private AspectCollection requiredAspects(
      Rule fromRule,
      Iterable<Aspect> aspects,
      Attribute attribute,
      AspectClass ownerAspect,
      Target toTarget)
      throws InconsistentAspectOrderException {
    if (!(toTarget instanceof Rule)) {
      return AspectCollection.EMPTY;
    }

    ImmutableList.Builder<Aspect> filteredAspectPath = ImmutableList.builder();
    ImmutableSet.Builder<AspectDescriptor> visibleAspects = ImmutableSet.builder();

    if (ownerAspect == null) {
      collectOriginatingAspects(
          fromRule, attribute, (Rule) toTarget, filteredAspectPath, visibleAspects);
    }

    collectPropagatingAspects(
        aspects, attribute, ownerAspect, (Rule) toTarget, filteredAspectPath, visibleAspects);
    try {
      return AspectCollection.create(filteredAspectPath.build(), visibleAspects.build());
    } catch (AspectCycleOnPathException e) {
      throw new InconsistentAspectOrderException(fromRule, attribute, toTarget, e);
    }
  }

  private void visitTargetVisibility(
      TargetAndConfiguration node, OrderedSetMultimap<DependencyKind, Label> outgoingLabels) {
    Target target = node.getTarget();
    outgoingLabels.putAll(VISIBILITY_DEPENDENCY, target.getVisibility().getDependencyLabels());
  }

  /**
   * Hook for the error case when an invalid package group reference is found.
   *
   * @param node the package group node with the includes attribute
   * @param label the invalid reference
   */
  protected abstract void invalidPackageGroupReferenceHook(TargetAndConfiguration node,
      Label label);

  /**
   * Returns the targets for the given labels.
   *
   * <p>Returns null if any targets are not ready to be returned at this moment because of missing
   * Skyframe dependencies. If getTargets returns null once or more during a {@link
   * #dependentNodeMap} call, the results of that call will be incomplete. As is usual in these
   * situation, the caller must return control to Skyframe and wait for the SkyFunction to be
   * restarted, at which point the requested dependencies will be available.
   */
  protected abstract Map<Label, Target> getTargets(
      Collection<Label> labels, Target fromTarget, NestedSetBuilder<Cause> rootCauses)
      throws InterruptedException;

  /**
   * Signals an inconsistency on aspect path: an aspect occurs twice on the path and
   * the second occurrence sees a different set of aspects.
   *
   * {@see AspectCycleOnPathException}
   */
  public class InconsistentAspectOrderException extends Exception {
    private final Location location;
    public InconsistentAspectOrderException(Rule originalRule, Attribute attribute, Target target,
        AspectCycleOnPathException e) {
      super(String.format("%s (when propagating from %s to %s via attribute %s)",
          e.getMessage(),
          originalRule.getLabel(),
          target.getLabel(),
          attribute.getName()));
      this.location = originalRule.getLocation();
    }

    public Location getLocation() {
      return location;
    }
  }
}
