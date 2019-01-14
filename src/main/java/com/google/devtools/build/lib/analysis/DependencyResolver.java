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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Resolver for dependencies between configured targets.
 *
 * <p>Includes logic to derive the right configurations depending on transition type.
 */
public abstract class DependencyResolver {
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
    OrderedSetMultimap<Attribute, Dependency> outgoingEdges = OrderedSetMultimap.create();
    if (target instanceof OutputFile) {
      Preconditions.checkNotNull(config);
      visitTargetVisibility(node, rootCauses, outgoingEdges.get(null));
      Rule rule = ((OutputFile) target).getGeneratingRule();
      outgoingEdges.put(null, Dependency.withConfiguration(rule.getLabel(), config));
    } else if (target instanceof InputFile) {
      visitTargetVisibility(node, rootCauses, outgoingEdges.get(null));
    } else if (target instanceof EnvironmentGroup) {
      visitTargetVisibility(node, rootCauses, outgoingEdges.get(null));
    } else if (target instanceof Rule) {
      visitRule(
          node,
          hostConfig,
          aspects,
          configConditions,
          toolchainLabels,
          rootCauses,
          outgoingEdges,
          trimmingTransitionFactory);
    } else if (target instanceof PackageGroup) {
      visitPackageGroup(node, (PackageGroup) target, rootCauses, outgoingEdges.get(null));
    } else {
      throw new IllegalStateException(target.getLabel().toString());
    }

    return outgoingEdges;
  }

  private void visitRule(
      TargetAndConfiguration node,
      BuildConfiguration hostConfig,
      Iterable<Aspect> aspects,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      ImmutableSet<Label> toolchainLabels,
      NestedSetBuilder<Cause> rootCauses,
      OrderedSetMultimap<Attribute, Dependency> outgoingEdges,
      @Nullable RuleTransitionFactory trimmingTransitionFactory)
      throws EvalException, InconsistentAspectOrderException, InterruptedException {
    Preconditions.checkArgument(node.getTarget() instanceof Rule, node);
    BuildConfiguration ruleConfig = Preconditions.checkNotNull(node.getConfiguration(), node);
    Rule rule = (Rule) node.getTarget();

    ConfiguredAttributeMapper attributeMap = ConfiguredAttributeMapper.of(rule, configConditions);
    attributeMap.validateAttributes();
    RuleResolver depResolver =
        new RuleResolver(
            rule,
            ruleConfig,
            aspects,
            attributeMap,
            rootCauses,
            outgoingEdges,
            trimmingTransitionFactory);

    visitTargetVisibility(node, rootCauses, outgoingEdges.get(null));
    resolveAttributes(depResolver, ruleConfig, hostConfig);

    // Add the rule's visibility labels (which may come from the rule or from package defaults).
    addExplicitDeps(depResolver, "visibility", rule.getVisibility().getDependencyLabels());

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
      addExplicitDeps(depResolver, RuleClass.COMPATIBLE_ENVIRONMENT_ATTR,
          rule.getPackage().getDefaultCompatibleWith());
    }
    if (!rule.isAttributeValueExplicitlySpecified(RuleClass.RESTRICTED_ENVIRONMENT_ATTR)) {
      addExplicitDeps(depResolver, RuleClass.RESTRICTED_ENVIRONMENT_ATTR,
          rule.getPackage().getDefaultRestrictedTo());
    }

    Attribute toolchainsAttribute =
        attributeMap.getAttributeDefinition(PlatformSemantics.RESOLVED_TOOLCHAINS_ATTR);
    resolveToolchainDependencies(outgoingEdges.get(toolchainsAttribute), toolchainLabels);
  }

  private void resolveAttributes(
      RuleResolver depResolver, BuildConfiguration ruleConfig, BuildConfiguration hostConfig)
      throws InterruptedException, InconsistentAspectOrderException {
    Rule rule = depResolver.rule;
    Label ruleLabel = rule.getLabel();
    ConfiguredAttributeMapper attributeMap = depResolver.attributeMap;
    ImmutableSet<String> mappedAttributes =
        ImmutableSet.copyOf(depResolver.attributeMap.getAttributeNames());
    Set<Label> labelsToFetch = new HashSet<>();
    Map<Label, Target> targetLookupResult = null;
    for (boolean collectingLabels : ImmutableList.of(Boolean.TRUE, Boolean.FALSE)) {
      for (AttributeAndOwner attributeAndOwner : depResolver.attributes) {
        Attribute attribute = attributeAndOwner.attribute;
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
                  ? depResolver.attributeMap.get(attribute.getName(), attribute.getType())
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

        if (collectingLabels) {
          labelsToFetch.addAll(labels);
        } else {
          for (Label label : labels) {
            Target target = targetLookupResult.get(label);
            if (target != null) {
              depResolver.registerEdge(attributeAndOwner, target);
            }
          }
        }
      }
      if (collectingLabels) {
        targetLookupResult =
            getTargets(labelsToFetch, rule, depResolver.rootCauses, labelsToFetch.size());
        if (targetLookupResult == null) {
          return;
        }
      }
    }
  }

  private void resolveToolchainDependencies(
      Set<Dependency> dependencies, ImmutableSet<Label> toolchainLabels) {
    for (Label label : toolchainLabels) {
      Dependency dependency =
          Dependency.withTransitionAndAspects(
              label, HostTransition.INSTANCE, AspectCollection.EMPTY);
      dependencies.add(dependency);
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
   * @param depResolver the resolver for this rule's deps
   * @param attrName the name of the attribute to add dependency labels to
   * @param labels the dependencies to add
   */
  private void addExplicitDeps(RuleResolver depResolver, String attrName, Collection<Label> labels)
      throws InterruptedException, InconsistentAspectOrderException {
    Rule rule = depResolver.rule;
    if (!rule.isAttrDefined(attrName, BuildType.LABEL_LIST)
        && !rule.isAttrDefined(attrName, BuildType.NODEP_LABEL_LIST)) {
      return;
    }
    Attribute attribute = rule.getRuleClassObject().getAttributeByName(attrName);
    Map<Label, Target> result = getTargets(labels, rule, depResolver.rootCauses, labels.size());
    if (result == null) {
      return;
    }
    AttributeAndOwner attributeAndOwner = new AttributeAndOwner(attribute);

    for (Target target : result.values()) {
      depResolver.registerEdge(attributeAndOwner, target);
    }
  }

  /**
   * Converts the given multimap of attributes to labels into a multi map of attributes to {@link
   * Dependency} objects using the proper configuration transition for each attribute.
   *
   * <p>Returns null if Skyframe dependencies are missing.
   *
   * @throws IllegalArgumentException if the {@code node} does not refer to a {@link Rule} instance
   */
  @Nullable
  public final Collection<Dependency> resolveRuleLabels(
      TargetAndConfiguration node,
      OrderedSetMultimap<Attribute, Label> depLabels,
      NestedSetBuilder<Cause> rootCauses,
      @Nullable RuleTransitionFactory trimmingTransitionFactory)
      throws InterruptedException, InconsistentAspectOrderException {
    Preconditions.checkArgument(node.getTarget() instanceof Rule);
    Rule rule = (Rule) node.getTarget();
    OrderedSetMultimap<Attribute, Dependency> outgoingEdges = OrderedSetMultimap.create();
    RuleResolver depResolver =
        new RuleResolver(
            rule,
            node.getConfiguration(),
            ImmutableList.<Aspect>of(),
            /*attributeMap=*/ null,
            rootCauses,
            outgoingEdges,
            trimmingTransitionFactory);
    Map<Label, Target> result = getTargets(depLabels.values(), rule, rootCauses, depLabels.size());
    if (result == null) {
      return null;
    }
    for (Map.Entry<Attribute, Collection<Label>> entry : depLabels.asMap().entrySet()) {
      AttributeAndOwner attributeAndOwner = new AttributeAndOwner(entry.getKey());
      for (Label depLabel : entry.getValue()) {
        Target target = result.get(depLabel);
        if (target != null) {
          depResolver.registerEdge(attributeAndOwner, target);
        }
      }
    }
    return outgoingEdges.values();
  }

  private void visitPackageGroup(
      TargetAndConfiguration node,
      PackageGroup packageGroup,
      NestedSetBuilder<Cause> rootCauses,
      Collection<Dependency> outgoingEdges)
      throws InterruptedException {
    List<Label> includes = packageGroup.getIncludes();
    Map<Label, Target> targetMap = getTargets(includes, packageGroup, rootCauses, includes.size());
    if (targetMap == null) {
      return;
    }
    Collection<Target> targets = targetMap.values();

    for (Target target : targets) {
      if (!(target instanceof PackageGroup)) {
        // Note that this error could also be caught in PackageGroupConfiguredTarget, but since
        // these have the null configuration, visiting the corresponding target would trigger an
        // analysis of a rule with a null configuration, which doesn't work.
        invalidPackageGroupReferenceHook(node, target.getLabel());
        continue;
      }

      outgoingEdges.add(Dependency.withNullConfiguration(target.getLabel()));
    }
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
      AttributeAndOwner attributeAndOwner,
      Rule target,
      ImmutableList.Builder<Aspect> filteredAspectPath,
      ImmutableSet.Builder<AspectDescriptor> visibleAspects) {

    Aspect lastAspect = null;
    for (Aspect aspect : aspectPath) {
      if (aspect.getAspectClass().equals(attributeAndOwner.ownerAspect)) {
        // Do not propagate over the aspect's own attributes.
        continue;
      }
      lastAspect = aspect;
      if (aspect.getDefinition().propagateAlong(attributeAndOwner.attribute)
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

  /**
   * Pair of (attribute, owner aspect if attribute is from an aspect).
   *
   * <p>For "plain" rule attributes, this wrapper class will have value (attribute, null).
   */
  final class AttributeAndOwner {
    final Attribute attribute;
    final @Nullable AspectClass ownerAspect;

    AttributeAndOwner(Attribute attribute) {
      this(attribute, null);
    }

    AttributeAndOwner(Attribute attribute, @Nullable AspectClass ownerAspect) {
      this.attribute = attribute;
      this.ownerAspect = ownerAspect;
    }
  }

  /**
   * Supplies the logic for translating <Attribute, Label> pairs for a rule into the
   * <Attribute, Dependency> pairs DependencyResolver ultimately returns.
   *
   * <p>The main difference between the two is that the latter applies configuration transitions,
   * i.e. it specifies not just which deps a rule has but also the configurations those deps
   * should take.
   */
  private class RuleResolver {
    private final Rule rule;
    private final BuildConfiguration ruleConfig;
    private final Iterable<Aspect> aspects;
    private final ConfiguredAttributeMapper attributeMap;
    private final NestedSetBuilder<Cause> rootCauses;
    private final OrderedSetMultimap<Attribute, Dependency> outgoingEdges;
    @Nullable private final RuleTransitionFactory trimmingTransitionFactory;
    private final List<AttributeAndOwner> attributes;

    /**
     * Constructs a new dependency resolver for the specified rule context.
     *
     * @param rule the rule being evaluated
     * @param ruleConfig the rule's configuration
     * @param aspects the aspects applied to this rule (if any)
     * @param attributeMap mapper for the rule's attribute values
     * @param rootCauses output collector for dep labels that can't be (loading phase) loaded
     * @param outgoingEdges output collector for the resolved dependencies
     */
    RuleResolver(
        Rule rule,
        BuildConfiguration ruleConfig,
        Iterable<Aspect> aspects,
        ConfiguredAttributeMapper attributeMap,
        NestedSetBuilder<Cause> rootCauses,
        OrderedSetMultimap<Attribute, Dependency> outgoingEdges,
        @Nullable RuleTransitionFactory trimmingTransitionFactory) {
      this.rule = rule;
      this.ruleConfig = ruleConfig;
      this.aspects = aspects;
      this.attributeMap = attributeMap;
      this.rootCauses = rootCauses;
      this.outgoingEdges = outgoingEdges;
      this.trimmingTransitionFactory = trimmingTransitionFactory;

      this.attributes =
          getAttributes(
              rule,
              // These are attributes that the application of `aspects` "path"
              // to the rule will see. Application of path is really the
              // application of the last aspect in the path, so we only let it see
              // it's own attributes.
              aspects);
    }

    /** Returns the attributes that should be visited for this rule/aspect combination. */
    private List<AttributeAndOwner> getAttributes(Rule rule, Iterable<Aspect> aspects) {
      ImmutableList.Builder<AttributeAndOwner> result = ImmutableList.builder();
      List<Attribute> ruleDefs = rule.getRuleClassObject().getAttributes();
      for (Attribute attribute : ruleDefs) {
        result.add(new AttributeAndOwner(attribute));
      }
      for (Aspect aspect : aspects) {
        for (Attribute attribute : aspect.getDefinition().getAttributes().values()) {
          result.add(new AttributeAndOwner(attribute, aspect.getAspectClass()));
        }
      }
      return result.build();
    }

    /**
     * Resolves the given dep for the given attribute, determining which configurations to apply to
     * it.
     */
    void registerEdge(AttributeAndOwner attributeAndOwner, Target toTarget)
        throws InconsistentAspectOrderException {
      ConfigurationTransition transition =
          TransitionResolver.evaluateTransition(
              ruleConfig,
              attributeAndOwner.attribute,
              toTarget,
              attributeMap,
              trimmingTransitionFactory);
      outgoingEdges.put(
          attributeAndOwner.attribute,
          transition == NullTransition.INSTANCE
              ? Dependency.withNullConfiguration(toTarget.getLabel())
              : Dependency.withTransitionAndAspects(
                  toTarget.getLabel(), transition, requiredAspects(attributeAndOwner, toTarget)));
    }

    private AspectCollection requiredAspects(AttributeAndOwner attributeAndOwner,
        final Target target) throws InconsistentAspectOrderException {
      if (!(target instanceof Rule)) {
        return AspectCollection.EMPTY;
      }


      ImmutableList.Builder<Aspect> filteredAspectPath = ImmutableList.builder();
      ImmutableSet.Builder<AspectDescriptor> visibleAspects = ImmutableSet.builder();

      if (attributeAndOwner.ownerAspect == null) {
        collectOriginatingAspects(
            rule, attributeAndOwner.attribute, (Rule) target, filteredAspectPath, visibleAspects);
      }

      collectPropagatingAspects(
          aspects, attributeAndOwner, (Rule) target, filteredAspectPath, visibleAspects);
      try {
        return AspectCollection.create(filteredAspectPath.build(), visibleAspects.build());
      } catch (AspectCycleOnPathException e) {
        throw new InconsistentAspectOrderException(rule, attributeAndOwner.attribute, target, e);
      }
    }
  }

  private void visitTargetVisibility(
      TargetAndConfiguration node,
      NestedSetBuilder<Cause> rootCauses,
      Collection<Dependency> outgoingEdges)
      throws InterruptedException {
    Target target = node.getTarget();
    List<Label> dependencyLabels = target.getVisibility().getDependencyLabels();
    Map<Label, Target> targetMap =
        getTargets(dependencyLabels, target, rootCauses, dependencyLabels.size());
    if (targetMap == null) {
      return;
    }
    Collection<Target> targets = targetMap.values();
    for (Target visibilityTarget : targets) {
      if (!(visibilityTarget instanceof PackageGroup)) {
        // Note that this error could also be caught in
        // AbstractConfiguredTarget.convertVisibility(), but we have an
        // opportunity here to avoid dependency cycles that result from
        // the visibility attribute of a rule referring to a rule that
        // depends on it (instead of its package)
        invalidVisibilityReferenceHook(node, visibilityTarget.getLabel());
        continue;
      }

      // Visibility always has null configuration
      outgoingEdges.add(Dependency.withNullConfiguration(visibilityTarget.getLabel()));
    }
  }

  /**
   * Hook for the error case when an invalid visibility reference is found.
   *
   * @param node the node with the visibility attribute
   * @param label the invalid visibility reference
   */
  protected abstract void invalidVisibilityReferenceHook(TargetAndConfiguration node, Label label);

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
      Iterable<Label> labels,
      Target fromTarget,
      NestedSetBuilder<Cause> rootCauses,
      int labelsSizeHint)
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
