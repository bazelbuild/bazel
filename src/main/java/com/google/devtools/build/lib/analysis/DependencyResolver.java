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
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.DynamicTransitionMapper;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.TransitionResolver;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransitionProxy;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.analysis.config.transitions.Transition;
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
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import java.util.ArrayList;
import java.util.Collection;
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
  private final TransitionResolver transitionResolver;

  protected DependencyResolver(DynamicTransitionMapper transitionMapper) {
    this.transitionResolver = new TransitionResolver(transitionMapper);
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
   * @param toolchainContext {@link ToolchainContext} for this target
   * @return a mapping of each attribute in this rule or aspects to its dependent nodes
   */
  public final OrderedSetMultimap<Attribute, Dependency> dependentNodeMap(
      TargetAndConfiguration node,
      BuildConfiguration hostConfig,
      @Nullable Aspect aspect,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      ToolchainContext toolchainContext)
      throws EvalException, InvalidConfigurationException, InterruptedException,
          InconsistentAspectOrderException {
    NestedSetBuilder<Label> rootCauses = NestedSetBuilder.<Label>stableOrder();
    OrderedSetMultimap<Attribute, Dependency> outgoingEdges =
        dependentNodeMap(
            node,
            hostConfig,
            aspect != null ? ImmutableList.of(aspect) : ImmutableList.<Aspect>of(),
            configConditions,
            toolchainContext,
            rootCauses);
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
   * @param toolchainContext context information for required toolchains
   * @param rootCauses collector for dep labels that can't be (loading phase) loaded @return a
   *     mapping of each attribute in this rule or aspects to its dependent nodes
   */
  public final OrderedSetMultimap<Attribute, Dependency> dependentNodeMap(
      TargetAndConfiguration node,
      BuildConfiguration hostConfig,
      Iterable<Aspect> aspects,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      @Nullable ToolchainContext toolchainContext,
      NestedSetBuilder<Label> rootCauses)
      throws EvalException, InvalidConfigurationException, InterruptedException,
          InconsistentAspectOrderException {
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
          node, hostConfig, aspects, configConditions, toolchainContext, rootCauses, outgoingEdges);
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
      @Nullable ToolchainContext toolchainContext,
      NestedSetBuilder<Label> rootCauses,
      OrderedSetMultimap<Attribute, Dependency> outgoingEdges)
      throws EvalException, InvalidConfigurationException, InconsistentAspectOrderException,
          InterruptedException {
    Preconditions.checkArgument(node.getTarget() instanceof Rule);
    BuildConfiguration ruleConfig = Preconditions.checkNotNull(node.getConfiguration());
    Rule rule = (Rule) node.getTarget();

    ConfiguredAttributeMapper attributeMap = ConfiguredAttributeMapper.of(rule, configConditions);
    attributeMap.validateAttributes();
    RuleResolver depResolver =
        new RuleResolver(rule, ruleConfig, aspects, attributeMap, rootCauses, outgoingEdges);

    visitTargetVisibility(node, rootCauses, outgoingEdges.get(null));
    resolveEarlyBoundAttributes(depResolver);
    resolveLateBoundAttributes(depResolver, ruleConfig, hostConfig);

    if (toolchainContext != null) {
      Attribute toolchainsAttribute =
          attributeMap.getAttributeDefinition(PlatformSemantics.TOOLCHAINS_ATTR);
      resolveToolchainDependencies(outgoingEdges.get(toolchainsAttribute), toolchainContext);
    }
  }

  /**
   * Resolves the dependencies for all attributes in this rule except late-bound attributes
   * (which require special processing: see {@link #resolveLateBoundAttributes}).
   */
  private void resolveEarlyBoundAttributes(RuleResolver depResolver)
      throws EvalException, InterruptedException, InconsistentAspectOrderException {
    Rule rule = depResolver.rule;

    resolveExplicitAttributes(depResolver);
    resolveImplicitAttributes(depResolver);

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
  }

  private void resolveExplicitAttributes(final RuleResolver depResolver)
      throws InterruptedException, InconsistentAspectOrderException {

    // Record error that might happen during label visitation.
    final InconsistentAspectOrderException[] exception = new InconsistentAspectOrderException[1];

    depResolver.attributeMap.visitLabels(
        new AttributeMap.AcceptsLabelAttribute() {
          @Override
          public void acceptLabelAttribute(Label label, Attribute attribute)
              throws InterruptedException {
            if (attribute.getType() == BuildType.NODEP_LABEL
                || attribute.isImplicit()
                || attribute.isLateBound()) {
              return;
            }
            try {
              depResolver.resolveDep(new AttributeAndOwner(attribute), label);
            } catch (InconsistentAspectOrderException e) {
              if (exception[0] == null) {
                exception[0] = e;
              }
            }
          }
        });

    if (exception[0] != null) {
      throw exception[0];
    }
  }

  /** Resolves the dependencies for all implicit attributes in this rule. */
  private void resolveImplicitAttributes(RuleResolver depResolver)
      throws InterruptedException, InconsistentAspectOrderException {
    // Since the attributes that come from aspects do not appear in attributeMap, we have to get
    // their values from somewhere else. This incidentally means that aspects attributes are not
    // configurable. It would be nice if that wasn't the case, but we'd have to revamp how
    // attribute mapping works, which is a large chunk of work.
    Rule rule = depResolver.rule;
    Label ruleLabel = rule.getLabel();
    ConfiguredAttributeMapper attributeMap = depResolver.attributeMap;
    ImmutableSet<String> mappedAttributes = ImmutableSet.copyOf(attributeMap.getAttributeNames());
    for (AttributeAndOwner attributeAndOwner : depResolver.attributes) {
      Attribute attribute = attributeAndOwner.attribute;
      if (!attribute.isImplicit() || !attribute.getCondition().apply(attributeMap)) {
        continue;
      }

      if (attribute.getType() == BuildType.LABEL) {
        Label label = mappedAttributes.contains(attribute.getName())
            ? attributeMap.get(attribute.getName(), BuildType.LABEL)
            : BuildType.LABEL.cast(attribute.getDefaultValue(rule));

        if (label != null) {
          label = ruleLabel.resolveRepositoryRelative(label);
          depResolver.resolveDep(attributeAndOwner, label);
        }
      } else if (attribute.getType() == BuildType.LABEL_LIST) {
        List<Label> labelList;
        if (mappedAttributes.contains(attribute.getName())) {
          labelList = new ArrayList<>();
          for (Label label : attributeMap.get(attribute.getName(), BuildType.LABEL_LIST)) {
            labelList.add(label);
          }
        } else {
          labelList = BuildType.LABEL_LIST.cast(attribute.getDefaultValue(rule));
        }
        for (Label label : labelList) {
          depResolver.resolveDep(attributeAndOwner, ruleLabel.resolveRepositoryRelative(label));
        }
      }
    }
  }

  /**
   * Resolves the dependencies for all late-bound attributes in this rule.
   *
   * <p>Late-bound attributes need special handling because they require configuration
   * transitions to determine their values.
   *
   * <p>In other words, the normal process of dependency resolution is:
   * <ol>
   *   <li>Find every label value in the rule's attributes</li>
   *   <li>Apply configuration transitions over each value to get its dep configuration
   *   <li>Return each value with its dep configuration</li>
   * </ol>
   *
   * This doesn't work for late-bound attributes because you can't get their values without
   * knowing the configuration first. And that configuration may not be the owning rule's
   * configuration. Specifically, {@link LateBoundDefault#useHostConfiguration()} switches to the
   * host config and late-bound split attributes branch into multiple split configs.
   *
   * <p>This method implements that logic and makes sure the normal configuration
   * transition logic mixes with it cleanly.
   *
   * @param depResolver the resolver for this rule's deps
   * @param ruleConfig the rule's configuration
   * @param hostConfig the equivalent host configuration
   */
  private void resolveLateBoundAttributes(
      RuleResolver depResolver,
      BuildConfiguration ruleConfig,
      BuildConfiguration hostConfig)
      throws EvalException, InvalidConfigurationException, InconsistentAspectOrderException,
             InterruptedException{
    ConfiguredAttributeMapper attributeMap = depResolver.attributeMap;
    for (AttributeAndOwner attributeAndOwner : depResolver.attributes) {
      Attribute attribute = attributeAndOwner.attribute;
      if (!attribute.isLateBound() || !attribute.getCondition().apply(attributeMap)) {
        continue;
      }

      LateBoundDefault<?, ?> lateBoundDefault = attribute.getLateBoundDefault();

      Collection<BuildOptions> splitOptions = getSplitOptions(attributeMap, attribute, ruleConfig);
      if (!splitOptions.isEmpty() && !ruleConfig.isHostConfiguration()) {
        // Late-bound attribute with a split transition:
        // Since we want to get the same results as TransitionResolver.evaluateTransition (but
        // skip it since we've already applied the split), we want to make sure this logic
        // doesn't do anything differently. TransitionResolver.evaluateTransition has additional
        // logic for host configs. So when we're in the host configuration we fall back to the
        // non-split branch, which calls TransitionResolver.evaluateTransition, which returns its
        // "host mode" result without ever looking at the split.
        Iterable<BuildConfiguration> splitConfigs =
            getConfigurations(ruleConfig.fragmentClasses(), splitOptions);
        if (splitConfigs == null) {
          continue; // Need Skyframe deps.
        }
        for (BuildConfiguration splitConfig : splitConfigs) {
          for (Label dep : resolveLateBoundAttribute(depResolver.rule, attribute,
              lateBoundDefault.useHostConfiguration() ? hostConfig : splitConfig, attributeMap)) {
            // Skip the normal config transition pipeline and directly feed the split config. This
            // is because the split already had to be applied to determine the attribute's value.
            // This makes the split logic in the normal pipeline redundant and potentially
            // incorrect.
            depResolver.resolveDep(attributeAndOwner, dep, splitConfig);
          }
        }
      } else {
        // Late-bound attribute without a split transition:
        for (Label dep : resolveLateBoundAttribute(depResolver.rule, attribute,
            lateBoundDefault.useHostConfiguration() ? hostConfig : ruleConfig, attributeMap)) {
          // Process this dep like a normal attribute.
          depResolver.resolveDep(attributeAndOwner, dep);
        }
      }
    }
  }

  private void resolveToolchainDependencies(
      Set<Dependency> dependencies, ToolchainContext toolchainContext) {
    for (Label label : toolchainContext.getResolvedToolchainLabels()) {
      Dependency dependency =
          Dependency.withTransitionAndAspects(
              label, HostTransition.INSTANCE, AspectCollection.EMPTY);
      dependencies.add(dependency);
    }
  }

  /**
   * Returns the BuildOptions if the rule's attribute triggers a split in this configuration, or
   * the empty collection if the attribute does not trigger a split transition or if the split
   * transition does not apply.
   *
   * <p>Even though the attribute may have a split, splits don't have to apply in every
   * configuration (see {@link SplitTransition#split}).
   */
  private static Collection<BuildOptions> getSplitOptions(ConfiguredAttributeMapper attributeMap,
      Attribute attribute,
      BuildConfiguration ruleConfig) {
    if (!attribute.hasSplitConfigurationTransition()) {
      return ImmutableList.<BuildOptions>of();
    }
    SplitTransition transition = attribute.getSplitTransition(attributeMap);
    return transition.split(ruleConfig.getOptions());
  }

  /**
   * Returns the label dependencies for the given late-bound attribute in this rule.
   *
   * @param rule the rule being evaluated
   * @param attribute the attribute to evaluate
   * @param config the configuration to evaluate the attribute in
   * @param attributeMap mapper to attribute values
   */
  private Iterable<Label> resolveLateBoundAttribute(
      Rule rule,
      Attribute attribute,
      BuildConfiguration config,
      AttributeMap attributeMap)
      throws EvalException, InterruptedException {
    Preconditions.checkArgument(attribute.isLateBound());

    Object actualValue =
        resolveLateBoundDefault(attribute.getLateBoundDefault(), rule, attributeMap, config);
    if (EvalUtils.isNullOrNone(actualValue)) {
      return ImmutableList.<Label>of();
    }
    try {
      ImmutableList.Builder<Label> deps = ImmutableList.builder();
      if (attribute.getType() == BuildType.LABEL) {
        deps.add(rule.getLabel().resolveRepositoryRelative(BuildType.LABEL.cast(actualValue)));
      } else if (attribute.getType() == BuildType.LABEL_LIST) {
        for (Label label : BuildType.LABEL_LIST.cast(actualValue)) {
          deps.add(rule.getLabel().resolveRepositoryRelative(label));
        }
      } else {
        throw new IllegalStateException(
            String.format(
                "Late bound attribute '%s' is not a label or a label list",
                attribute.getName()));
      }
      return deps.build();
    } catch (ClassCastException e) { // From either of the cast calls above.
      throw new EvalException(
          rule.getLocation(),
          String.format(
              "When computing the default value of %s, expected '%s', got '%s'",
              attribute.getName(),
              attribute.getType(),
              EvalUtils.getDataTypeName(actualValue, true)));
    }
  }

  @VisibleForTesting(/* used to test LateBoundDefaults' default values */ )
  public static <FragmentT, ValueT> ValueT resolveLateBoundDefault(
      LateBoundDefault<FragmentT, ValueT> lateBoundDefault,
      Rule rule,
      AttributeMap attributeMap,
      BuildConfiguration config) throws EvalException {
    Class<FragmentT> fragmentClass = lateBoundDefault.getFragmentClass();
    // TODO(b/65746853): remove this when nothing uses it anymore
    if (BuildConfiguration.class.equals(fragmentClass)) {
      return lateBoundDefault.resolve(rule, attributeMap, fragmentClass.cast(config));
    }
    if (Void.class.equals(fragmentClass)) {
      return lateBoundDefault.resolve(rule, attributeMap, null);
    }
    FragmentT fragment =
        fragmentClass.cast(
            config.getFragment((Class<? extends BuildConfiguration.Fragment>) fragmentClass));
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
  private void addExplicitDeps(RuleResolver depResolver, String attrName, Iterable<Label> labels)
      throws InterruptedException, InconsistentAspectOrderException {
    Rule rule = depResolver.rule;
    if (!rule.isAttrDefined(attrName, BuildType.LABEL_LIST)
        && !rule.isAttrDefined(attrName, BuildType.NODEP_LABEL_LIST)) {
      return;
    }
    Attribute attribute = rule.getRuleClassObject().getAttributeByName(attrName);
    for (Label label : labels) {
      depResolver.resolveDep(new AttributeAndOwner(attribute), label);
    }
  }

  /**
   * Converts the given multimap of attributes to labels into a multi map of attributes to {@link
   * Dependency} objects using the proper configuration transition for each attribute.
   *
   * @throws IllegalArgumentException if the {@code node} does not refer to a {@link Rule} instance
   */
  public final Collection<Dependency> resolveRuleLabels(
      TargetAndConfiguration node,
      OrderedSetMultimap<Attribute, Label> depLabels,
      NestedSetBuilder<Label> rootCauses)
      throws InterruptedException, InconsistentAspectOrderException {
    Preconditions.checkArgument(node.getTarget() instanceof Rule);
    Rule rule = (Rule) node.getTarget();
    OrderedSetMultimap<Attribute, Dependency> outgoingEdges = OrderedSetMultimap.create();
    RuleResolver depResolver = new RuleResolver(
        rule, node.getConfiguration(), ImmutableList.<Aspect>of(),
        /*attributeMap=*/null, rootCauses, outgoingEdges);
    Map<Attribute, Collection<Label>> m = depLabels.asMap();
    for (Map.Entry<Attribute, Collection<Label>> entry : depLabels.asMap().entrySet()) {
      for (Label depLabel : entry.getValue()) {
        depResolver.resolveDep(new AttributeAndOwner(entry.getKey()), depLabel);
      }
    }
    return outgoingEdges.values();
  }

  private void visitPackageGroup(
      TargetAndConfiguration node,
      PackageGroup packageGroup,
      NestedSetBuilder<Label> rootCauses,
      Collection<Dependency> outgoingEdges)
      throws InterruptedException {
    for (Label label : packageGroup.getIncludes()) {
      Target target = getTarget(packageGroup, label, rootCauses);
      if (target == null) {
        continue;
      }
      if (!(target instanceof PackageGroup)) {
        // Note that this error could also be caught in PackageGroupConfiguredTarget, but since
        // these have the null configuration, visiting the corresponding target would trigger an
        // analysis of a rule with a null configuration, which doesn't work.
        invalidPackageGroupReferenceHook(node, label);
        continue;
      }

      outgoingEdges.add(Dependency.withNullConfiguration(label));
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
    private final NestedSetBuilder<Label> rootCauses;
    private final OrderedSetMultimap<Attribute, Dependency> outgoingEdges;
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
    RuleResolver(Rule rule, BuildConfiguration ruleConfig, Iterable<Aspect> aspects,
        ConfiguredAttributeMapper attributeMap, NestedSetBuilder<Label> rootCauses,
        OrderedSetMultimap<Attribute, Dependency> outgoingEdges) {
      this.rule = rule;
      this.ruleConfig = ruleConfig;
      this.aspects = aspects;
      this.attributeMap = attributeMap;
      this.rootCauses = rootCauses;
      this.outgoingEdges = outgoingEdges;

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
     * Resolves the given dep for the given attribute, including determining which configurations to
     * apply to it.
     */
    void resolveDep(AttributeAndOwner attributeAndOwner, Label depLabel)
        throws InterruptedException, InconsistentAspectOrderException {
      Target toTarget = getTarget(rule, depLabel, rootCauses);
      if (toTarget == null) {
        return; // Skip this round: we still need to Skyframe-evaluate the dep's target.
      }
      Transition transition = transitionResolver.evaluateTransition(
          ruleConfig, rule, attributeAndOwner.attribute, toTarget, attributeMap);
      outgoingEdges.put(
          attributeAndOwner.attribute,
          transition == ConfigurationTransitionProxy.NULL
              ? Dependency.withNullConfiguration(depLabel)
              : Dependency.withTransitionAndAspects(depLabel, transition,
                    requiredAspects(attributeAndOwner, toTarget)));
    }

    /**
     * Resolves the given dep for the given attribute using a pre-prepared configuration.
     *
     * <p>Use this method with care: it skips Bazel's standard config transition semantics ({@link
     * TransitionResolver#evaluateTransition}). That means attributes passed through here won't
     * obey standard rules on which configurations apply to their deps. This should only be done for
     * special circumstances that really justify the difference. When in doubt, use {@link
     * #resolveDep(AttributeAndOwner, Label)}.
     */
    void resolveDep(AttributeAndOwner attributeAndOwner, Label depLabel, BuildConfiguration config)
        throws InterruptedException, InconsistentAspectOrderException {
      Target toTarget = getTarget(rule, depLabel, rootCauses);
      if (toTarget == null) {
        return; // Skip this round: this is either a loading error or unevaluated Skyframe dep.
      }
      outgoingEdges.put(
          attributeAndOwner.attribute,
          transitionResolver.usesNullConfiguration(toTarget)
              ? Dependency.withNullConfiguration(depLabel)
              : Dependency.withTransitionAndAspects(depLabel, new FixedTransition(
                    config.getOptions()), requiredAspects(attributeAndOwner, toTarget)));
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

  /**
   * A patch transition that returns a fixed set of options regardless of the input.
   */
  private static class FixedTransition implements PatchTransition {
    private final BuildOptions toOptions;

    FixedTransition(BuildOptions toOptions) {
      this.toOptions = toOptions;
    }

    @Override
    public BuildOptions apply(BuildOptions options) {
      return toOptions;
    }
  }

  private void visitTargetVisibility(
      TargetAndConfiguration node,
      NestedSetBuilder<Label> rootCauses,
      Collection<Dependency> outgoingEdges)
      throws InterruptedException {
    Target target = node.getTarget();
    for (Label label : target.getVisibility().getDependencyLabels()) {
      Target visibilityTarget = getTarget(target, label, rootCauses);
      if (visibilityTarget == null) {
        continue;
      }
      if (!(visibilityTarget instanceof PackageGroup)) {
        // Note that this error could also be caught in
        // AbstractConfiguredTarget.convertVisibility(), but we have an
        // opportunity here to avoid dependency cycles that result from
        // the visibility attribute of a rule referring to a rule that
        // depends on it (instead of its package)
        invalidVisibilityReferenceHook(node, label);
        continue;
      }

      // Visibility always has null configuration
      outgoingEdges.add(Dependency.withNullConfiguration(label));
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
   * Hook for the error case where a dependency is missing.
   *
   * @param from the target referencing the missing target
   * @param to the missing target
   * @param e the exception that was thrown, e.g., by {@link #getTarget}
   */
  protected abstract void missingEdgeHook(Target from, Label to, NoSuchThingException e)
      throws InterruptedException;

  /**
   * Returns the target by the given label.
   *
   * <p>Returns null if the target is not ready to be returned at this moment. If getTarget returns
   * null once or more during a {@link #dependentNodeMap} call, the results of that call will be
   * incomplete. For use within Skyframe, where several iterations may be needed to discover all
   * dependencies.
   */
  @Nullable
  protected abstract Target getTarget(Target from, Label label, NestedSetBuilder<Label> rootCauses)
      throws InterruptedException;

  /**
   * Returns the build configurations with the given options and fragments, in the same order as the
   * input options.
   *
   * <p>Returns null if any configurations aren't ready to be returned at this moment. If
   * getConfigurations returns null once or more during a {@link #dependentNodeMap} call, the
   * results of that call will be incomplete. For use within Skyframe, where several iterations may
   * be needed to discover all dependencies.
   */
  @Nullable
  protected abstract List<BuildConfiguration> getConfigurations(
      Set<Class<? extends BuildConfiguration.Fragment>> fragments,
      Iterable<BuildOptions> buildOptions)
      throws InvalidConfigurationException, InterruptedException;

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
