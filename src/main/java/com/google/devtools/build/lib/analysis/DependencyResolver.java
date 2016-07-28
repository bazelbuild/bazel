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

import com.google.common.base.Verify;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.LateBoundDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.EnvironmentGroup;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
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
  protected DependencyResolver() {
  }

  /**
   * Returns ids for dependent nodes of a given node, sorted by attribute. Note that some
   * dependencies do not have a corresponding attribute here, and we use the null attribute to
   * represent those edges. Visibility attributes are only visited if {@code visitVisibility} is
   * {@code true}.
   *
   * <p>If {@code aspect} is null, returns the dependent nodes of the configured
   * target node representing the given target and configuration, otherwise that of the aspect
   * node accompanying the aforementioned configured target node for the specified aspect.
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
   * @param hostConfig the configuration this target would use if it was evaluated as a host
   *     tool. This is needed to support {@link LateBoundDefault#useHostConfiguration()}.
   * @param aspect the aspect applied to this target (if any)
   * @param configConditions resolver for config_setting labels
   *
   * @return a mapping of each attribute in this rule or aspect to its dependent nodes
   */
  public final ListMultimap<Attribute, Dependency> dependentNodeMap(
      TargetAndConfiguration node,
      BuildConfiguration hostConfig,
      @Nullable Aspect aspect,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions)
      throws EvalException, InterruptedException {
    NestedSetBuilder<Label> rootCauses = NestedSetBuilder.<Label>stableOrder();
    ListMultimap<Attribute, Dependency> outgoingEdges = dependentNodeMap(
        node, hostConfig, aspect, configConditions, rootCauses);
    if (!rootCauses.isEmpty()) {
      throw new IllegalStateException(rootCauses.build().iterator().next().toString());
    }
    return outgoingEdges;
  }

  /**
   * Returns ids for dependent nodes of a given node, sorted by attribute. Note that some
   * dependencies do not have a corresponding attribute here, and we use the null attribute to
   * represent those edges. Visibility attributes are only visited if {@code visitVisibility} is
   * {@code true}.
   *
   * <p>If {@code aspect} is null, returns the dependent nodes of the configured
   * target node representing the given target and configuration, otherwise that of the aspect
   * node accompanying the aforementioned configured target node for the specified aspect.
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
   * @param hostConfig the configuration this target would use if it was evaluated as a host
   *     tool. This is needed to support {@link LateBoundDefault#useHostConfiguration()}.
   * @param aspect the aspect applied to this target (if any)
   * @param configConditions resolver for config_setting labels
   * @param rootCauses collector for dep labels that can't be (loading phase) loaded
   *
   * @return a mapping of each attribute in this rule or aspect to its dependent nodes
   */
  public final ListMultimap<Attribute, Dependency> dependentNodeMap(
      TargetAndConfiguration node,
      BuildConfiguration hostConfig,
      @Nullable Aspect aspect,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      NestedSetBuilder<Label> rootCauses)
      throws EvalException, InterruptedException {
    Target target = node.getTarget();
    BuildConfiguration config = node.getConfiguration();
    ListMultimap<Attribute, Dependency> outgoingEdges = ArrayListMultimap.create();
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
      visitRule(node, hostConfig, aspect, configConditions, rootCauses, outgoingEdges);
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
      @Nullable Aspect aspect,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      NestedSetBuilder<Label> rootCauses,
      ListMultimap<Attribute, Dependency> outgoingEdges)
      throws EvalException, InterruptedException {
    Preconditions.checkArgument(node.getTarget() instanceof Rule);
    BuildConfiguration ruleConfig = Preconditions.checkNotNull(node.getConfiguration());
    Rule rule = (Rule) node.getTarget();

    ConfiguredAttributeMapper attributeMap = ConfiguredAttributeMapper.of(rule, configConditions);
    attributeMap.validateAttributes();
    RuleResolver depResolver =
        new RuleResolver(rule, ruleConfig, aspect, attributeMap, rootCauses, outgoingEdges);

    visitTargetVisibility(node, rootCauses, outgoingEdges.get(null));
    resolveEarlyBoundAttributes(depResolver);
    resolveLateBoundAttributes(depResolver, ruleConfig, hostConfig);
  }

  /**
   * Resolves the dependencies for all attributes in this rule except late-bound attributes
   * (which require special processing: see {@link #resolveLateBoundAttributes}).
   */
  private void resolveEarlyBoundAttributes(RuleResolver depResolver)
      throws EvalException, InterruptedException {
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

  private void resolveExplicitAttributes(final RuleResolver depResolver) {
    depResolver.attributeMap.visitLabels(
        new AttributeMap.AcceptsLabelAttribute() {
          @Override
          public void acceptLabelAttribute(Label label, Attribute attribute) {
            if (attribute.getType() == BuildType.NODEP_LABEL || attribute.isImplicit()
                || attribute.isLateBound()) {
              return;
            }
            depResolver.resolveDep(attribute, label);
          }
        });
  }

  /**
   * Resolves the dependencies for all implicit attributes in this rule.
   */
  private void resolveImplicitAttributes(RuleResolver depResolver) {
    // Since the attributes that come from aspects do not appear in attributeMap, we have to get
    // their values from somewhere else. This incidentally means that aspects attributes are not
    // configurable. It would be nice if that wasn't the case, but we'd have to revamp how
    // attribute mapping works, which is a large chunk of work.
    Rule rule = depResolver.rule;
    Label ruleLabel = rule.getLabel();
    ConfiguredAttributeMapper attributeMap = depResolver.attributeMap;
    ImmutableSet<String> mappedAttributes = ImmutableSet.copyOf(attributeMap.getAttributeNames());
    for (Attribute attribute : depResolver.attributes) {
      if (!attribute.isImplicit() || !attribute.getCondition().apply(attributeMap)) {
        continue;
      }

      if (attribute.getType() == BuildType.LABEL) {
        Label label = mappedAttributes.contains(attribute.getName())
            ? attributeMap.get(attribute.getName(), BuildType.LABEL)
            : BuildType.LABEL.cast(attribute.getDefaultValue(rule));

        if (label != null) {
          label = ruleLabel.resolveRepositoryRelative(label);
          depResolver.resolveDep(attribute, label);
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
          depResolver.resolveDep(attribute, ruleLabel.resolveRepositoryRelative(label));
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
      throws EvalException, InterruptedException {
    ConfiguredAttributeMapper attributeMap = depResolver.attributeMap;
    for (Attribute attribute : depResolver.attributes) {
      if (!attribute.isLateBound() || !attribute.getCondition().apply(attributeMap)) {
        continue;
      }

      @SuppressWarnings("unchecked")
      LateBoundDefault<BuildConfiguration> lateBoundDefault =
        (LateBoundDefault<BuildConfiguration>) attribute.getLateBoundDefault();

      if (attribute.hasSplitConfigurationTransition()) {
        // Late-bound attribute with a split transition:
        // Since we want to get the same results as BuildConfiguration.evaluateTransition (but
        // skip it since we've already applied the split), we want to make sure this logic
        // doesn't do anything differently. evaluateTransition has additional logic
        // for host configs and attributes with configurators. So we check here that neither of
        // of those apply, in the name of keeping the fork as simple as possible.
        Verify.verify(attribute.getConfigurator() == null);
        Verify.verify(!lateBoundDefault.useHostConfiguration());
        for (BuildConfiguration splitConfig : ruleConfig.getSplitConfigurations(
            attribute.getSplitTransition(depResolver.rule))) {
          // TODO(gregce): support dynamic split transitions
          for (Label dep : resolveLateBoundAttribute(
              depResolver.rule, attribute, splitConfig, attributeMap)) {
            // Skip the normal config transition pipeline and directly feed the split config. This
            // is because the split already had to be applied to determine the attribute's value.
            // This makes the split logic in the normal pipeline redundant and potentially
            // incorrect.
            depResolver.resolveDep(attribute, dep, splitConfig);
          }
        }
      } else {
        // Late-bound attribute without a split transition:
        for (Label dep : resolveLateBoundAttribute(depResolver.rule, attribute,
            lateBoundDefault.useHostConfiguration() ? hostConfig : ruleConfig, attributeMap)) {
          // Process this dep like a normal attribute.
          depResolver.resolveDep(attribute, dep);
        }
      }
    }
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

    @SuppressWarnings("unchecked")
    LateBoundDefault<BuildConfiguration> lateBoundDefault =
      (LateBoundDefault<BuildConfiguration>) attribute.getLateBoundDefault();

    // TODO(bazel-team): This might be too expensive - can we cache this somehow?
    if (!lateBoundDefault.getRequiredConfigurationFragments().isEmpty()) {
      if (!config.hasAllFragments(lateBoundDefault.getRequiredConfigurationFragments())) {
        return ImmutableList.<Label>of();
      }
    }

    // TODO(bazel-team): We should check if the implementation tries to access an undeclared
    // fragment.
    Object actualValue = lateBoundDefault.resolve(rule, attributeMap, config);
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

  /**
   * Adds new dependencies to the given rule under the given attribute name
   *
   * @param depResolver the resolver for this rule's deps
   * @param attrName the name of the attribute to add dependency labels to
   * @param labels the dependencies to add
   */
  private void addExplicitDeps(RuleResolver depResolver, String attrName, Iterable<Label> labels) {
    Rule rule = depResolver.rule;
    if (!rule.isAttrDefined(attrName, BuildType.LABEL_LIST)
        && !rule.isAttrDefined(attrName, BuildType.NODEP_LABEL_LIST)) {
      return;
    }
    Attribute attribute = rule.getRuleClassObject().getAttributeByName(attrName);
    for (Label label : labels) {
      depResolver.resolveDep(attribute, label);
    }
  }

  /**
   * Converts the given multimap of attributes to labels into a multi map of attributes to
   * {@link Dependency} objects using the proper configuration transition for each attribute.
   *
   * @throws IllegalArgumentException if the {@code node} does not refer to a {@link Rule} instance
   */
  public final Collection<Dependency> resolveRuleLabels(TargetAndConfiguration node,
      ListMultimap<Attribute, Label> depLabels, NestedSetBuilder<Label> rootCauses) {
    Preconditions.checkArgument(node.getTarget() instanceof Rule);
    Rule rule = (Rule) node.getTarget();
    ListMultimap<Attribute, Dependency> outgoingEdges = ArrayListMultimap.create();
    RuleResolver depResolver = new RuleResolver(rule, node.getConfiguration(), /*aspect=*/null,
        /*attributeMap=*/null, rootCauses, outgoingEdges);
    for (Map.Entry<Attribute, Collection<Label>> entry : depLabels.asMap().entrySet()) {
      for (Label depLabel : entry.getValue()) {
        depResolver.resolveDep(entry.getKey(), depLabel);
      }
    }
    return outgoingEdges.values();
  }

  private void visitPackageGroup(TargetAndConfiguration node, PackageGroup packageGroup,
      NestedSetBuilder<Label> rootCauses, Collection<Dependency> outgoingEdges) {
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

  private ImmutableSet<AspectDescriptor> requiredAspects(
      @Nullable Aspect aspect, Attribute attribute, final Target target, Rule originalRule) {
    if (!(target instanceof Rule)) {
      return ImmutableSet.of();
    }

    Iterable<Aspect> aspectCandidates = extractAspectCandidates(aspect, attribute, originalRule);
    RuleClass ruleClass = ((Rule) target).getRuleClassObject();
    ImmutableSet.Builder<AspectDescriptor> result = ImmutableSet.builder();
    for (Aspect aspectCandidate : aspectCandidates) {
      if (ruleClass.canHaveAnyProvider() || Sets.difference(
              aspectCandidate.getDefinition().getRequiredProviders(),
              ruleClass.getAdvertisedProviders())
          .isEmpty()) {
        result.add(
            new AspectDescriptor(
                aspectCandidate.getAspectClass(),
                aspectCandidate.getParameters()));
      }
    }
    return result.build();
  }

  private static Iterable<Aspect> extractAspectCandidates(
      @Nullable Aspect aspect, Attribute attribute, Rule originalRule) {
    // The order of this set will be deterministic. This is necessary because this order eventually
    // influences the order in which aspects are merged into the main configured target, which in
    // turn influences which aspect takes precedence if two emit the same provider (maybe this
    // should be an error)
    Set<Aspect> aspectCandidates = new LinkedHashSet<>();
    aspectCandidates.addAll(attribute.getAspects(originalRule));
    if (aspect != null) {
      for (AspectClass aspectClass :
          aspect.getDefinition().getAttributeAspects().get(attribute.getName())) {
        if (aspectClass.equals(aspect.getAspectClass())) {
          aspectCandidates.add(aspect);
        } else if (aspectClass instanceof NativeAspectClass) {
          aspectCandidates.add(
              Aspect.forNative((NativeAspectClass) aspectClass, aspect.getParameters()));
        } else {
          // If we ever want to support this specifying arbitrary aspects for Skylark aspects,
          // we will need to do a Skyframe call here to load an up-to-date definition.
          throw new IllegalStateException(
              "Skylark aspect classes sending different aspects along attributes is not supported");
        }
      }
    }
    return aspectCandidates;
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
    private final Aspect aspect;
    private final ConfiguredAttributeMapper attributeMap;
    private final NestedSetBuilder<Label> rootCauses;
    private final ListMultimap<Attribute, Dependency> outgoingEdges;
    private final List<Attribute> attributes;

    /**
     * Constructs a new dependency resolver for the specified rule context.
     *
     * @param rule the rule being evaluated
     * @param ruleConfig the rule's configuration
     * @param aspect the aspect applied to this rule (if any)
     * @param attributeMap mapper for the rule's attribute values
     * @param rootCauses output collector for dep labels that can't be (loading phase) loaded
     * @param outgoingEdges output collector for the resolved dependencies
     */
    RuleResolver(Rule rule, BuildConfiguration ruleConfig, Aspect aspect,
        ConfiguredAttributeMapper attributeMap, NestedSetBuilder<Label> rootCauses,
        ListMultimap<Attribute, Dependency> outgoingEdges) {
      this.rule = rule;
      this.ruleConfig = ruleConfig;
      this.aspect = aspect;
      this.attributeMap = attributeMap;
      this.rootCauses = rootCauses;
      this.outgoingEdges = outgoingEdges;
      this.attributes = getAttributes(rule, aspect);
    }

    /**
     * Returns the attributes that should be visited for this rule/aspect combination.
     */
    private List<Attribute> getAttributes(Rule rule, @Nullable Aspect aspect) {
      List<Attribute> ruleDefs = rule.getRuleClassObject().getAttributes();
      if (aspect == null) {
        return ruleDefs;
      } else {
        return ImmutableList.<Attribute>builder()
            .addAll(ruleDefs)
            .addAll(aspect.getDefinition().getAttributes().values())
            .build();
      }
    }

    /**
     * Resolves the given dep for the given attribute, including determining which
     * configurations to apply to it.
     */
    void resolveDep(Attribute attribute, Label depLabel) {
      // Late-bound split attributes are separately handled (see LateBoundSplitResolver).
      Verify.verify(!(attribute.isLateBound() && attribute.hasSplitConfigurationTransition()));
      Target toTarget = getTarget(rule, depLabel, rootCauses);
      if (toTarget == null) {
        return; // Skip this round: we still need to Skyframe-evaluate the dep's target.
      }
      BuildConfiguration.TransitionApplier resolver = ruleConfig.getTransitionApplier();
      ruleConfig.evaluateTransition(rule, attribute, toTarget, resolver);
      // An <Attribute, Label> pair can resolve to multiple deps because of split transitions.
      for (Dependency dependency :
          resolver.getDependencies(depLabel, requiredAspects(aspect, attribute, toTarget, rule))) {
        outgoingEdges.put(attribute, dependency);
      }
    }

    /**
     * Resolves the given dep for the given attribute using a pre-prepared configuration.
     *
     * <p>Use this method with care: it skips Bazel's standard config transition semantics
     * ({@link BuildConfiguration#evaluateTransition}). That means attributes passed through here
     * won't obey standard rules on which configurations apply to their deps. This should only
     * be done for special circumstances that really justify the difference. When in doubt, use
     * {@link #resolveDep(Attribute, Label)}.
     */
    void resolveDep(Attribute attribute, Label depLabel, BuildConfiguration config) {
      Target toTarget = getTarget(rule, depLabel, rootCauses);
      if (toTarget == null) {
        return; // Skip this round: this is either a loading error or unevaluated Skyframe dep.
      }
      BuildConfiguration.TransitionApplier transitionApplier = config.getTransitionApplier();
      if (BuildConfiguration.usesNullConfiguration(toTarget)) {
        transitionApplier.applyTransition(Attribute.ConfigurationTransition.NULL);
      }
      Dependency dep = Iterables.getOnlyElement(transitionApplier.getDependencies(depLabel,
          requiredAspects(aspect, attribute, toTarget, rule)));
      outgoingEdges.put(attribute, dep);
    }
  }

  private void visitTargetVisibility(TargetAndConfiguration node,
      NestedSetBuilder<Label> rootCauses, Collection<Dependency> outgoingEdges) {
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
  protected abstract void missingEdgeHook(Target from, Label to, NoSuchThingException e);

  /**
   * Returns the target by the given label.
   *
   * <p>Returns null if the target is not ready to be returned at this moment. If getTarget returns
   * null once or more during a {@link #dependentNodeMap} call, the results of that call will be
   * incomplete. For use within Skyframe, where several iterations may be needed to discover
   * all dependencies.
   */
  @Nullable
  protected abstract Target getTarget(Target from, Label label, NestedSetBuilder<Label> rootCauses);
}
