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
import static com.google.devtools.build.lib.analysis.DependencyKind.VISIBILITY_DEPENDENCY;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.AspectCollection.AspectCycleOnPathException;
import com.google.devtools.build.lib.analysis.DependencyKind.AttributeDependencyKind;
import com.google.devtools.build.lib.analysis.DependencyKind.ToolchainDependencyKind;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.config.TransitionResolver;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NullTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.Attribute.LateBoundDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.EnvironmentGroup;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.ToolchainContextKey;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/**
 * Resolver for dependencies between configured targets.
 *
 * <p>Includes logic to derive the right configurations depending on transition type.
 */
public abstract class DependencyResolver {

  /**
   * Returns whether or not to use the new toolchain transition. Checks the global incompatible
   * change flag and the rule's toolchain transition readiness attribute.
   */
  // TODO(#10523): Remove this when the migration period for toolchain transitions has ended.
  public static boolean shouldUseToolchainTransition(
      @Nullable BuildConfiguration configuration, Target target) {
    return shouldUseToolchainTransition(
        configuration, target instanceof Rule ? (Rule) target : null);
  }

  /**
   * Returns whether or not to use the new toolchain transition. Checks the global incompatible
   * change flag and the rule's toolchain transition readiness attribute.
   */
  // TODO(#10523): Remove this when the migration period for toolchain transitions has ended.
  public static boolean shouldUseToolchainTransition(
      @Nullable BuildConfiguration configuration, @Nullable Rule rule) {
    // Check whether the global incompatible change flag is set.
    if (configuration != null) {
      PlatformOptions platformOptions = configuration.getOptions().get(PlatformOptions.class);
      if (platformOptions != null && platformOptions.overrideToolchainTransition) {
        return true;
      }
    }

    // Check the rule definition to see if it is ready.
    if (rule != null && rule.getRuleClassObject().useToolchainTransition()) {
      return true;
    }

    // Default to false.
    return false;
  }

  /**
   * What we know about a dependency edge after factoring in the properties of the configured target
   * that the edge originates from, but not the properties of target it points to.
   */
  @AutoValue
  abstract static class PartiallyResolvedDependency {
    abstract Label getLabel();

    abstract ConfigurationTransition getTransition();

    abstract ImmutableList<Aspect> getPropagatingAspects();

    @Nullable
    abstract ToolchainContextKey getToolchainContextKey();

    /** A Builder to create instances of PartiallyResolvedDependency. */
    @AutoValue.Builder
    abstract static class Builder {
      abstract Builder setLabel(Label label);

      abstract Builder setTransition(ConfigurationTransition transition);

      abstract Builder setPropagatingAspects(List<Aspect> propagatingAspects);

      @Nullable
      abstract Builder setToolchainContextKey(ToolchainContextKey toolchainContextKey);

      abstract PartiallyResolvedDependency build();
    }

    static Builder builder() {
      return new AutoValue_DependencyResolver_PartiallyResolvedDependency.Builder()
          .setPropagatingAspects(ImmutableList.of());
    }

    public DependencyKey.Builder getDependencyKeyBuilder() {
      return DependencyKey.builder()
          .setLabel(getLabel())
          .setToolchainContextKey(getToolchainContextKey());
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
   * @param toolchainContexts the toolchain contexts for this target
   * @param trimmingTransitionFactory the transition factory used to trim rules (note: this is a
   *     temporary feature; see the corresponding methods in ConfiguredRuleClassProvider)
   * @return a mapping of each attribute in this rule or aspects to its dependent nodes
   */
  public final OrderedSetMultimap<DependencyKind, DependencyKey> dependentNodeMap(
      TargetAndConfiguration node,
      BuildConfiguration hostConfig,
      @Nullable Aspect aspect,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      @Nullable ToolchainCollection<ToolchainContext> toolchainContexts,
      boolean useToolchainTransition,
      @Nullable TransitionFactory<Rule> trimmingTransitionFactory)
      throws EvalException, InterruptedException, InconsistentAspectOrderException {
    NestedSetBuilder<Cause> rootCauses = NestedSetBuilder.stableOrder();
    OrderedSetMultimap<DependencyKind, DependencyKey> outgoingEdges =
        dependentNodeMap(
            node,
            hostConfig,
            aspect != null ? ImmutableList.of(aspect) : ImmutableList.of(),
            configConditions,
            toolchainContexts,
            useToolchainTransition,
            rootCauses,
            trimmingTransitionFactory);
    if (!rootCauses.isEmpty()) {
      throw new IllegalStateException(rootCauses.build().toList().iterator().next().toString());
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
   * @param toolchainContexts the toolchain contexts for this target
   * @param trimmingTransitionFactory the transition factory used to trim rules (note: this is a
   *     temporary feature; see the corresponding methods in ConfiguredRuleClassProvider)
   * @param rootCauses collector for dep labels that can't be (loading phase) loaded
   * @return a mapping of each attribute in this rule or aspects to its dependent nodes
   */
  public final OrderedSetMultimap<DependencyKind, DependencyKey> dependentNodeMap(
      TargetAndConfiguration node,
      BuildConfiguration hostConfig,
      Iterable<Aspect> aspects,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      @Nullable ToolchainCollection<ToolchainContext> toolchainContexts,
      boolean useToolchainTransition,
      NestedSetBuilder<Cause> rootCauses,
      @Nullable TransitionFactory<Rule> trimmingTransitionFactory)
      throws EvalException, InterruptedException, InconsistentAspectOrderException {
    Target target = node.getTarget();
    BuildConfiguration config = node.getConfiguration();
    OrderedSetMultimap<DependencyKind, Label> outgoingLabels = OrderedSetMultimap.create();

    // TODO(bazel-team): Figure out a way to implement the below (and partiallyResolveDependencies)
    // using LabelVisitationUtils.
    Rule fromRule = null;
    ConfiguredAttributeMapper attributeMap = null;
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
      fromRule = (Rule) target;
      attributeMap = ConfiguredAttributeMapper.of(fromRule, configConditions);
      visitRule(node, hostConfig, aspects, attributeMap, toolchainContexts, outgoingLabels);
    } else if (target instanceof PackageGroup) {
      outgoingLabels.putAll(VISIBILITY_DEPENDENCY, ((PackageGroup) target).getIncludes());
    } else {
      throw new IllegalStateException(target.getLabel().toString());
    }

    Map<Label, Target> targetMap = getTargets(outgoingLabels, node, rootCauses);
    if (targetMap == null) {
      // Dependencies could not be resolved. Try again when they are loaded by Skyframe.
      return OrderedSetMultimap.create();
    }

    OrderedSetMultimap<DependencyKind, PartiallyResolvedDependency> partiallyResolvedDeps =
        partiallyResolveDependencies(
            outgoingLabels,
            fromRule,
            attributeMap,
            toolchainContexts,
            useToolchainTransition,
            aspects);

    OrderedSetMultimap<DependencyKind, DependencyKey> outgoingEdges =
        fullyResolveDependencies(
            partiallyResolvedDeps, targetMap, node.getConfiguration(), trimmingTransitionFactory);

    return outgoingEdges;
  }

  /**
   * Factor in the properties of the current rule into the dependency edge calculation.
   *
   * <p>The target of the dependency edges depends on two things: the rule that depends on them and
   * the type of target they depend on. This function takes the rule into account. Accordingly, it
   * should <b>NOT</b> get the {@link Target} instances representing the targets of the dependency
   * edges as an argument.
   */
  private OrderedSetMultimap<DependencyKind, PartiallyResolvedDependency>
      partiallyResolveDependencies(
          OrderedSetMultimap<DependencyKind, Label> outgoingLabels,
          @Nullable Rule fromRule,
          ConfiguredAttributeMapper attributeMap,
          @Nullable ToolchainCollection<ToolchainContext> toolchainContexts,
          boolean useToolchainTransition,
          Iterable<Aspect> aspects)
          throws EvalException {
    OrderedSetMultimap<DependencyKind, PartiallyResolvedDependency> partiallyResolvedDeps =
        OrderedSetMultimap.create();

    for (Map.Entry<DependencyKind, Label> entry : outgoingLabels.entries()) {
      Label toLabel = entry.getValue();

      if (DependencyKind.isToolchain(entry.getKey())) {
        // This dependency is a toolchain. Its package has not been loaded and therefore we can't
        // determine which aspects and which rule configuration transition we should use, so just
        // use sensible defaults. Not depending on their package makes the error message reporting
        // a missing toolchain a bit better.
        // TODO(lberki): This special-casing is weird. Find a better way to depend on toolchains.
        // TODO(#10523): Remove check when this is fully released.
        // This logic needs to stay in sync with the dep finding logic in
        // //third_party/bazel/src/main/java/com/google/devtools/build/lib/analysis/Util.java#findImplicitDeps.
        if (useToolchainTransition) {
          ToolchainDependencyKind tdk = (ToolchainDependencyKind) entry.getKey();
          ToolchainContext toolchainContext =
              toolchainContexts.getToolchainContext(tdk.getExecGroupName());
          partiallyResolvedDeps.put(
              entry.getKey(),
              PartiallyResolvedDependency.builder()
                  .setLabel(toLabel)
                  .setTransition(NoTransition.INSTANCE)
                  .setToolchainContextKey(toolchainContext.key())
                  .build());
        } else {
          // Legacy approach: use a HostTransition.
          partiallyResolvedDeps.put(
              entry.getKey(),
              PartiallyResolvedDependency.builder()
                  .setLabel(toLabel)
                  .setTransition(HostTransition.INSTANCE)
                  .build());
        }
        continue;
      }

      if (entry.getKey() == VISIBILITY_DEPENDENCY) {
        partiallyResolvedDeps.put(
            VISIBILITY_DEPENDENCY,
            PartiallyResolvedDependency.builder()
                .setLabel(toLabel)
                .setTransition(NullTransition.INSTANCE)
                .setPropagatingAspects(ImmutableList.of())
                .build());
        continue;
      }

      if (entry.getKey() == OUTPUT_FILE_RULE_DEPENDENCY) {
        partiallyResolvedDeps.put(
            OUTPUT_FILE_RULE_DEPENDENCY,
            PartiallyResolvedDependency.builder()
                .setLabel(toLabel)
                .setTransition(NoTransition.INSTANCE)
                .setPropagatingAspects(ImmutableList.of())
                .build());
        continue;
      }

      Attribute attribute = entry.getKey().getAttribute();
      ImmutableList.Builder<Aspect> propagatingAspects = ImmutableList.builder();
      propagatingAspects.addAll(attribute.getAspects(fromRule));
      collectPropagatingAspects(
          aspects, attribute.getName(), entry.getKey().getOwningAspect(), propagatingAspects);

      Label executionPlatformLabel = null;
      if (toolchainContexts != null) {
        if (attribute.getTransitionFactory() instanceof ExecutionTransitionFactory) {
          String execGroup =
              ((ExecutionTransitionFactory) attribute.getTransitionFactory()).getExecGroup();
          if (!toolchainContexts.hasToolchainContext(execGroup)) {
            String error =
                String.format(
                    "Attr '%s' declares a transition for non-existent exec group '%s'",
                    attribute.getName(), execGroup);
            if (fromRule != null) {
              throw new EvalException(fromRule.getLocation(), error);
            } else {
              throw Starlark.errorf("%s", error);
            }
          }
          if (toolchainContexts.getToolchainContext(execGroup).executionPlatform() != null) {
            executionPlatformLabel =
                toolchainContexts.getToolchainContext(execGroup).executionPlatform().label();
          }
        }
      }

      AttributeTransitionData attributeTransitionData =
          AttributeTransitionData.builder()
              .attributes(attributeMap)
              .executionPlatform(executionPlatformLabel)
              .build();
      ConfigurationTransition attributeTransition =
          attribute.getTransitionFactory().create(attributeTransitionData);
      partiallyResolvedDeps.put(
          entry.getKey(),
          PartiallyResolvedDependency.builder()
              .setLabel(toLabel)
              .setTransition(attributeTransition)
              .setPropagatingAspects(propagatingAspects.build())
              .build());
    }
    return partiallyResolvedDeps;
  }

  /**
   * Factor in the properties of the target where the dependency points to in the dependency edge
   * calculation.
   *
   * <p>The target of the dependency edges depends on two things: the rule that depends on them and
   * the type of target they depend on. This function takes the rule into account. Accordingly, it
   * should <b>NOT</b> get the {@link Rule} instance representing the rule whose dependencies are
   * being calculated as an argument or its attributes and it should <b>NOT</b> do anything with the
   * keys of {@code partiallyResolvedDeps} other than passing them on to the output map.
   */
  private OrderedSetMultimap<DependencyKind, DependencyKey> fullyResolveDependencies(
      OrderedSetMultimap<DependencyKind, PartiallyResolvedDependency> partiallyResolvedDeps,
      Map<Label, Target> targetMap,
      BuildConfiguration originalConfiguration,
      @Nullable TransitionFactory<Rule> trimmingTransitionFactory)
      throws InconsistentAspectOrderException {
    OrderedSetMultimap<DependencyKind, DependencyKey> outgoingEdges = OrderedSetMultimap.create();

    for (Map.Entry<DependencyKind, PartiallyResolvedDependency> entry :
        partiallyResolvedDeps.entries()) {
      PartiallyResolvedDependency partiallyResolvedDependency = entry.getValue();

      Target toTarget = targetMap.get(partiallyResolvedDependency.getLabel());
      if (toTarget == null) {
        // Dependency pointing to non-existent target. This error was reported in getTargets(), so
        // we can just ignore this dependency.
        continue;
      }

      ConfigurationTransition transition =
          TransitionResolver.evaluateTransition(
              originalConfiguration,
              partiallyResolvedDependency.getTransition(),
              toTarget,
              trimmingTransitionFactory);

      AspectCollection requiredAspects =
          filterPropagatingAspects(partiallyResolvedDependency.getPropagatingAspects(), toTarget);

      DependencyKey.Builder dependencyKeyBuilder =
          partiallyResolvedDependency.getDependencyKeyBuilder();
      outgoingEdges.put(
          entry.getKey(),
          dependencyKeyBuilder.setTransition(transition).setAspects(requiredAspects).build());
    }
    return outgoingEdges;
  }

  private void visitRule(
      TargetAndConfiguration node,
      BuildConfiguration hostConfig,
      Iterable<Aspect> aspects,
      ConfiguredAttributeMapper attributeMap,
      @Nullable ToolchainCollection<ToolchainContext> toolchainContexts,
      OrderedSetMultimap<DependencyKind, Label> outgoingLabels)
      throws EvalException {
    Preconditions.checkArgument(node.getTarget() instanceof Rule, node);
    BuildConfiguration ruleConfig = Preconditions.checkNotNull(node.getConfiguration(), node);
    Rule rule = (Rule) node.getTarget();

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

    if (toolchainContexts != null) {
      for (Map.Entry<String, ToolchainContext> entry :
          toolchainContexts.getContextMap().entrySet()) {
        outgoingLabels.putAll(
            DependencyKind.forExecGroup(entry.getKey()),
            entry.getValue().resolvedToolchainLabels());
      }
    }

    if (!rule.isAttributeValueExplicitlySpecified(RuleClass.APPLICABLE_LICENSES_ATTR)) {
      addExplicitDeps(
          outgoingLabels,
          rule,
          RuleClass.APPLICABLE_LICENSES_ATTR,
          rule.getPackage().getDefaultApplicableLicenses());
    }
  }

  private void resolveAttributes(
      OrderedSetMultimap<DependencyKind, Label> outgoingLabels,
      Rule rule,
      ConfiguredAttributeMapper attributeMap,
      Iterable<Aspect> aspects,
      BuildConfiguration ruleConfig,
      BuildConfiguration hostConfig) {
    Label ruleLabel = rule.getLabel();
    for (AttributeDependencyKind dependencyKind : getAttributes(rule, aspects)) {
      Attribute attribute = dependencyKind.getAttribute();
      if (!attribute.getCondition().apply(attributeMap)
          // Not only is resolving CONFIG_SETTING_DEPS_ATTRIBUTE deps here wasteful, since the only
          // place they're used is in ConfiguredTargetFunction.getConfigConditions, but it actually
          // breaks trimming as shown by
          // FeatureFlagManualTrimmingTest#featureFlagInUnusedSelectBranchButNotInTransitiveConfigs_DoesNotError
          // because it resolves a dep that trimming (correctly) doesn't account for because it's
          // part of an unchosen select() branch.
          || attribute.getName().equals(RuleClass.CONFIG_SETTING_DEPS_ATTRIBUTE)) {
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
            dependencyKind.getOwningAspect() == null
                ? attributeMap.get(attribute.getName(), attribute.getType())
                : attribute.getDefaultValue(rule);
        if (attributeValue instanceof ComputedDefault) {
          attributeValue = ((ComputedDefault) attributeValue).getDefault(attributeMap);
        }
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
    Preconditions.checkState(!attribute.getTransitionFactory().isSplit());
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
        fragmentClass.cast(attributeConfig.getFragment((Class<? extends Fragment>) fragmentClass));
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
   * Collects the aspects from {@code aspectPath} that need to be propagated along the attribute
   * {@code attributeName}.
   *
   * <p>It can happen that some of the aspects cannot be propagated if the dependency doesn't have a
   * provider that's required by them. These will be filtered out after the rule class of the
   * dependency is known.
   */
  private static void collectPropagatingAspects(
      Iterable<Aspect> aspectPath,
      String attributeName,
      @Nullable AspectClass aspectOwningAttribute,
      ImmutableList.Builder<Aspect> filteredAspectPath) {
    for (Aspect aspect : aspectPath) {
      if (aspect.getAspectClass().equals(aspectOwningAttribute)) {
        // Do not propagate over the aspect's own attributes.
        continue;
      }

      if (aspect.getDefinition().propagateAlong(attributeName)) {
        filteredAspectPath.add(aspect);
      }
    }
  }

  /** Returns the attributes that should be visited for this rule/aspect combination. */
  private List<AttributeDependencyKind> getAttributes(Rule rule, Iterable<Aspect> aspects) {
    ImmutableList.Builder<AttributeDependencyKind> result = ImmutableList.builder();
    // If processing aspects, aspect attribute names may conflict with the attribute names of
    // rules they attach to. If this occurs, the highest-level aspect attribute takes precedence.
    LinkedHashSet<String> aspectProcessedAttributes = new LinkedHashSet<>();

    for (Aspect aspect : aspects) {
      for (Attribute attribute : aspect.getDefinition().getAttributes().values()) {
        if (!aspectProcessedAttributes.contains(attribute.getName())) {
          result.add(AttributeDependencyKind.forAspect(attribute, aspect.getAspectClass()));
          aspectProcessedAttributes.add(attribute.getName());
        }
      }
    }
    List<Attribute> ruleDefs = rule.getRuleClassObject().getAttributes();
    for (Attribute attribute : ruleDefs) {
      if (!aspectProcessedAttributes.contains(attribute.getName())) {
        result.add(AttributeDependencyKind.forRule(attribute));
      }
    }
    return result.build();
  }

  /**
   * Filter the set of aspects that are to be propagated according to the dependency type and the
   * set of advertised providers of the dependency.
   */
  private AspectCollection filterPropagatingAspects(ImmutableList<Aspect> aspects, Target toTarget)
      throws InconsistentAspectOrderException {
    if (toTarget instanceof OutputFile) {
      aspects =
          aspects.stream()
              .filter(aspect -> aspect.getDefinition().applyToGeneratingRules())
              .collect(ImmutableList.toImmutableList());
      toTarget = ((OutputFile) toTarget).getGeneratingRule();
    }

    if (!(toTarget instanceof Rule) || aspects.isEmpty()) {
      return AspectCollection.EMPTY;
    }

    Rule toRule = (Rule) toTarget;
    ImmutableList.Builder<Aspect> filteredAspectPath = ImmutableList.builder();
    ImmutableSet.Builder<AspectDescriptor> visibleAspects = ImmutableSet.builder();

    for (Aspect aspect : aspects) {
      if (aspect
          .getDefinition()
          .getRequiredProviders()
          .isSatisfiedBy(toRule.getRuleClassObject().getAdvertisedProviders())) {
        filteredAspectPath.add(aspect);
        visibleAspects.add(aspect.getDescriptor());
      }
    }
    try {
      return AspectCollection.create(filteredAspectPath.build(), visibleAspects.build());
    } catch (AspectCycleOnPathException e) {
      throw new InconsistentAspectOrderException(toTarget, e);
    }
  }

  private void visitTargetVisibility(
      TargetAndConfiguration node, OrderedSetMultimap<DependencyKind, Label> outgoingLabels) {
    Target target = node.getTarget();
    outgoingLabels.putAll(VISIBILITY_DEPENDENCY, target.getVisibility().getDependencyLabels());
  }

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
      OrderedSetMultimap<DependencyKind, Label> labelMap,
      TargetAndConfiguration fromNode,
      NestedSetBuilder<Cause> rootCauses)
      throws InterruptedException;
}
