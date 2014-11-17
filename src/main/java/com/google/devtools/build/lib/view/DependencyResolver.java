// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.view;

import com.google.common.base.Preconditions;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.collect.ImmutableSortedKeyListMultimap;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Attribute.LateBoundDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.config.ConfigMatchingProvider;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

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
   */
  public final ListMultimap<Attribute, TargetAndConfiguration> dependentNodeMap(
      TargetAndConfiguration node, Set<ConfigMatchingProvider> configConditions)
      throws EvalException {
    Target target = node.getTarget();
    BuildConfiguration config = node.getConfiguration();
    ListMultimap<Attribute, TargetAndConfiguration> outgoingEdges = ArrayListMultimap.create();
    if (target instanceof OutputFile) {
      Preconditions.checkNotNull(config);
      visitTargetVisibility(node, outgoingEdges.get(null));
      Rule rule = ((OutputFile) target).getGeneratingRule();
      outgoingEdges.get(null).add(new TargetAndConfiguration(rule, config));
    } else if (target instanceof InputFile) {
      visitTargetVisibility(node, outgoingEdges.get(null));
    } else if (target instanceof Rule) {
      Preconditions.checkNotNull(config);
      visitTargetVisibility(node, outgoingEdges.get(null));
      Rule rule = (Rule) target;
      ListMultimap<Attribute, Label> labelMap = resolveLateBoundAttributes(
          rule, config, configConditions);
      visitRule(rule, config, labelMap, outgoingEdges);
    } else if (target instanceof PackageGroup) {
      visitPackageGroup(node, (PackageGroup) target, outgoingEdges.get(null));
    } else {
      throw new IllegalStateException(target.getLabel().toString());
    }
    return outgoingEdges;
  }

  private ListMultimap<Attribute, Label> resolveLateBoundAttributes(Rule rule,
      BuildConfiguration configuration, Set<ConfigMatchingProvider> configConditions)
      throws EvalException {
    final ImmutableSortedKeyListMultimap.Builder<Attribute, Label> builder =
        ImmutableSortedKeyListMultimap.builder();
    ConfiguredAttributeMapper attributes = ConfiguredAttributeMapper.of(rule, configConditions);

    attributes.validateAttributes();
    attributes.visitLabels(
        new AttributeMap.AcceptsLabelAttribute() {
          @Override
          public void acceptLabelAttribute(Label label, Attribute attribute) {
            String attributeName = attribute.getName();
            if (attributeName.equals("abi_deps")) {
              // abi_deps is handled specially: we visit only the branch that
              // needs to be taken based on the configuration.
              return;
            }

            if (attribute.getType() == Type.NODEP_LABEL) {
              return;
            }

            if (Attribute.isLateBound(attributeName)) {
              // Late-binding attributes are handled specially.
              return;
            }

            builder.put(attribute, label);
          }
        });

    // TODO(bazel-team): Remove this in favor of the new configurable attributes.
    if (attributes.getAttributeDefinition("abi_deps") != null) {
      Attribute depsAttribute = attributes.getAttributeDefinition("deps");
      MakeVariableExpander.Context context = new ConfigurationMakeVariableContext(
          rule.getPackage(), configuration);
      String abi = null;
      try {
        abi = MakeVariableExpander.expand(attributes.get("abi", Type.STRING), context);
      } catch (MakeVariableExpander.ExpansionException e) {
        // Ignore this. It will be handled during the analysis phase.
      }

      if (abi != null) {
        for (Map.Entry<String, List<Label>> entry
            : attributes.get("abi_deps", Type.LABEL_LIST_DICT).entrySet()) {
          try {
            if (Pattern.matches(entry.getKey(), abi)) {
              for (Label label : entry.getValue()) {
                builder.put(depsAttribute, label);
              }
            }
          } catch (PatternSyntaxException e) {
            // Ignore this. It will be handled during the analysis phase.
          }
        }
      }
    }

    // Handle late-bound attributes.
    for (Attribute attribute : rule.getAttributes()) {
      String attributeName = attribute.getName();
      if (Attribute.isLateBound(attributeName) && attribute.getCondition().apply(attributes)) {
        @SuppressWarnings("unchecked")
        LateBoundDefault<BuildConfiguration> lateBoundDefault =
            (LateBoundDefault<BuildConfiguration>) attribute.getLateBoundDefault();
        BuildConfiguration actualConfig = configuration;
        if (lateBoundDefault != null && lateBoundDefault.useHostConfiguration()) {
          actualConfig =
              configuration.getConfiguration(ConfigurationTransition.HOST);
        }

        if (attribute.getType() == Type.LABEL) {
          Label label;
          label = Type.LABEL.cast(lateBoundDefault.getDefault(rule, actualConfig));
          if (label != null) {
            builder.put(attribute, label);
          }
        } else if (attribute.getType() == Type.LABEL_LIST) {
          builder.putAll(attribute, Type.LABEL_LIST.cast(
              lateBoundDefault.getDefault(rule, actualConfig)));
        } else {
          throw new AssertionError("Unknown attribute: '" + attributeName + "'");
        }
      }
    }

    // Handle visibility
    builder.putAll(rule.getRuleClassObject().getAttributeByName("visibility"),
        rule.getVisibility().getDependencyLabels());
    return builder.build();
  }

  /**
   * A variant of {@link #dependentNodeMap} that only returns the values of the resulting map, and
   * also converts any internally thrown {@link EvalException} instances into {@link
   * IllegalStateException}.
   */
  public final Collection<TargetAndConfiguration> dependentNodes(
      TargetAndConfiguration node, Set<ConfigMatchingProvider> configConditions) {
    try {
      return dependentNodeMap(node, configConditions).values();
    } catch (EvalException e) {
      throw new IllegalStateException(e);
    }
  }

  /**
   * Converts the given multi map of attributes to labels into a multi map of attributes to
   * (target, configuration) pairs using the proper configuration transition for each attribute.
   *
   * @throws IllegalArgumentException if the {@code node} does not refer to a {@link Rule} instance
   */
  public final ListMultimap<Attribute, TargetAndConfiguration> resolveRuleLabels(
      TargetAndConfiguration node, ListMultimap<Attribute, Label> labelMap) {
    Preconditions.checkArgument(node.getTarget() instanceof Rule);
    Rule rule = (Rule) node.getTarget();
    ListMultimap<Attribute, TargetAndConfiguration> outgoingEdges = ArrayListMultimap.create();
    visitRule(rule, node.getConfiguration(), labelMap, outgoingEdges);
    return outgoingEdges;
  }

  private void visitPackageGroup(TargetAndConfiguration node, PackageGroup packageGroup,
      Collection<TargetAndConfiguration> outgoingEdges) {
    for (Label label : packageGroup.getIncludes()) {
      try {
        Target target = getTarget(label);
        if (target == null) {
          return;
        }
        if (!(target instanceof PackageGroup)) {
          // Note that this error could also be caught in PackageGroupConfiguredTarget, but since
          // these have the null configuration, visiting the corresponding target would trigger an
          // analysis of a rule with a null configuration, which doesn't work.
          invalidPackageGroupReferenceHook(node, label);
          continue;
        }

        outgoingEdges.add(new TargetAndConfiguration(target, node.getConfiguration()));
      } catch (NoSuchThingException e) {
        // Don't visit targets that don't exist (--keep_going)
      }
    }
  }

  private void visitRule(Rule rule, BuildConfiguration config,
      ListMultimap<Attribute, Label> labelMap,
      ListMultimap<Attribute, TargetAndConfiguration> outgoingEdges) {
    Preconditions.checkNotNull(config);
    Preconditions.checkNotNull(labelMap);
    for (Map.Entry<Attribute, Collection<Label>> entry : labelMap.asMap().entrySet()) {
      Attribute attribute = entry.getKey();
      for (Label label : entry.getValue()) {
        Target toTarget;
        try {
          toTarget = getTarget(label);
        } catch (NoSuchThingException e) {
          throw new IllegalStateException("not found: " + label + " from " + rule + " in "
              + attribute.getName());
        }
        if (toTarget == null) {
          continue;
        }
        Iterable<BuildConfiguration> toConfigurations = config.evaluateTransition(
            rule, attribute, toTarget);
        for (BuildConfiguration toConfiguration : toConfigurations) {
          outgoingEdges.get(entry.getKey()).add(
              new TargetAndConfiguration(toTarget, toConfiguration));
        }
      }
    }
  }

  private void visitTargetVisibility(TargetAndConfiguration node,
      Collection<TargetAndConfiguration> outgoingEdges) {
    for (Label label : node.getTarget().getVisibility().getDependencyLabels()) {
      try {
        Target visibilityTarget = getTarget(label);
        if (visibilityTarget == null) {
          return;
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
        outgoingEdges.add(new TargetAndConfiguration(visibilityTarget, null));
      } catch (NoSuchThingException e) {
        // Don't visit targets that don't exist (--keep_going)
      }
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
   * Returns the target by the given label.
   *
   * <p>Throws {@link NoSuchThingException} if the target is known not to exist.
   *
   * <p>Returns null if the target is not ready to be returned at this moment. If getTarget returns
   * null once or more during a {@link #dependentNodeMap} call, the results of that call will be
   * incomplete. For use within Skyframe, where several iterations may be needed to discover
   * all dependencies.
   */
  @Nullable
  protected abstract Target getTarget(Label label) throws NoSuchThingException;
}
