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
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.config.ConfigMatchingProvider;

import java.util.Collection;
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
   * represent those edges.
   */
  public final ListMultimap<Attribute, TargetAndConfiguration> dependentNodeMap(
      TargetAndConfiguration node, ListMultimap<Attribute, Label> labelMap) {
    return dependentNodeMap(node, labelMap, /*visitVisibility=*/true);
  }

  /**
   * Variation that lets the caller choose whether to visit visibility labels in
   * addition to what's explicitly requested.
   */
  public final ListMultimap<Attribute, TargetAndConfiguration> dependentNodeMap(
      TargetAndConfiguration node, ListMultimap<Attribute, Label> labelMap,
      boolean visitVisibility) {
    Target target = node.getTarget();
    ListMultimap<Attribute, TargetAndConfiguration> outgoingEdges = ArrayListMultimap.create();
    if (target instanceof OutputFile) {
      Rule rule = ((OutputFile) target).getGeneratingRule();
      addEdge(rule, node.getConfiguration(), outgoingEdges.get(null));
      visitTargetVisibility(node, outgoingEdges.get(null));
    } else if (target instanceof InputFile) {
      visitTargetVisibility(node, outgoingEdges.get(null));
    } else if (target instanceof Rule) {
      if (visitVisibility) {
        visitTargetVisibility(node, outgoingEdges.get(null));
      }
      visitRule(node, (Rule) target, labelMap, outgoingEdges);
    } else if (target instanceof PackageGroup) {
      visitPackageGroup(node, (PackageGroup) target, outgoingEdges.get(null));
    } else {
      throw new IllegalStateException(target.getLabel().toString());
    }
    return outgoingEdges;
  }

  /**
   * Variation that computes the rule's (Attribute --> Label) map internally. This should be
   * avoided if the caller has reasonable access to somewhere where this has already been computed.
   *
   * <p>TODO(bazel-team): Remove this version when non-SkyFrame code is stripped out.
   */
  public final ListMultimap<Attribute, TargetAndConfiguration> dependentNodeMap(
      TargetAndConfiguration node, Set<ConfigMatchingProvider> configConditions) {
    ListMultimap<Attribute, Label> labelMap = null;
    if (node.getTarget() instanceof Rule) {
      try {
        labelMap = new LateBoundAttributeHelper((Rule) node.getTarget(), node.getConfiguration(),
            configConditions).createAttributeMap();
      } catch (EvalException e) {
        throw new IllegalStateException(e);
      }
    }
    return dependentNodeMap(node, labelMap);
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

        addEdge(target, node.getConfiguration(), outgoingEdges);
      } catch (NoSuchThingException e) {
        // Don't visit targets that don't exist (--keep_going)
      }
    }
  }

  private void addEdge(Target toTarget, BuildConfiguration configuration,
      Collection<TargetAndConfiguration> outgoingEdges) {
    outgoingEdges.add(new TargetAndConfiguration(toTarget, configuration));
  }

  private void visitLabelInAttribute(TargetAndConfiguration from, Rule fromRule, Label to,
      Attribute attribute, Collection<TargetAndConfiguration> outgoingEdges) {
    Preconditions.checkNotNull(from.getConfiguration());
    Target toTarget;
    try {
      toTarget = getTarget(to);
    } catch (NoSuchThingException e) {
      throw new IllegalStateException("not found: " + to + " from " + from + " in "
          + attribute.getName());
    }

    if (toTarget == null) {
      return;
    }

    Iterable<BuildConfiguration> toConfigurations = from.getConfiguration().evaluateTransition(
        fromRule, attribute, toTarget);
    for (BuildConfiguration toConfiguration : toConfigurations) {
      addEdge(toTarget, toConfiguration, outgoingEdges);
    }
  }

  private void visitRule(final TargetAndConfiguration node, Rule rule,
      ListMultimap<Attribute, Label> labelMap,
      ListMultimap<Attribute, TargetAndConfiguration> outgoingEdges) {
    Preconditions.checkNotNull(labelMap);
    for (Map.Entry<Attribute, Collection<Label>> entry : labelMap.asMap().entrySet()) {
      for (Label label : entry.getValue()) {
        visitLabelInAttribute(node, rule, label, entry.getKey(), outgoingEdges.get(entry.getKey()));
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
        addEdge(visibilityTarget, null, outgoingEdges);
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
