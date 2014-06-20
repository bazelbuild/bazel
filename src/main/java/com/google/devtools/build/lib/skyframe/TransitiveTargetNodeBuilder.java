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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeBuilder;
import com.google.devtools.build.skyframe.NodeBuilderException;
import com.google.devtools.build.skyframe.NodeKey;
import com.google.devtools.build.skyframe.NodeOrException;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * This class builds transitive Target nodes such that evaluating a Target node is similar to
 * running it through the LabelVisitor.
 */
public class TransitiveTargetNodeBuilder implements NodeBuilder {

  @Override
  public Node build(NodeKey key, Environment env) throws TransitiveTargetNodeBuilderException {
    Label label = (Label) key.getNodeName();

    NodeKey packageKey = PackageNode.key(label.getPackageFragment());
    NodeKey targetKey = TargetMarkerNode.key(label);
    Target target;
    boolean packageLoadedSuccessfully;
    boolean successfulTransitiveLoading = true;
    NestedSetBuilder<Label> transitiveRootCauses = NestedSetBuilder.stableOrder();
    NoSuchTargetException errorLoadingTarget = null;
    try {
      TargetMarkerNode targetNode = (TargetMarkerNode) env.getDepOrThrow(targetKey,
          NoSuchThingException.class);      
      if (targetNode == null) {
        return null;
      }
      PackageNode packageNode = (PackageNode) env.getDepOrThrow(packageKey, 
          NoSuchThingException.class);
      
      packageLoadedSuccessfully = true;
      target = packageNode.getPackage().getTarget(label.getName());
    } catch (NoSuchTargetException e) {
      target = e.getTarget();
      if (target == null) {
        throw new TransitiveTargetNodeBuilderException(key, e);
      }
      successfulTransitiveLoading = false;
      transitiveRootCauses.add(label);
      errorLoadingTarget = e;
      packageLoadedSuccessfully = e.getPackageLoadedSuccessfully();
    } catch (NoSuchPackageException e) {
      throw new TransitiveTargetNodeBuilderException(key, e);
    } catch (NoSuchThingException e) {
      throw new IllegalStateException(e
          + " not NoSuchTargetException or NoSuchPackageException");
    }

    NestedSetBuilder<PathFragment> transitiveSuccessfulPkgs = NestedSetBuilder.stableOrder();
    NestedSetBuilder<PathFragment> transitiveUnsuccessfulPkgs = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Label> transitiveTargets = NestedSetBuilder.stableOrder();

    PathFragment packageName = target.getPackage().getNameFragment();
    if (packageLoadedSuccessfully) {
      transitiveSuccessfulPkgs.add(packageName);
    } else {
      transitiveUnsuccessfulPkgs.add(packageName);
    }
    transitiveTargets.add(target.getLabel());
    for (Map.Entry<NodeKey, NodeOrException<NoSuchThingException>> entry :
        env.getDepsOrThrow(getLabelDepKeys(target), NoSuchThingException.class).entrySet()) {
      Label depLabel = (Label) entry.getKey().getNodeName();
      TransitiveTargetNode transitiveTargetNode;
      try {
        transitiveTargetNode = (TransitiveTargetNode) entry.getValue().get();
        if (transitiveTargetNode == null) {
          continue;
        }
      } catch (NoSuchPackageException | NoSuchTargetException e) {
        successfulTransitiveLoading = false;
        transitiveRootCauses.add(depLabel);
        maybeReportErrorAboutMissingEdge(target, depLabel, e, env.getListener());
        continue;
      } catch (NoSuchThingException e) {
        throw new IllegalStateException("Unexpected Exception type from TransitiveTargetNode.", e);
      }
      transitiveSuccessfulPkgs.addTransitive(
          transitiveTargetNode.getTransitiveSuccessfulPackages());
      transitiveUnsuccessfulPkgs.addTransitive(
          transitiveTargetNode.getTransitiveUnsuccessfulPackages());
      transitiveTargets.addTransitive(transitiveTargetNode.getTransitiveTargets());
      NestedSet<Label> rootCauses = transitiveTargetNode.getTransitiveRootCauses();
      if (rootCauses != null) {
        successfulTransitiveLoading = false;
        transitiveRootCauses.addTransitive(rootCauses);
        if (transitiveTargetNode.getErrorLoadingTarget() != null) {
          maybeReportErrorAboutMissingEdge(target, depLabel,
              transitiveTargetNode.getErrorLoadingTarget(), env.getListener());
        }
      }
    }

    if (env.depsMissing()) {
      return null;
    }

    NestedSet<PathFragment> successfullyLoadedPackages = transitiveSuccessfulPkgs.build();
    NestedSet<PathFragment> unsuccessfullyLoadedPackages = transitiveUnsuccessfulPkgs.build();
    NestedSet<Label> loadedTargets = transitiveTargets.build();
    if (successfulTransitiveLoading) {
      return TransitiveTargetNode.successfulTransitiveLoading(successfullyLoadedPackages,
          unsuccessfullyLoadedPackages, loadedTargets);
    } else {
      NestedSet<Label> rootCauses = transitiveRootCauses.build();
      return TransitiveTargetNode.unsuccessfulTransitiveLoading(successfullyLoadedPackages,
          unsuccessfullyLoadedPackages, loadedTargets, rootCauses, errorLoadingTarget);
    }
  }

  @Override
  public String extractTag(NodeKey nodeKey) {
    return Label.print(((Label) nodeKey.getNodeName()));
  }

  private static void maybeReportErrorAboutMissingEdge(Target target, Label depLabel,
      NoSuchThingException e, ErrorEventListener listener) {
    if (e instanceof NoSuchTargetException) {
      NoSuchTargetException nste = (NoSuchTargetException) e;
      if (nste.getLabel().equals(depLabel)) {
        listener.error(TargetUtils.getLocationMaybe(target),
            TargetUtils.formatMissingEdge(target, depLabel, e));
      }
    } else if (e instanceof NoSuchPackageException) {
      NoSuchPackageException nspe = (NoSuchPackageException) e;
      if (nspe.getPackageName().equals(depLabel.getPackageName())) {
        listener.error(TargetUtils.getLocationMaybe(target),
            TargetUtils.formatMissingEdge(target, depLabel, e));
      }
    }
  }

  private static Iterable<NodeKey> getLabelDepKeys(Target target) {
    List<NodeKey> depKeys = Lists.newArrayList();
    for (Label depLabel : getLabelDeps(target)) {
      depKeys.add(TransitiveTargetNode.key(depLabel));
    }
    return depKeys;
  }

  // TODO(bazel-team): Unify this logic with that in LabelVisitor, and possibly DependencyResolver.
  private static Iterable<Label> getLabelDeps(Target target) {
    final Set<Label> labels = new HashSet<>();
    if (target instanceof OutputFile) {
      Rule rule = ((OutputFile) target).getGeneratingRule();
      labels.add(rule.getLabel());
      visitTargetVisibility(target, labels);
    } else if (target instanceof InputFile) {
      visitTargetVisibility(target, labels);
    } else if (target instanceof Rule) {
      visitTargetVisibility(target, labels);
      labels.addAll(((Rule) target).getLabels(Rule.NO_NODEP_ATTRIBUTES));
    } else if (target instanceof PackageGroup) {
      visitPackageGroup((PackageGroup) target, labels);
    }
    return labels;
  }

  private static void visitTargetVisibility(Target target, Set<Label> labels) {
    for (Label label : target.getVisibility().getDependencyLabels()) {
      labels.add(label);
    }
  }

  private static void visitPackageGroup(PackageGroup packageGroup, Set<Label> labels) {
    for (final Label include : packageGroup.getIncludes()) {
      labels.add(include);
    }
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link TransitiveTargetNodeBuilder#build}.
   */
  private static class TransitiveTargetNodeBuilderException extends NodeBuilderException {
    /**
     * Used to propagate an error from a direct target dependency to the
     * target that depended on it.
     */
    public TransitiveTargetNodeBuilderException(NodeKey key, NoSuchPackageException e) {
      super(key, e);
    }

    /**
     * In nokeep_going mode, used to propagate an error from a direct target dependency to the
     * target that depended on it.
     *
     * In keep_going mode, used the same way, but only for targets that could not be loaded at all
     * (we proceed with transitive loading on targets that contain errors).
     */
    public TransitiveTargetNodeBuilderException(NodeKey key, NoSuchTargetException e) {
      super(key, e);
    }
  }
}
