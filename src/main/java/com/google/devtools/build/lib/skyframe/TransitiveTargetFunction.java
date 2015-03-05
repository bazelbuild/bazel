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
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * This class builds transitive Target values such that evaluating a Target value is similar to
 * running it through the LabelVisitor.
 */
public class TransitiveTargetFunction implements SkyFunction {

  @Override
  public SkyValue compute(SkyKey key, Environment env) throws TransitiveTargetFunctionException {
    Label label = (Label) key.argument();
    SkyKey packageKey = PackageValue.key(label.getPackageIdentifier());
    SkyKey targetKey = TargetMarkerValue.key(label);
    Target target;
    boolean packageLoadedSuccessfully;
    boolean successfulTransitiveLoading = true;
    NestedSetBuilder<Label> transitiveRootCauses = NestedSetBuilder.stableOrder();
    NoSuchTargetException errorLoadingTarget = null;
    try {
      TargetMarkerValue targetValue = (TargetMarkerValue) env.getValueOrThrow(targetKey,
          NoSuchThingException.class);
      if (targetValue == null) {
        return null;
      }
      PackageValue packageValue = (PackageValue) env.getValueOrThrow(packageKey,
          NoSuchThingException.class);
      if (packageValue == null) {
        return null;
      }

      packageLoadedSuccessfully = true;
      target = packageValue.getPackage().getTarget(label.getName());
    } catch (NoSuchTargetException e) {
      target = e.getTarget();
      if (target == null) {
        throw new TransitiveTargetFunctionException(e);
      }
      successfulTransitiveLoading = false;
      transitiveRootCauses.add(label);
      errorLoadingTarget = e;
      packageLoadedSuccessfully = e.getPackageLoadedSuccessfully();
    } catch (NoSuchPackageException e) {
      throw new TransitiveTargetFunctionException(e);
    } catch (NoSuchThingException e) {
      throw new IllegalStateException(e
          + " not NoSuchTargetException or NoSuchPackageException");
    }

    NestedSetBuilder<PackageIdentifier> transitiveSuccessfulPkgs = NestedSetBuilder.stableOrder();
    NestedSetBuilder<PackageIdentifier> transitiveUnsuccessfulPkgs = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Label> transitiveTargets = NestedSetBuilder.stableOrder();

    PackageIdentifier packageId = target.getPackage().getPackageIdentifier();
    if (packageLoadedSuccessfully) {
      transitiveSuccessfulPkgs.add(packageId);
    } else {
      transitiveUnsuccessfulPkgs.add(packageId);
    }
    transitiveTargets.add(target.getLabel());
    for (Map.Entry<SkyKey, ValueOrException<NoSuchThingException>> entry :
        env.getValuesOrThrow(getLabelDepKeys(target), NoSuchThingException.class).entrySet()) {
      Label depLabel = (Label) entry.getKey().argument();
      TransitiveTargetValue transitiveTargetValue;
      try {
        transitiveTargetValue = (TransitiveTargetValue) entry.getValue().get();
        if (transitiveTargetValue == null) {
          continue;
        }
      } catch (NoSuchPackageException | NoSuchTargetException e) {
        successfulTransitiveLoading = false;
        transitiveRootCauses.add(depLabel);
        maybeReportErrorAboutMissingEdge(target, depLabel, e, env.getListener());
        continue;
      } catch (NoSuchThingException e) {
        throw new IllegalStateException("Unexpected Exception type from TransitiveTargetValue.", e);
      }
      transitiveSuccessfulPkgs.addTransitive(
          transitiveTargetValue.getTransitiveSuccessfulPackages());
      transitiveUnsuccessfulPkgs.addTransitive(
          transitiveTargetValue.getTransitiveUnsuccessfulPackages());
      transitiveTargets.addTransitive(transitiveTargetValue.getTransitiveTargets());
      NestedSet<Label> rootCauses = transitiveTargetValue.getTransitiveRootCauses();
      if (rootCauses != null) {
        successfulTransitiveLoading = false;
        transitiveRootCauses.addTransitive(rootCauses);
        if (transitiveTargetValue.getErrorLoadingTarget() != null) {
          maybeReportErrorAboutMissingEdge(target, depLabel,
              transitiveTargetValue.getErrorLoadingTarget(), env.getListener());
        }
      }
    }

    if (env.valuesMissing()) {
      return null;
    }

    NestedSet<PackageIdentifier> successfullyLoadedPackages = transitiveSuccessfulPkgs.build();
    NestedSet<PackageIdentifier> unsuccessfullyLoadedPackages = transitiveUnsuccessfulPkgs.build();
    NestedSet<Label> loadedTargets = transitiveTargets.build();
    if (successfulTransitiveLoading) {
      return TransitiveTargetValue.successfulTransitiveLoading(successfullyLoadedPackages,
          unsuccessfullyLoadedPackages, loadedTargets);
    } else {
      NestedSet<Label> rootCauses = transitiveRootCauses.build();
      return TransitiveTargetValue.unsuccessfulTransitiveLoading(successfullyLoadedPackages,
          unsuccessfullyLoadedPackages, loadedTargets, rootCauses, errorLoadingTarget);
    }
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((Label) skyKey.argument()));
  }

  private static void maybeReportErrorAboutMissingEdge(Target target, Label depLabel,
      NoSuchThingException e, EventHandler eventHandler) {
    if (e instanceof NoSuchTargetException) {
      NoSuchTargetException nste = (NoSuchTargetException) e;
      if (depLabel.equals(nste.getLabel())) {
        eventHandler.handle(Event.error(TargetUtils.getLocationMaybe(target),
            TargetUtils.formatMissingEdge(target, depLabel, e)));
      }
    } else if (e instanceof NoSuchPackageException) {
      NoSuchPackageException nspe = (NoSuchPackageException) e;
      if (nspe.getPackageName().equals(depLabel.getPackageName())) {
        eventHandler.handle(Event.error(TargetUtils.getLocationMaybe(target),
            TargetUtils.formatMissingEdge(target, depLabel, e)));
      }
    }
  }

  private static Iterable<SkyKey> getLabelDepKeys(Target target) {
    List<SkyKey> depKeys = Lists.newArrayList();
    for (Label depLabel : getLabelDeps(target)) {
      depKeys.add(TransitiveTargetValue.key(depLabel));
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
    labels.addAll(target.getVisibility().getDependencyLabels());
  }

  private static void visitPackageGroup(PackageGroup packageGroup, Set<Label> labels) {
    labels.addAll(packageGroup.getIncludes());
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link TransitiveTargetFunction#compute}.
   */
  private static class TransitiveTargetFunctionException extends SkyFunctionException {
    /**
     * Used to propagate an error from a direct target dependency to the
     * target that depended on it.
     */
    public TransitiveTargetFunctionException(NoSuchPackageException e) {
      super(e, Transience.PERSISTENT);
    }

    /**
     * In nokeep_going mode, used to propagate an error from a direct target dependency to the
     * target that depended on it.
     *
     * In keep_going mode, used the same way, but only for targets that could not be loaded at all
     * (we proceed with transitive loading on targets that contain errors).
     */
    public TransitiveTargetFunctionException(NoSuchTargetException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
