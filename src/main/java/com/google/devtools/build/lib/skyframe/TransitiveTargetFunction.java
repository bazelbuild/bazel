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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;

import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

/**
 * This class builds transitive Target values such that evaluating a Target value is similar to
 * running it through the LabelVisitor.
 */
public class TransitiveTargetFunction implements SkyFunction {

  private final ConfiguredRuleClassProvider ruleClassProvider;

  TransitiveTargetFunction(RuleClassProvider ruleClassProvider) {
    this.ruleClassProvider = (ConfiguredRuleClassProvider) ruleClassProvider;
  }

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
          NoSuchPackageException.class);
      if (packageValue == null) {
        return null;
      }

      packageLoadedSuccessfully = true;
      try {
        target = packageValue.getPackage().getTarget(label.getName());
      } catch (NoSuchTargetException unexpected) {
        // Not expected since the TargetMarkerFunction would have failed earlier if the Target
        // was not present.
        throw new IllegalStateException(unexpected);
      }
    } catch (NoSuchTargetException e) {
      if (!e.hasTarget()) {
        throw new TransitiveTargetFunctionException(e);
      }

      // We know that a Target may be extracted, but we need to get it out of the Package
      // (which is known to be in error).
      Package pkg;
      try {
        PackageValue packageValue = (PackageValue) env.getValueOrThrow(packageKey,
            NoSuchPackageException.class);
        if (packageValue == null) {
          return null;
        }
        throw new IllegalStateException("Expected bad package: " + label.getPackageIdentifier());
      } catch (NoSuchPackageException nsp) {
        pkg = Preconditions.checkNotNull(nsp.getPackage(), label.getPackageIdentifier());
      }
      try {
        target = pkg.getTarget(label.getName());
      } catch (NoSuchTargetException nste) {
        throw new IllegalStateException("Expected target to exist", nste);
      }

      successfulTransitiveLoading = false;
      transitiveRootCauses.add(label);
      errorLoadingTarget = e;
      packageLoadedSuccessfully = false;
    } catch (NoSuchPackageException e) {
      throw new TransitiveTargetFunctionException(e);
    } catch (NoSuchThingException e) {
      throw new IllegalStateException(e + " not NoSuchTargetException or NoSuchPackageException");
    }

    NestedSetBuilder<PackageIdentifier> transitiveSuccessfulPkgs = NestedSetBuilder.stableOrder();
    NestedSetBuilder<PackageIdentifier> transitiveUnsuccessfulPkgs = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Label> transitiveTargets = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Class<? extends BuildConfiguration.Fragment>> transitiveConfigFragments =
        NestedSetBuilder.stableOrder();
    // No need to store directly required fragments that are also required by deps.
    Set<Class<? extends BuildConfiguration.Fragment>> configFragmentsFromDeps =
        new LinkedHashSet<>();

    PackageIdentifier packageId = target.getPackage().getPackageIdentifier();
    if (packageLoadedSuccessfully) {
      transitiveSuccessfulPkgs.add(packageId);
    } else {
      transitiveUnsuccessfulPkgs.add(packageId);
    }
    transitiveTargets.add(target.getLabel());

    // Process deps from attributes of current target.
    Iterable<SkyKey> depKeys = getLabelDepKeys(target);
    successfulTransitiveLoading &= processDeps(env, target, transitiveRootCauses,
        transitiveSuccessfulPkgs, transitiveUnsuccessfulPkgs, transitiveTargets, depKeys,
        transitiveConfigFragments, configFragmentsFromDeps);
    if (env.valuesMissing()) {
      return null;
    }
    // Process deps from aspects.
    depKeys = getLabelAspectKeys(target, env);
    successfulTransitiveLoading &= processDeps(env, target, transitiveRootCauses,
        transitiveSuccessfulPkgs, transitiveUnsuccessfulPkgs, transitiveTargets, depKeys,
        transitiveConfigFragments, configFragmentsFromDeps);
    if (env.valuesMissing()) {
      return null;
    }

    // Get configuration fragments directly required by this target.
    if (target instanceof Rule) {
      Set<Class<?>> configFragments =
          target.getAssociatedRule().getRuleClassObject().getRequiredConfigurationFragments();
      // An empty result means this rule requires all fragments (which practically means
      // the rule isn't yet declaring its actually needed fragments). So load everything.
      configFragments = configFragments.isEmpty() ? getAllFragments() : configFragments;
      for (Class<?> fragment : configFragments) {
        if (!configFragmentsFromDeps.contains(fragment)) {
          transitiveConfigFragments.add((Class<? extends BuildConfiguration.Fragment>) fragment);
        }
      }
    }

    NestedSet<PackageIdentifier> successfullyLoadedPackages = transitiveSuccessfulPkgs.build();
    NestedSet<PackageIdentifier> unsuccessfullyLoadedPackages = transitiveUnsuccessfulPkgs.build();
    NestedSet<Label> loadedTargets = transitiveTargets.build();
    if (successfulTransitiveLoading) {
      return TransitiveTargetValue.successfulTransitiveLoading(successfullyLoadedPackages,
          unsuccessfullyLoadedPackages, loadedTargets, transitiveConfigFragments.build());
    } else {
      NestedSet<Label> rootCauses = transitiveRootCauses.build();
      return TransitiveTargetValue.unsuccessfulTransitiveLoading(successfullyLoadedPackages,
          unsuccessfullyLoadedPackages, loadedTargets, rootCauses, errorLoadingTarget,
          transitiveConfigFragments.build());
    }
  }

  /**
   * Returns every configuration fragment known to the system.
   */
  private Set<Class<?>> getAllFragments() {
    ImmutableSet.Builder<Class<?>> builder =
        ImmutableSet.builder();
    for (ConfigurationFragmentFactory factory : ruleClassProvider.getConfigurationFragments()) {
      builder.add(factory.creates());
    }
    return builder.build();
  }

  private boolean processDeps(Environment env, Target target,
      NestedSetBuilder<Label> transitiveRootCauses,
      NestedSetBuilder<PackageIdentifier> transitiveSuccessfulPkgs,
      NestedSetBuilder<PackageIdentifier> transitiveUnsuccessfulPkgs,
      NestedSetBuilder<Label> transitiveTargets, Iterable<SkyKey> depKeys,
      NestedSetBuilder<Class<? extends BuildConfiguration.Fragment>> transitiveConfigFragments,
      Set<Class<? extends BuildConfiguration.Fragment>> addedConfigFragments) {
    boolean successfulTransitiveLoading = true;
    for (Entry<SkyKey, ValueOrException<NoSuchThingException>> entry :
        env.getValuesOrThrow(depKeys, NoSuchThingException.class).entrySet()) {
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

      NestedSet<Class<? extends BuildConfiguration.Fragment>> depFragments =
          transitiveTargetValue.getTransitiveConfigFragments();
      Collection<Class<? extends BuildConfiguration.Fragment>> depFragmentsAsCollection =
          depFragments.toCollection();
      // The simplest collection technique would be to unconditionally add all deps' nested
      // sets to the current target's nested set. But when there's large overlap between their
      // fragment needs, this produces unnecessarily bloated nested sets and a lot of references
      // that don't contribute anything unique to the required fragment set. So we optimize here
      // by completely skipping sets that don't offer anything new. More fine-tuned optimization
      // is possible, but this offers a good balance between simplicity and practical efficiency.
      if (!addedConfigFragments.containsAll(depFragmentsAsCollection)) {
        transitiveConfigFragments.addTransitive(depFragments);
        addedConfigFragments.addAll(depFragmentsAsCollection);
      }
    }
    return successfulTransitiveLoading;
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

  private static Iterable<SkyKey> getLabelAspectKeys(Target target, Environment env) {
    List<SkyKey> depKeys = Lists.newArrayList();
    if (target instanceof Rule) {
      Multimap<Attribute, Label> transitions =
          ((Rule) target).getTransitions(Rule.NO_NODEP_ATTRIBUTES);
      for (Entry<Attribute, Label> entry : transitions.entries()) {
        SkyKey packageKey = PackageValue.key(entry.getValue().getPackageIdentifier());
        try {
          PackageValue pkgValue = (PackageValue) env.getValueOrThrow(packageKey,
              NoSuchThingException.class);
          if (pkgValue == null) {
            continue;
          }
          Collection<Label> labels = AspectDefinition.visitAspectsIfRequired(target, entry.getKey(),
              pkgValue.getPackage().getTarget(entry.getValue().getName())).values();
          for (Label label : labels) {
            depKeys.add(TransitiveTargetValue.key(label));
          }
        } catch (NoSuchThingException e) {
          // Do nothing. This error was handled when we computed the corresponding
          // TransitiveTargetValue.
        }
      }
    }
    return depKeys;
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
      visitRule(target, labels);
    } else if (target instanceof PackageGroup) {
      visitPackageGroup((PackageGroup) target, labels);
    }
    return labels;
  }

  private static void visitRule(Target target, Set<Label> labels) {
    labels.addAll(((Rule) target).getLabels(Rule.NO_NODEP_ATTRIBUTES));
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
