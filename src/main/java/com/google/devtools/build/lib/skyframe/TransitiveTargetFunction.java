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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AdvertisedProviderSet;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.skyframe.TransitiveTargetFunction.TransitiveTargetValueBuilder;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException2;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * This class builds transitive Target values such that evaluating a Target value is similar to
 * running it through the LabelVisitor.
 */
public class TransitiveTargetFunction
    extends TransitiveBaseTraversalFunction<TransitiveTargetValueBuilder> {

  private final ConfiguredRuleClassProvider ruleClassProvider;

  TransitiveTargetFunction(RuleClassProvider ruleClassProvider) {
    this.ruleClassProvider = (ConfiguredRuleClassProvider) ruleClassProvider;
  }

  @Override
  Label argumentFromKey(SkyKey key) {
    return ((TransitiveTargetKey) key).getLabel();
  }

  @Override
  SkyKey getKey(Label label) {
    return TransitiveTargetKey.of(label);
  }

  @Override
  TransitiveTargetValueBuilder processTarget(Label label, TargetAndErrorIfAny targetAndErrorIfAny) {
    Target target = targetAndErrorIfAny.getTarget();
    boolean packageLoadedSuccessfully = targetAndErrorIfAny.isPackageLoadedSuccessfully();
    return new TransitiveTargetValueBuilder(label, target, packageLoadedSuccessfully);
  }

  @Override
  void processDeps(
      TransitiveTargetValueBuilder builder,
      EventHandler eventHandler,
      TargetAndErrorIfAny targetAndErrorIfAny,
      Iterable<Map.Entry<SkyKey, ValueOrException2<NoSuchPackageException, NoSuchTargetException>>>
          depEntries) {
    boolean successfulTransitiveLoading = builder.isSuccessfulTransitiveLoading();
    Target target = targetAndErrorIfAny.getTarget();
    NestedSetBuilder<Label> transitiveRootCauses = builder.getTransitiveRootCauses();

    for (Map.Entry<SkyKey, ValueOrException2<NoSuchPackageException, NoSuchTargetException>> entry :
        depEntries) {
      Label depLabel = ((TransitiveTargetKey) entry.getKey()).getLabel();
      TransitiveTargetValue transitiveTargetValue;
      try {
        transitiveTargetValue = (TransitiveTargetValue) entry.getValue().get();
        if (transitiveTargetValue == null) {
          continue;
        }
      } catch (NoSuchPackageException | NoSuchTargetException e) {
        successfulTransitiveLoading = false;
        transitiveRootCauses.add(depLabel);
        maybeReportErrorAboutMissingEdge(target, depLabel, e, eventHandler);
        continue;
      }
      builder.getTransitiveTargets().addTransitive(transitiveTargetValue.getTransitiveTargets());
      NestedSet<Label> rootCauses = transitiveTargetValue.getTransitiveRootCauses();
      if (rootCauses != null) {
        successfulTransitiveLoading = false;
        transitiveRootCauses.addTransitive(rootCauses);
        if (transitiveTargetValue.getErrorLoadingTarget() != null) {
          maybeReportErrorAboutMissingEdge(target, depLabel,
              transitiveTargetValue.getErrorLoadingTarget(), eventHandler);
        }
      }

      NestedSet<Class<? extends Fragment>> depFragments =
          transitiveTargetValue.getTransitiveConfigFragments();
      ImmutableList<Class<? extends Fragment>> depFragmentsAsList = depFragments.toList();
      // The simplest collection technique would be to unconditionally add all deps' nested
      // sets to the current target's nested set. But when there's large overlap between their
      // fragment needs, this produces unnecessarily bloated nested sets and a lot of references
      // that don't contribute anything unique to the required fragment set. So we optimize here
      // by completely skipping sets that don't offer anything new. More fine-tuned optimization
      // is possible, but this offers a good balance between simplicity and practical efficiency.
      Set<Class<? extends Fragment>> addedConfigFragments = builder.getConfigFragmentsFromDeps();
      if (!addedConfigFragments.containsAll(depFragmentsAsList)) {
        builder.getTransitiveConfigFragments().addTransitive(depFragments);
        addedConfigFragments.addAll(depFragmentsAsList);
      }
    }
    builder.setSuccessfulTransitiveLoading(successfulTransitiveLoading);
  }

  @Override
  @SuppressWarnings("unchecked")
  public SkyValue computeSkyValue(
      TargetAndErrorIfAny targetAndErrorIfAny, TransitiveTargetValueBuilder builder) {
    Target target = targetAndErrorIfAny.getTarget();
    NoSuchTargetException errorLoadingTarget = targetAndErrorIfAny.getErrorLoadingTarget();

    // Get configuration fragments directly required by this rule.
    if (target instanceof Rule) {
      Rule rule = (Rule) target;

      // Declared by the rule class:
      ConfigurationFragmentPolicy configurationFragmentPolicy =
          rule.getRuleClassObject().getConfigurationFragmentPolicy();
      for (ConfigurationFragmentFactory factory : ruleClassProvider.getConfigurationFragments()) {
        Class<? extends Fragment> fragment = factory.creates();
        // isLegalConfigurationFragment considers both natively declared fragments and Skylark
        // (named) fragments.
        if (configurationFragmentPolicy.isLegalConfigurationFragment(fragment)) {
          addFragmentIfNew(builder, fragment.asSubclass(Fragment.class));
        }
      }

      // Declared by late-bound attributes:
      for (Attribute attr : rule.getAttributes()) {
        if (attr.isLateBound()
            && attr.getLateBoundDefault().getFragmentClass() != null
            && Fragment.class.isAssignableFrom(attr.getLateBoundDefault().getFragmentClass())) {
          addFragmentIfNew(
              builder,
              (Class<? extends Fragment>) // unchecked cast
                  attr.getLateBoundDefault().getFragmentClass());
        }
      }

      // config_setting rules have values like {"some_flag": "some_value"} that need the
      // corresponding fragments in their configurations to properly resolve:
      addFragmentsIfNew(builder, getFragmentsFromRequiredOptions(rule));

      // Fragments to unconditionally include:
      for (Class<? extends Fragment> universalFragment :
          ruleClassProvider.getUniversalFragments()) {
        addFragmentIfNew(builder, universalFragment);
      }
    }

    return builder.build(errorLoadingTarget);
  }

  private Set<Class<? extends Fragment>> getFragmentsFromRequiredOptions(Rule rule) {
    Set<String> requiredOptions =
      rule.getRuleClassObject().getOptionReferenceFunction().apply(rule);
    ImmutableSet.Builder<Class<? extends Fragment>> optionsFragments = new ImmutableSet.Builder<>();
    for (String requiredOption : requiredOptions) {
      Class<? extends Fragment> fragment =
          ruleClassProvider.getConfigurationFragmentForOption(requiredOption);
      // Null values come from CoreOptions, which is implicitly included.
      if (fragment != null) {
        optionsFragments.add(fragment);
      }
    }
    return optionsFragments.build();
  }

  private void addFragmentIfNew(TransitiveTargetValueBuilder builder,
      Class<? extends Fragment> fragment) {
    // This only checks that the deps don't already use this fragment, not the parent rule itself.
    // So duplicates are still possible. We can further optimize if needed.
    if (!builder.getConfigFragmentsFromDeps().contains(fragment)) {
      builder.getTransitiveConfigFragments().add(fragment);
    }
  }

  private void addFragmentsIfNew(
      TransitiveTargetValueBuilder builder, Iterable<? extends Class<?>> fragments) {
    // We take Iterable<?> instead of Iterable<Class<?>> or Iterable<Class<? extends Fragment>>
    // because both of the latter are passed as actual parameters and there's no way to consistently
    // cast to one of them. In actuality, all values are Class<? extends Fragment>, but the values
    // coming from Attribute.java don't have access to the Fragment symbol since Attribute is built
    // in a different library.
    for (Class<?> fragment : fragments) {
      addFragmentIfNew(builder, fragment.asSubclass(Fragment.class));
    }
  }

  @Nullable
  @Override
  protected AdvertisedProviderSet getAdvertisedProviderSet(
      Label toLabel,
      @Nullable ValueOrException2<NoSuchPackageException, NoSuchTargetException> toVal,
      Environment env)
      throws InterruptedException {
    SkyKey packageKey = PackageValue.key(toLabel.getPackageIdentifier());
    Target toTarget;
    try {
      PackageValue pkgValue =
          (PackageValue) env.getValueOrThrow(packageKey, NoSuchPackageException.class);
      if (pkgValue == null) {
        return null;
      }
      Package pkg = pkgValue.getPackage();
      if (pkg.containsErrors()) {
        // Do nothing interesting. This error was handled when we computed the corresponding
        // TransitiveTargetValue.
        return null;
      }
      toTarget = pkgValue.getPackage().getTarget(toLabel.getName());
    } catch (NoSuchThingException e) {
      // Do nothing interesting. This error was handled when we computed the corresponding
      // TransitiveTargetValue.
      return null;
    }
    if (!(toTarget instanceof Rule)) {
      // Aspect can be declared only for Rules.
      return null;
    }
    return ((Rule) toTarget).getRuleClassObject().getAdvertisedProviders();
  }

  private static void maybeReportErrorAboutMissingEdge(
      Target target, Label depLabel, NoSuchThingException e, EventHandler eventHandler) {
    if (e instanceof NoSuchTargetException) {
      NoSuchTargetException nste = (NoSuchTargetException) e;
      if (depLabel.equals(nste.getLabel())) {
        eventHandler.handle(
            Event.error(
                TargetUtils.getLocationMaybe(target),
                TargetUtils.formatMissingEdge(target, depLabel, e)));
      }
    } else if (e instanceof NoSuchPackageException) {
      NoSuchPackageException nspe = (NoSuchPackageException) e;
      if (nspe.getPackageId().equals(depLabel.getPackageIdentifier())) {
        eventHandler.handle(
            Event.error(
                TargetUtils.getLocationMaybe(target),
                TargetUtils.formatMissingEdge(target, depLabel, e)));
      }
    }
  }

  /**
   * Holds values accumulated across the given target and its transitive dependencies for the
   * purpose of constructing a {@link TransitiveTargetValue}.
   *
   * <p>Note that this class is mutable! The {@code successfulTransitiveLoading} property is
   * initialized with the {@code packageLoadedSuccessfully} constructor parameter, and may be
   * modified if a transitive dependency is found to be in error.
   */
  static class TransitiveTargetValueBuilder {
    private boolean successfulTransitiveLoading;
    private final NestedSetBuilder<Label> transitiveTargets;
    private final NestedSetBuilder<Class<? extends Fragment>> transitiveConfigFragments;
    private final Set<Class<? extends Fragment>> configFragmentsFromDeps;
    private final NestedSetBuilder<Label> transitiveRootCauses;

    public TransitiveTargetValueBuilder(Label label, Target target,
        boolean packageLoadedSuccessfully) {
      this.transitiveTargets = NestedSetBuilder.stableOrder();
      this.transitiveConfigFragments = NestedSetBuilder.stableOrder();
      // No need to store directly required fragments that are also required by deps.
      this.configFragmentsFromDeps = new LinkedHashSet<>();
      this.transitiveRootCauses = NestedSetBuilder.stableOrder();

      this.successfulTransitiveLoading = packageLoadedSuccessfully;
      if (!packageLoadedSuccessfully) {
        transitiveRootCauses.add(label);
      }
      transitiveTargets.add(target.getLabel());
    }

    public NestedSetBuilder<Label> getTransitiveTargets() {
      return transitiveTargets;
    }

    public NestedSetBuilder<Class<? extends Fragment>> getTransitiveConfigFragments() {
      return transitiveConfigFragments;
    }

    public Set<Class<? extends Fragment>> getConfigFragmentsFromDeps() {
      return configFragmentsFromDeps;
    }

    public NestedSetBuilder<Label> getTransitiveRootCauses() {
      return transitiveRootCauses;
    }

    public boolean isSuccessfulTransitiveLoading() {
      return successfulTransitiveLoading;
    }

    public void setSuccessfulTransitiveLoading(boolean successfulTransitiveLoading) {
      this.successfulTransitiveLoading = successfulTransitiveLoading;
    }

    public SkyValue build(@Nullable NoSuchTargetException errorLoadingTarget) {
      NestedSet<Label> loadedTargets = transitiveTargets.build();
      NestedSet<Class<? extends Fragment>> configFragments = transitiveConfigFragments.build();
      return successfulTransitiveLoading
          ? TransitiveTargetValue.successfulTransitiveLoading(loadedTargets, configFragments)
          : TransitiveTargetValue.unsuccessfulTransitiveLoading(
              loadedTargets,
              transitiveRootCauses.build(),
              errorLoadingTarget,
              configFragments);
    }
  }
}
