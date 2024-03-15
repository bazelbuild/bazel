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

import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AdvertisedProviderSet;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.skyframe.TransitiveTargetFunction.TransitiveTargetValueBuilder;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import javax.annotation.Nullable;

/**
 * This class builds transitive Target values such that evaluating a Target value is similar to
 * running it through the LabelVisitor.
 */
final class TransitiveTargetFunction
    extends TransitiveBaseTraversalFunction<TransitiveTargetValueBuilder> {

  @Override
  Label argumentFromKey(SkyKey key) {
    return ((TransitiveTargetKey) key).getLabel();
  }

  @Override
  SkyKey getKey(Label label) {
    return TransitiveTargetKey.of(label);
  }

  @Override
  TransitiveTargetValueBuilder processTarget(
      TargetLoadingUtil.TargetAndErrorIfAny targetAndErrorIfAny) {
    Target target = targetAndErrorIfAny.getTarget();
    boolean packageLoadedSuccessfully = targetAndErrorIfAny.isPackageLoadedSuccessfully();
    return new TransitiveTargetValueBuilder(target, packageLoadedSuccessfully);
  }

  @Override
  void processDeps(
      TransitiveTargetValueBuilder builder,
      EventHandler eventHandler,
      TargetLoadingUtil.TargetAndErrorIfAny targetAndErrorIfAny,
      SkyframeLookupResult depEntries,
      Iterable<? extends SkyKey> depKeys) {
    boolean successfulTransitiveLoading = builder.isSuccessfulTransitiveLoading();
    Target target = targetAndErrorIfAny.getTarget();

    for (SkyKey skyKey : depKeys) {
      Label depLabel = ((TransitiveTargetKey) skyKey).getLabel();
      TransitiveTargetValue transitiveTargetValue;
      try {
        transitiveTargetValue =
            (TransitiveTargetValue)
                depEntries.getOrThrow(
                    skyKey, NoSuchPackageException.class, NoSuchTargetException.class);
      } catch (NoSuchPackageException | NoSuchTargetException e) {
        successfulTransitiveLoading = false;
        maybeReportErrorAboutMissingEdge(target, depLabel, e, eventHandler);
        continue;
      }
      if (transitiveTargetValue == null) {
        BugReport.sendNonFatalBugReport(
            new IllegalStateException(
                "TransitiveTargetValue " + skyKey + " was missing, this should never happen"));
        continue;
      }
      builder.getTransitiveTargets().addTransitive(transitiveTargetValue.getTransitiveTargets());
      if (transitiveTargetValue.encounteredLoadingError()) {
        successfulTransitiveLoading = false;
        if (transitiveTargetValue.getErrorLoadingTarget() != null) {
          maybeReportErrorAboutMissingEdge(target, depLabel,
              transitiveTargetValue.getErrorLoadingTarget(), eventHandler);
        }
      }
    }

    builder.setSuccessfulTransitiveLoading(successfulTransitiveLoading);
  }

  @Override
  public SkyValue computeSkyValue(
      TargetLoadingUtil.TargetAndErrorIfAny targetAndErrorIfAny,
      TransitiveTargetValueBuilder builder) {
    NoSuchTargetException errorLoadingTarget = targetAndErrorIfAny.getErrorLoadingTarget();
    return builder.build(errorLoadingTarget);
  }

  @Nullable
  @Override
  protected AdvertisedProviderSet getAdvertisedProviderSet(
      Label toLabel, SkyValue toVal, Environment env) throws InterruptedException {
    SkyKey packageKey = toLabel.getPackageIdentifier();
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

    public TransitiveTargetValueBuilder(Target target, boolean packageLoadedSuccessfully) {
      this.transitiveTargets = NestedSetBuilder.stableOrder();
      this.successfulTransitiveLoading = packageLoadedSuccessfully;
      transitiveTargets.add(target.getLabel());
    }

    public NestedSetBuilder<Label> getTransitiveTargets() {
      return transitiveTargets;
    }

    public boolean isSuccessfulTransitiveLoading() {
      return successfulTransitiveLoading;
    }

    public void setSuccessfulTransitiveLoading(boolean successfulTransitiveLoading) {
      this.successfulTransitiveLoading = successfulTransitiveLoading;
    }

    public SkyValue build(@Nullable NoSuchTargetException errorLoadingTarget) {
      NestedSet<Label> loadedTargets = transitiveTargets.build();
      return successfulTransitiveLoading
          ? TransitiveTargetValue.successfulTransitiveLoading(loadedTargets)
          : TransitiveTargetValue.unsuccessfulTransitiveLoading(loadedTargets, errorLoadingTarget);
    }
  }
}
