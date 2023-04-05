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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AdvertisedProviderSet;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.skyframe.TargetLoadingUtil.TargetAndErrorIfAny;
import com.google.devtools.build.skyframe.GroupedDeps;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;

/**
 * This class is like {@link TransitiveTargetFunction}, but the values it returns do not contain
 * {@link com.google.devtools.build.lib.collect.nestedset.NestedSet}s. It performs the side-effects
 * of {@link TransitiveTargetFunction} (i.e., ensuring that transitive targets and their packages
 * have been loaded). It evaluates to a {@link TransitiveTraversalValue} that contains the first
 * error message it encountered, and a set of names of providers if the target is a rule.
 */
public class TransitiveTraversalFunction
    extends TransitiveBaseTraversalFunction<
        TransitiveTraversalFunction.FirstErrorMessageAccumulator> {

  @Override
  Label argumentFromKey(SkyKey key) {
    return (Label) key.argument();
  }

  @Override
  SkyKey getKey(Label label) {
    return TransitiveTraversalValue.key(label);
  }

  @Override
  FirstErrorMessageAccumulator processTarget(TargetAndErrorIfAny targetAndErrorIfAny) {
    NoSuchTargetException errorIfAny = targetAndErrorIfAny.getErrorLoadingTarget();
    String errorMessageIfAny = errorIfAny == null ? null : errorIfAny.getMessage();
    return new FirstErrorMessageAccumulator(errorMessageIfAny);
  }

  @Override
  void processDeps(
      FirstErrorMessageAccumulator accumulator,
      EventHandler eventHandler,
      TargetAndErrorIfAny targetAndErrorIfAny,
      SkyframeLookupResult depEntries,
      Iterable<? extends SkyKey> depKeys) {
    for (SkyKey skyKey : depKeys) {
      TransitiveTraversalValue transitiveTraversalValue;
      try {
        transitiveTraversalValue =
            (TransitiveTraversalValue)
                depEntries.getOrThrow(
                    skyKey, NoSuchPackageException.class, NoSuchTargetException.class);
      } catch (NoSuchPackageException | NoSuchTargetException e) {
        accumulator.maybeSet(e.getMessage());
        continue;
      }
      if (transitiveTraversalValue == null) {
        BugReport.sendBugReport(
            new IllegalStateException(
                "TransitiveTargetValue " + skyKey + " was missing, this should never happen"));
        continue;
      }
      String errorMessage = transitiveTraversalValue.getErrorMessage();
      if (errorMessage != null) {
        accumulator.maybeSet(errorMessage);
      }
    }
  }

  @Override
  protected AdvertisedProviderSet getAdvertisedProviderSet(
      Label toLabel, SkyValue toVal, Environment env) {
    return ((TransitiveTraversalValue) toVal).getProviders();
  }

  @Override
  SkyValue computeSkyValue(
      TargetAndErrorIfAny targetAndErrorIfAny, FirstErrorMessageAccumulator accumulator) {
    boolean targetLoadedSuccessfully = targetAndErrorIfAny.getErrorLoadingTarget() == null;
    String errorMessage = accumulator.getFirstErrorMessage();
    return targetLoadedSuccessfully
        ? TransitiveTraversalValue.forTarget(targetAndErrorIfAny.getTarget(), errorMessage)
        : TransitiveTraversalValue.unsuccessfulTransitiveTraversal(
            errorMessage, targetAndErrorIfAny.getTarget());
  }

  @Override
  Collection<SkyKey> getLabelDepKeys(
      SkyFunction.Environment env, TargetAndErrorIfAny targetAndErrorIfAny)
      throws InterruptedException {
    // As a performance optimization we may already know the deps we are  about to request from
    // last time #compute was called. By requesting these from the environment, we can avoid
    // repeating the label visitation step. For TransitiveTraversalFunction#compute, the label deps
    // dependency group is requested immediately after the package.
    //
    // IMPORTANT: No other package values should be requested inside
    // TransitiveTraversalFunction#compute from this point forward.
    Collection<SkyKey> oldDepKeys = getDepsAfterLastPackageDep(env, /* offset= */ 1);
    return oldDepKeys == null ? super.getLabelDepKeys(env, targetAndErrorIfAny) : oldDepKeys;
  }

  @Override
  Iterable<SkyKey> getStrictLabelAspectDepKeys(
      SkyFunction.Environment env,
      SkyframeLookupResult depMap,
      TargetAndErrorIfAny targetAndErrorIfAny)
      throws InterruptedException {
    // As a performance optimization we may already know the deps we are  about to request from
    // last time #compute was called. By requesting these from the environment, we can avoid
    // repeating the label visitation step. For TransitiveTraversalFunction#compute, the label
    // aspect deps dependency group is requested two groups after the package.
    Collection<SkyKey> oldAspectDepKeys = getDepsAfterLastPackageDep(env, /* offset= */ 2);
    return oldAspectDepKeys == null
        ? super.getStrictLabelAspectDepKeys(env, depMap, targetAndErrorIfAny)
        : oldAspectDepKeys;
  }

  @Nullable
  private static Collection<SkyKey> getDepsAfterLastPackageDep(
      SkyFunction.Environment env, int offset) {
    GroupedDeps temporaryDirectDeps = env.getTemporaryDirectDeps();
    if (temporaryDirectDeps == null) {
      return null;
    }
    int lastPackageDepIndex = getLastPackageValueIndex(temporaryDirectDeps);
    if (lastPackageDepIndex == -1
        || temporaryDirectDeps.numGroups() <= lastPackageDepIndex + offset) {
      return null;
    }
    return temporaryDirectDeps.getDepGroup(lastPackageDepIndex + offset);
  }

  private static int getLastPackageValueIndex(GroupedDeps directDeps) {
    int directDepsNumGroups = directDeps.numGroups();
    for (int i = directDepsNumGroups - 1; i >= 0; i--) {
      List<SkyKey> depGroup = directDeps.getDepGroup(i);
      if (depGroup.size() == 1 && depGroup.get(0).functionName().equals(SkyFunctions.PACKAGE)) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Keeps track of the first error message encountered while traversing itself and its
   * dependencies.
   */
  static class FirstErrorMessageAccumulator {

    @Nullable private String firstErrorMessage;

    public FirstErrorMessageAccumulator(@Nullable String firstErrorMessage) {
      this.firstErrorMessage = firstErrorMessage;
    }

    /** Remembers {@code errorMessage} if it is the first error message. */
    void maybeSet(String errorMessage) {
      Preconditions.checkNotNull(errorMessage);
      if (firstErrorMessage == null) {
        firstErrorMessage = errorMessage;
      }
    }

    @Nullable
    String getFirstErrorMessage() {
      return firstErrorMessage;
    }
  }
}
