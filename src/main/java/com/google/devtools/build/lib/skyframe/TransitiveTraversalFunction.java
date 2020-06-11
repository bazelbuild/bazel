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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AdvertisedProviderSet;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.util.GroupedList;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException2;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
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
        TransitiveTraversalFunction.DeterministicErrorMessageAccumulator> {

  @Override
  Label argumentFromKey(SkyKey key) {
    return (Label) key.argument();
  }

  @Override
  SkyKey getKey(Label label) {
    return TransitiveTraversalValue.key(label);
  }

  @Override
  DeterministicErrorMessageAccumulator processTarget(
      Label label, TargetAndErrorIfAny targetAndErrorIfAny) {
    NoSuchTargetException errorIfAny = targetAndErrorIfAny.getErrorLoadingTarget();
    String errorMessageIfAny = errorIfAny == null ? null : errorIfAny.getMessage();
    return DeterministicErrorMessageAccumulator.create(errorMessageIfAny);
  }

  @Override
  void processDeps(
      DeterministicErrorMessageAccumulator accumulator,
      EventHandler eventHandler,
      TargetAndErrorIfAny targetAndErrorIfAny,
      Iterable<Map.Entry<SkyKey, ValueOrException2<NoSuchPackageException, NoSuchTargetException>>>
          depEntries) {
    for (Map.Entry<SkyKey, ValueOrException2<NoSuchPackageException, NoSuchTargetException>> entry :
        depEntries) {
      TransitiveTraversalValue transitiveTraversalValue;
      try {
        transitiveTraversalValue = (TransitiveTraversalValue) entry.getValue().get();
        if (transitiveTraversalValue == null) {
          continue;
        }
      } catch (NoSuchPackageException | NoSuchTargetException e) {
        accumulator.maybeSet(e.getMessage());
        continue;
      }
      String errorMessage = transitiveTraversalValue.getErrorMessage();
      if (errorMessage != null) {
        accumulator.maybeSet(errorMessage);
      }
    }
  }

  @Nullable
  @Override
  protected AdvertisedProviderSet getAdvertisedProviderSet(
      Label toLabel,
      @Nullable ValueOrException2<NoSuchPackageException, NoSuchTargetException> toVal,
      Environment env) {
    if (toVal == null) {
      return null;
    }
    try {
      return ((TransitiveTraversalValue) toVal.get()).getProviders();
    } catch (NoSuchThingException e) {
      // Do nothing interesting. This error was handled when we computed the corresponding
      // TransitiveTargetValue.
      return null;
    }
  }

  @Override
  SkyValue computeSkyValue(
      TargetAndErrorIfAny targetAndErrorIfAny, DeterministicErrorMessageAccumulator accumulator) {
    boolean targetLoadedSuccessfully = targetAndErrorIfAny.getErrorLoadingTarget() == null;
    String errorMessage = accumulator.getErrorMessage();
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
    Collection<SkyKey> oldDepKeys = getDepsAfterLastPackageDep(env, /*offset=*/ 1);
    return oldDepKeys == null ? super.getLabelDepKeys(env, targetAndErrorIfAny) : oldDepKeys;
  }

  @Override
  Iterable<SkyKey> getStrictLabelAspectDepKeys(
      SkyFunction.Environment env,
      Map<SkyKey, ValueOrException2<NoSuchPackageException, NoSuchTargetException>> depMap,
      TargetAndErrorIfAny targetAndErrorIfAny)
      throws InterruptedException {
    // As a performance optimization we may already know the deps we are  about to request from
    // last time #compute was called. By requesting these from the environment, we can avoid
    // repeating the label visitation step. For TransitiveTraversalFunction#compute, the label
    // aspect deps dependency group is requested two groups after the package.
    Collection<SkyKey> oldAspectDepKeys = getDepsAfterLastPackageDep(env, /*offset=*/ 2);
    return oldAspectDepKeys == null
        ? super.getStrictLabelAspectDepKeys(env, depMap, targetAndErrorIfAny)
        : oldAspectDepKeys;
  }

  @Nullable
  private static Collection<SkyKey> getDepsAfterLastPackageDep(
      SkyFunction.Environment env, int offset) {
    GroupedList<SkyKey> temporaryDirectDeps = env.getTemporaryDirectDeps();
    if (temporaryDirectDeps == null) {
      return null;
    }
    int lastPackageDepIndex = getLastPackageValueIndex(temporaryDirectDeps);
    if (lastPackageDepIndex == -1
        || temporaryDirectDeps.listSize() <= lastPackageDepIndex + offset) {
      return null;
    }
    return temporaryDirectDeps.get(lastPackageDepIndex + offset);
  }

  private static int getLastPackageValueIndex(GroupedList<SkyKey> directDeps) {
    int directDepsNumGroups = directDeps.listSize();
    for (int i = directDepsNumGroups - 1; i >= 0; i--) {
      List<SkyKey> depGroup = directDeps.get(i);
      if (depGroup.size() == 1 && depGroup.get(0).functionName().equals(SkyFunctions.PACKAGE)) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Keeps track of a deterministic error message encountered while traversing itself and its
   * dependencies: either the error it was initialized with, or the shortest error it encounters,
   * with ties broken alphabetically.
   *
   * <p>This preserves the behavior that the local target's error is the most important, and is
   * cheap (constant-time) to compute the comparison between strings, unless they have the same
   * length, which is unlikely.
   */
  interface DeterministicErrorMessageAccumulator {
    @Nullable
    String getErrorMessage();

    default void maybeSet(String errorMessage) {}

    static DeterministicErrorMessageAccumulator create(@Nullable String errorMessage) {
      if (errorMessage != null) {
        return () -> errorMessage;
      }
      return new UpdateableErrorMessageAccumulator();
    }

    class UpdateableErrorMessageAccumulator implements DeterministicErrorMessageAccumulator {
      private static final Comparator<String> LENGTH_THEN_ALPHABETICAL =
          Comparator.nullsLast(
              Comparator.comparingInt(String::length).thenComparing(Comparator.naturalOrder()));

      @Nullable private String errorMessage;

      @Override
      public void maybeSet(String errorMessage) {
        Preconditions.checkNotNull(errorMessage);
        if (LENGTH_THEN_ALPHABETICAL.compare(this.errorMessage, errorMessage) > 0) {
          this.errorMessage = errorMessage;
        }
      }

      @Nullable
      @Override
      public String getErrorMessage() {
        return errorMessage;
      }
    }
  }
}
