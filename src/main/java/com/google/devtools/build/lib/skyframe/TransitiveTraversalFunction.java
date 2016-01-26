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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.TransitiveTraversalFunction.FirstErrorMessageAccumulator;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException2;

import java.util.Collection;
import java.util.Map.Entry;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * This class is like {@link TransitiveTargetFunction}, but the values it returns do not contain
 * {@link NestedSet}s. It performs the side-effects of {@link TransitiveTargetFunction} (i.e.,
 * ensuring that transitive targets and their packages have been loaded). It evaluates to a
 * {@link TransitiveTraversalValue} that contains the first error message it encountered, and a
 * set of names of providers if the target is a rule.
 */
public class TransitiveTraversalFunction
    extends TransitiveBaseTraversalFunction<FirstErrorMessageAccumulator> {

  @Override
  SkyKey getKey(Label label) {
    return TransitiveTraversalValue.key(label);
  }

  @Override
  FirstErrorMessageAccumulator processTarget(Label label, TargetAndErrorIfAny targetAndErrorIfAny) {
    NoSuchTargetException errorIfAny = targetAndErrorIfAny.getErrorLoadingTarget();
    String errorMessageIfAny = errorIfAny == null ? null : errorIfAny.getMessage();
    return new FirstErrorMessageAccumulator(errorMessageIfAny);
  }

  @Override
  void processDeps(
      FirstErrorMessageAccumulator accumulator,
      EventHandler eventHandler,
      TargetAndErrorIfAny targetAndErrorIfAny,
      Iterable<Entry<SkyKey, ValueOrException2<NoSuchPackageException, NoSuchTargetException>>>
          depEntries) {
    for (Entry<SkyKey, ValueOrException2<NoSuchPackageException, NoSuchTargetException>> entry :
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
      String firstErrorMessage = transitiveTraversalValue.getFirstErrorMessage();
      if (firstErrorMessage != null) {
        accumulator.maybeSet(firstErrorMessage);
      }
    }
  }

  protected Collection<Label> getAspectLabels(Rule fromRule, Attribute attr, Label toLabel,
      ValueOrException2<NoSuchPackageException, NoSuchTargetException> toVal, Environment env) {
    try {
      if (toVal == null) {
        return ImmutableList.of();
      }
      TransitiveTraversalValue traversalVal = (TransitiveTraversalValue) toVal.get();
      if (traversalVal == null || traversalVal.getProviders() == null) {
        return ImmutableList.of();
      }
      // Retrieve the providers of the dep from the TransitiveTraversalValue, so we can avoid
      // issuing a dep on its defining Package.
      Set<String> providers = traversalVal.getProviders();
      return AspectDefinition.visitAspectsIfRequired(fromRule, attr, providers,
          DependencyFilter.ALL_DEPS).values();
    } catch (NoSuchThingException e) {
      // Do nothing. This error was handled when we computed the corresponding
      // TransitiveTargetValue.
      return ImmutableList.of();
    }
  }

  @Override
  SkyValue computeSkyValue(
      TargetAndErrorIfAny targetAndErrorIfAny, FirstErrorMessageAccumulator accumulator) {
    boolean targetLoadedSuccessfully = targetAndErrorIfAny.getErrorLoadingTarget() == null;
    String firstErrorMessage = accumulator.getFirstErrorMessage();
    return targetLoadedSuccessfully
        ? TransitiveTraversalValue.forTarget(targetAndErrorIfAny.getTarget(), firstErrorMessage)
        : TransitiveTraversalValue.unsuccessfulTransitiveTraversal(firstErrorMessage);
  }

  @Override
  TargetMarkerValue getTargetMarkerValue(SkyKey targetMarkerKey, Environment env)
      throws NoSuchTargetException, NoSuchPackageException {
    return TargetMarkerFunction.computeTargetMarkerValue(targetMarkerKey, env);
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

    /** Remembers {@param errorMessage} if it is the first error message. */
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
