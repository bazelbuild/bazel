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
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.TransitiveTraversalFunction.DummyAccumulator;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException2;

import java.util.Collection;
import java.util.Map.Entry;
import java.util.Set;

/**
 * This class is like {@link TransitiveTargetFunction}, but the values it returns do not contain
 * {@link NestedSet}s. It should be used only when the side-effects of {@link
 * TransitiveTargetFunction} on the skyframe graph are desired (i.e., ensuring that transitive
 * targets and their packages have been loaded).
 */
public class TransitiveTraversalFunction extends TransitiveBaseTraversalFunction<DummyAccumulator> {

  @Override
  SkyKey getKey(Label label) {
    return TransitiveTraversalValue.key(label);
  }

  @Override
  DummyAccumulator processTarget(Label label, TargetAndErrorIfAny targetAndErrorIfAny) {
    return DummyAccumulator.INSTANCE;
  }

  @Override
  void processDeps(DummyAccumulator processedTargets, EventHandler eventHandler,
      TargetAndErrorIfAny targetAndErrorIfAny,
      Iterable<Entry<SkyKey, ValueOrException2<NoSuchPackageException, NoSuchTargetException>>>
          depEntries) {
  }

  protected Collection<Label> getAspectLabels(Target fromTarget, Attribute attr, Label toLabel,
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
      return AspectDefinition.visitAspectsIfRequired(fromTarget, attr, providers).values();
    } catch (NoSuchThingException e) {
      // Do nothing. This error was handled when we computed the corresponding
      // TransitiveTargetValue.
      return ImmutableList.of();
    }
  }

  @Override
  SkyValue computeSkyValue(TargetAndErrorIfAny targetAndErrorIfAny,
      DummyAccumulator processedTargets) {
    NoSuchTargetException errorLoadingTarget = targetAndErrorIfAny.getErrorLoadingTarget();
    return errorLoadingTarget == null
        ? TransitiveTraversalValue.forTarget(targetAndErrorIfAny.getTarget())
        : TransitiveTraversalValue.unsuccessfulTransitiveTraversal(errorLoadingTarget);
  }

 /**
   * Because {@link TransitiveTraversalFunction} is invoked only when its side-effects are desired,
   * this value accumulator has nothing to keep track of.
   */
  static class DummyAccumulator {
    static final DummyAccumulator INSTANCE = new DummyAccumulator();

    private DummyAccumulator() {}
  }
}
