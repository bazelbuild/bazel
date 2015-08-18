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

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectFactory;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.TransitiveTraversalFunction.DummyAccumulator;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException2;

import java.util.Map.Entry;

/**
 * This class is like {@link TransitiveTargetFunction}, but the values it returns do not contain
 * {@link NestedSet}s. It should be used only when the side-effects of {@link
 * TransitiveTargetFunction} are desired (i.e., loading transitive targets and their packages, and
 * emitting error events).
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
    Target target = targetAndErrorIfAny.getTarget();
    for (Entry<SkyKey, ValueOrException2<NoSuchPackageException, NoSuchTargetException>> entry :
        depEntries) {
      Label depLabel = (Label) entry.getKey().argument();
      TransitiveTraversalValue transitiveTraversalValue;
      try {
        transitiveTraversalValue = (TransitiveTraversalValue) entry.getValue().get();
        if (transitiveTraversalValue == null) {
          continue;
        }
      } catch (NoSuchPackageException | NoSuchTargetException e) {
        maybeReportErrorAboutMissingEdge(target, depLabel, e, eventHandler);
        continue;
      }
      if (transitiveTraversalValue.getErrorLoadingTarget() != null) {
        maybeReportErrorAboutMissingEdge(target, depLabel,
            transitiveTraversalValue.getErrorLoadingTarget(), eventHandler);
      }
    }
  }

  @Override
  SkyValue computeSkyValue(TargetAndErrorIfAny targetAndErrorIfAny,
      DummyAccumulator processedTargets) {
    NoSuchTargetException errorLoadingTarget = targetAndErrorIfAny.getErrorLoadingTarget();
    return errorLoadingTarget == null
        ? TransitiveTraversalValue.SUCCESSFUL_TRANSITIVE_TRAVERSAL_VALUE
        : TransitiveTraversalValue.unsuccessfulTransitiveTraversal(errorLoadingTarget);
  }

  @Override
  protected Iterable<SkyKey> getStrictLabelAspectKeys(Target target, Environment env) {
    return ImmutableSet.of();
  }

  @Override
  protected Iterable<SkyKey> getConservativeLabelAspectKeys(Target target) {
    if (!(target instanceof Rule)) {
      return ImmutableSet.of();
    }
    Rule rule = (Rule) target;
    Multimap<Attribute, Label> attibuteMap = LinkedHashMultimap.create();
    for (Attribute attribute : rule.getTransitions(Rule.NO_NODEP_ATTRIBUTES).keys()) {
      for (Class<? extends AspectFactory<?, ?, ?>> aspectFactory : attribute.getAspects()) {
        AspectDefinition.addAllAttributesOfAspect(rule, attibuteMap,
            AspectFactory.Util.create(aspectFactory).getDefinition(), Rule.ALL_DEPS);
      }
    }

    ImmutableSet.Builder<SkyKey> depKeys = new ImmutableSet.Builder<>();
    for (Label label : attibuteMap.values()) {
      depKeys.add(getKey(label));
    }
    return depKeys.build();
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
