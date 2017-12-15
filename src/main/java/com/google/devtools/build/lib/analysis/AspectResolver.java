// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.configuredtargets.MergedConfiguredTarget;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.AspectFunction;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.ValueOrException2;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Returns the aspects to attach to rule dependencies.
 */
public final class AspectResolver {
  /**
   * Given a list of {@link Dependency} objects, returns a multimap from the {@link Dependency}s to
   * the {@link ConfiguredAspect} instances that should be merged with them.
   *
   * <p>Returns null if the required aspects are not yet available from Skyframe.
   */
  @Nullable
  public static OrderedSetMultimap<Dependency, ConfiguredAspect> resolveAspectDependencies(
      SkyFunction.Environment env,
      Map<SkyKey, ConfiguredTarget> configuredTargetMap,
      Iterable<Dependency> deps,
      @Nullable NestedSetBuilder<Package> transitivePackages)
      throws AspectFunction.AspectCreationException, InterruptedException {
    OrderedSetMultimap<Dependency, ConfiguredAspect> result = OrderedSetMultimap.create();
    Set<SkyKey> allAspectKeys = new HashSet<>();
    for (Dependency dep : deps) {
      allAspectKeys.addAll(getAspectKeys(dep).values());
    }

    Map<SkyKey, ValueOrException2<AspectFunction.AspectCreationException, NoSuchThingException>>
        depAspects = env.getValuesOrThrow(allAspectKeys,
            AspectFunction.AspectCreationException.class, NoSuchThingException.class);

    for (Dependency dep : deps) {
      Map<AspectDescriptor, SkyKey> aspectToKeys = getAspectKeys(dep);

      for (AspectCollection.AspectDeps depAspect : dep.getAspects().getVisibleAspects()) {
        SkyKey aspectKey = aspectToKeys.get(depAspect.getAspect());

        AspectValue aspectValue;
        try {
          // TODO(ulfjack): Catch all thrown AspectCreationException and NoSuchThingException
          // instances and merge them into a single Exception to get full root cause data.
          aspectValue = (AspectValue) depAspects.get(aspectKey).get();
        } catch (NoSuchThingException e) {
          throw new AspectFunction.AspectCreationException(
              String.format(
                  "Evaluation of aspect %s on %s failed: %s",
                  depAspect.getAspect().getAspectClass().getName(),
                  dep.getLabel(),
                  e.toString()));
        }

        if (aspectValue == null) {
          // Dependent aspect has either not been computed yet or is in error.
          return null;
        }

        // Validate that aspect is applicable to "bare" configured target.
        ConfiguredTarget associatedTarget = configuredTargetMap
            .get(ConfiguredTargetValue.key(dep.getLabel(), dep.getConfiguration()));
        if (!aspectMatchesConfiguredTarget(associatedTarget, aspectValue.getAspect())) {
          continue;
        }

        result.put(dep, aspectValue.getConfiguredAspect());
        if (transitivePackages != null) {
          transitivePackages.addTransitive(
              aspectValue.getTransitivePackagesForPackageRootResolution());
        }
      }
    }
    return result;
  }

  /**
   * Merges each direct dependency configured target with the aspects associated with it.
   *
   * <p>Note that the combination of a configured target and its associated aspects are not
   * represented by a Skyframe node. This is because there can possibly be many different
   * combinations of aspects for a particular configured target, so it would result in a
   * combinatorial explosion of Skyframe nodes.
   */
  public static OrderedSetMultimap<Attribute, ConfiguredTarget> mergeAspects(
      OrderedSetMultimap<Attribute, Dependency> depValueNames,
      Map<SkyKey, ConfiguredTarget> depConfiguredTargetMap,
      OrderedSetMultimap<Dependency, ConfiguredAspect> depAspectMap)
      throws MergedConfiguredTarget.DuplicateException {
    OrderedSetMultimap<Attribute, ConfiguredTarget> result = OrderedSetMultimap.create();

    for (Map.Entry<Attribute, Dependency> entry : depValueNames.entries()) {
      Dependency dep = entry.getValue();
      SkyKey depKey = ConfiguredTargetValue.key(dep.getLabel(), dep.getConfiguration());
      ConfiguredTarget depConfiguredTarget = depConfiguredTargetMap.get(depKey);

      result.put(entry.getKey(),
          MergedConfiguredTarget.of(depConfiguredTarget, depAspectMap.get(dep)));
    }

    return result;
  }

  private static Map<AspectDescriptor, SkyKey> getAspectKeys(Dependency dep) {
    HashMap<AspectDescriptor, SkyKey> result = new HashMap<>();
    AspectCollection aspects = dep.getAspects();
    for (AspectCollection.AspectDeps aspectDeps : aspects.getVisibleAspects()) {
      buildAspectKey(aspectDeps, result, dep);
    }
    return result;
  }

  private static AspectValue.AspectKey buildAspectKey(AspectCollection.AspectDeps aspectDeps,
      HashMap<AspectDescriptor, SkyKey> result, Dependency dep) {
    if (result.containsKey(aspectDeps.getAspect())) {
      return (AspectValue.AspectKey) result.get(aspectDeps.getAspect()).argument();
    }

    ImmutableList.Builder<AspectValue.AspectKey> dependentAspects = ImmutableList.builder();
    for (AspectCollection.AspectDeps path : aspectDeps.getDependentAspects()) {
      dependentAspects.add(buildAspectKey(path, result, dep));
    }
    AspectValue.AspectKey aspectKey = AspectValue.createAspectKey(
        dep.getLabel(), dep.getConfiguration(),
        dependentAspects.build(),
        aspectDeps.getAspect(),
        dep.getAspectConfiguration(aspectDeps.getAspect()));
    result.put(aspectKey.getAspectDescriptor(), aspectKey.getSkyKey());
    return aspectKey;
  }

  public static boolean aspectMatchesConfiguredTarget(final ConfiguredTarget dep, Aspect aspect) {
    if (!aspect.getDefinition().applyToFiles() && !(dep.getTarget() instanceof Rule)) {
      return false;
    }
    if (dep.getTarget().getAssociatedRule() == null) {
      // even aspects that 'apply to files' cannot apply to input files.
      return false;
    }
    return dep.satisfies(aspect.getDefinition().getRequiredProviders());
  }
}
