// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AdvertisedProviderSet;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.LabelVisitationUtils;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.TargetLoadingUtil.TargetAndErrorIfAny;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;

/**
 * This class can be extended to define {@link SkyFunction}s that traverse a target and its
 * transitive dependencies and return values based on that traversal.
 *
 * <p>The {@code ProcessedTargetsT} type parameter represents the result of processing a target and
 * its transitive dependencies.
 *
 * <p>{@code TransitiveBaseTraversalFunction} asks for one to be constructed via {@link
 * #processTarget}, and then asks for it to be updated based on the current target's attributes'
 * dependencies via {@link #processDeps}, and then asks for it to be updated based on the current
 * target' aspects' dependencies via {@link #processDeps}. Finally, it calls {@link
 * #computeSkyValue} with the {#code ProcessedTargets} to get the {@link SkyValue} to return.
 */
public abstract class TransitiveBaseTraversalFunction<ProcessedTargetsT> implements SkyFunction {
  /**
   * Returns a {@link SkyKey} corresponding to the traversal of a target specified by {@code label}
   * and its transitive dependencies.
   *
   * <p>Extenders of this class should implement this function to return a key with their
   * specialized {@link SkyFunction}'s name.
   *
   * <p>{@link TransitiveBaseTraversalFunction} calls this for each dependency of a target, and
   * then gets their values from the environment.
   *
   * <p>The key's {@link SkyFunction} may throw at most {@link NoSuchPackageException} and
   * {@link NoSuchTargetException}. Other exception types are not handled by {@link
   * TransitiveBaseTraversalFunction}.
   */
  abstract SkyKey getKey(Label label);

  abstract ProcessedTargetsT processTarget(TargetAndErrorIfAny targetAndErrorIfAny);

  abstract void processDeps(
      ProcessedTargetsT processedTargets,
      EventHandler eventHandler,
      TargetAndErrorIfAny targetAndErrorIfAny,
      SkyframeLookupResult depEntries,
      Iterable<? extends SkyKey> depKeys);

  /**
   * Returns a {@link SkyValue} based on the target and any errors it has, and the values
   * accumulated across it and a traversal of its transitive dependencies.
   */
  abstract SkyValue computeSkyValue(
      TargetAndErrorIfAny targetAndErrorIfAny, ProcessedTargetsT processedTargets);

  abstract Label argumentFromKey(SkyKey key);

  @Nullable
  @Override
  public SkyValue compute(SkyKey key, Environment env)
      throws TransitiveBaseTraversalFunctionException, InterruptedException {
    Label label = argumentFromKey(key);
    TargetAndErrorIfAny targetAndErrorIfAny;
    try {
      targetAndErrorIfAny = loadTarget(env, label);
    } catch (NoSuchTargetException e) {
      throw new TransitiveBaseTraversalFunctionException(e);
    } catch (NoSuchPackageException e) {
      throw new TransitiveBaseTraversalFunctionException(e);
    }
    if (targetAndErrorIfAny == null) {
      return null;
    }

    // Process deps from attributes. It is essential that the last getValue(s) call we made to
    // skyframe for building this node was for the corresponding PackageValue.
    Collection<SkyKey> labelDepKeys = getLabelDepKeys(env, targetAndErrorIfAny);

    SkyframeLookupResult depMap = env.getValuesAndExceptions(labelDepKeys);
    if (env.valuesMissing()) {
      return null;
    }
    // Process deps from aspects. It is essential that the second-to-last getValue(s) call we
    // made to skyframe for building this node was for the corresponding PackageValue.
    Iterable<SkyKey> labelAspectKeys =
        getStrictLabelAspectDepKeys(env, depMap, targetAndErrorIfAny);
    SkyframeLookupResult labelAspectEntries = env.getValuesAndExceptions(labelAspectKeys);
    if (env.valuesMissing()) {
      return null;
    }

    ProcessedTargetsT processedTargets = processTarget(targetAndErrorIfAny);
    processDeps(processedTargets, env.getListener(), targetAndErrorIfAny, depMap, labelDepKeys);
    processDeps(
        processedTargets,
        env.getListener(),
        targetAndErrorIfAny,
        labelAspectEntries,
        labelAspectKeys);

    return computeSkyValue(targetAndErrorIfAny, processedTargets);
  }

  Collection<SkyKey> getLabelDepKeys(
      SkyFunction.Environment env, TargetAndErrorIfAny targetAndErrorIfAny)
      throws InterruptedException {
    ImmutableSet.Builder<SkyKey> depsBuilder = ImmutableSet.builder();
    LabelVisitationUtils.visitTarget(
        targetAndErrorIfAny.getTarget(),
        DependencyFilter.NO_NODEP_ATTRIBUTES_EXCEPT_VISIBILITY,
        (fromTarget, attribute, toLabel) -> depsBuilder.add(getKey(toLabel)));
    return depsBuilder.build();
  }

  Iterable<SkyKey> getStrictLabelAspectDepKeys(
      SkyFunction.Environment env,
      SkyframeLookupResult depMap,
      TargetAndErrorIfAny targetAndErrorIfAny)
      throws InterruptedException {
    return getStrictLabelAspectKeys(targetAndErrorIfAny.getTarget(), depMap, env);
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(argumentFromKey(skyKey));
  }

  /**
   * Return an Iterable of SkyKeys corresponding to the Aspect-related dependencies of target.
   *
   * <p>This method may return a precise set of aspect keys, but may need to request additional
   * dependencies from the env to do so.
   */
  private Iterable<SkyKey> getStrictLabelAspectKeys(
      Target target, SkyframeLookupResult depMap, Environment env) throws InterruptedException {
    if (!(target instanceof Rule rule)) {
      // Aspects can be declared only for Rules.
      return ImmutableList.of();
    }

    if (!rule.hasAspects()) {
      return ImmutableList.of();
    }

    List<SkyKey> depKeys = Lists.newArrayList();
    Multimap<Attribute, Label> transitions =
        rule.getTransitions(DependencyFilter.NO_NODEP_ATTRIBUTES);
    for (Attribute attribute : transitions.keySet()) {
      for (Aspect aspect : attribute.getAspects(rule)) {
        if (hasDepThatSatisfies(aspect, transitions.get(attribute), depMap, env)) {
          AspectDefinition.forEachLabelDepFromAllAttributesOfAspect(
              aspect,
              DependencyFilter.ALL_DEPS,
              (aspectAttribute, aspectLabel) -> depKeys.add(getKey(aspectLabel)));
        }
      }
    }
    return depKeys;
  }

  @Nullable
  protected abstract AdvertisedProviderSet getAdvertisedProviderSet(
      Label toLabel, SkyValue toVal, Environment env) throws InterruptedException;

  private boolean hasDepThatSatisfies(
      Aspect aspect, Iterable<Label> depLabels, SkyframeLookupResult fullDepMap, Environment env)
      throws InterruptedException {
    for (Label depLabel : depLabels) {
      SkyValue toVal;
      try {
        toVal =
            fullDepMap.getOrThrow(
                getKey(depLabel), NoSuchPackageException.class, NoSuchTargetException.class);
      } catch (NoSuchPackageException | NoSuchTargetException e) {
        continue;
      }
      AdvertisedProviderSet advertisedProviderSet = getAdvertisedProviderSet(depLabel, toVal, env);
      if (advertisedProviderSet != null
          && AspectDefinition.satisfies(aspect, advertisedProviderSet)) {
        return true;
      }
    }
    return false;
  }

  @Nullable
  TargetAndErrorIfAny loadTarget(Environment env, Label label)
      throws NoSuchTargetException, NoSuchPackageException, InterruptedException {
    Object o = TargetLoadingUtil.loadTarget(env, label);
    return o instanceof TargetAndErrorIfAny ? (TargetAndErrorIfAny) o : null;
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by {@link
   * TransitiveTraversalFunction#compute}.
   */
  public static class TransitiveBaseTraversalFunctionException extends SkyFunctionException {
    /**
     * Used to propagate an error from a direct target dependency to the target that depended on
     * it.
     */
    public TransitiveBaseTraversalFunctionException(NoSuchPackageException e) {
      super(e, Transience.PERSISTENT);
    }

    /**
     * In nokeep_going mode, used to propagate an error from a direct target dependency to the
     * target that depended on it.
     *
     * <p>In keep_going mode, used the same way, but only for targets that could not be loaded at
     * all (we proceed with transitive loading on targets that contain errors).</p>
     */
    public TransitiveBaseTraversalFunctionException(NoSuchTargetException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
