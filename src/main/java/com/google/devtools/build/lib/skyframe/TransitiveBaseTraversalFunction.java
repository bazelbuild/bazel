// Copyright 2015 Google Inc. All rights reserved.
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
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException2;

import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * This class can be extended to define {@link SkyFunction}s that traverse a target and its
 * transitive dependencies and return values based on that traversal.
 *
 * <p>The {@code TProcessedTargets} type parameter represents the result of processing a target and
 * its transitive dependencies.
 *
 * <p>{@code TransitiveBaseTraversalFunction} asks for one to be constructed via {@link
 * #processTarget}, and then asks for it to be updated based on the current target's
 * attributes' dependencies via {@link #processDeps}, and then asks for it to be updated based
 * on the current target' aspects' dependencies via {@link #processDeps}. Finally, it calls
 * {@link #computeSkyValue} with the {#code ProcessedTargets} to get the {@link SkyValue} to
 * return.
 */
abstract class TransitiveBaseTraversalFunction<TProcessedTargets>
    implements SkyFunction {

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

  abstract TProcessedTargets processTarget(Label label,
      TargetAndErrorIfAny targetAndErrorIfAny);

  abstract void processDeps(TProcessedTargets processedTargets, EventHandler eventHandler,
      TargetAndErrorIfAny targetAndErrorIfAny,
      Iterable<Entry<SkyKey, ValueOrException2<NoSuchPackageException, NoSuchTargetException>>>
          depEntries);

  /**
   * Returns a {@link SkyValue} based on the target and any errors it has, and the values
   * accumulated across it and a traversal of its transitive dependencies.
   */
  abstract SkyValue computeSkyValue(TargetAndErrorIfAny targetAndErrorIfAny,
      TProcessedTargets processedTargets);

  @Override
  public SkyValue compute(SkyKey key, Environment env)
      throws TransitiveBaseTraversalFunctionException {
    Label label = (Label) key.argument();
    LoadTargetResults loadTargetResults;
    try {
      loadTargetResults = loadTarget(env, label);
    } catch (NoSuchTargetException e) {
      throw new TransitiveBaseTraversalFunctionException(e);
    } catch (NoSuchPackageException e) {
      throw new TransitiveBaseTraversalFunctionException(e);
    }
    LoadTargetResultsType loadTargetResultsType = loadTargetResults.getType();
    if (loadTargetResultsType.equals(LoadTargetResultsType.VALUES_MISSING)) {
      return null;
    }
    Preconditions.checkState(
        loadTargetResultsType.equals(LoadTargetResultsType.TARGET_AND_ERROR_IF_ANY),
        loadTargetResultsType);
    TargetAndErrorIfAny targetAndErrorIfAny = (TargetAndErrorIfAny) loadTargetResults;
    TProcessedTargets processedTargets = processTarget(label, targetAndErrorIfAny);

    // Process deps from attributes and conservative aspects of current target.
    Iterable<SkyKey> labelDepKeys = getLabelDepKeys(targetAndErrorIfAny.getTarget());
    Iterable<SkyKey> labelAspectKeys =
        getConservativeLabelAspectKeys(targetAndErrorIfAny.getTarget());
    Iterable<SkyKey> depAndAspectKeys = Iterables.concat(labelDepKeys, labelAspectKeys);

    Set<Entry<SkyKey, ValueOrException2<NoSuchPackageException, NoSuchTargetException>>>
        depsAndAspectEntries = env.getValuesOrThrow(depAndAspectKeys,
        NoSuchPackageException.class, NoSuchTargetException.class).entrySet();
    processDeps(processedTargets, env.getListener(), targetAndErrorIfAny, depsAndAspectEntries);
    if (env.valuesMissing()) {
      return null;
    }


    // Process deps from strict aspects.
    labelAspectKeys = getStrictLabelAspectKeys(targetAndErrorIfAny.getTarget(), env);
    Set<Entry<SkyKey, ValueOrException2<NoSuchPackageException, NoSuchTargetException>>>
        labelAspectEntries = env.getValuesOrThrow(labelAspectKeys, NoSuchPackageException.class,
        NoSuchTargetException.class).entrySet();
    processDeps(processedTargets, env.getListener(), targetAndErrorIfAny, labelAspectEntries);
    if (env.valuesMissing()) {
      return null;
    }

    return computeSkyValue(targetAndErrorIfAny, processedTargets);
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((Label) skyKey.argument()));
  }

  /**
   * Return an Iterable of SkyKeys corresponding to the Aspect-related dependencies of target.
   *
   *  <p>This method may return a precise set of aspect keys, but may need to request additional
   *  dependencies from the env to do so.
   *
   *  <p>Subclasses should implement only one of #getStrictLabelAspectKeys and
   *  @getConservativeLabelAspectKeys.
   */
  protected abstract Iterable<SkyKey> getStrictLabelAspectKeys(Target target, Environment env);

  /**
   * Return an Iterable of SkyKeys corresponding to the Aspect-related dependencies of target.
   *
   *  <p>This method may return a conservative over-approximation of the exact set.
   */
  protected abstract Iterable<SkyKey> getConservativeLabelAspectKeys(Target target);

  private Iterable<SkyKey> getLabelDepKeys(Target target) {
    List<SkyKey> depKeys = Lists.newArrayList();
    for (Label depLabel : getLabelDeps(target)) {
      depKeys.add(getKey(depLabel));
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

  protected void maybeReportErrorAboutMissingEdge(Target target, Label depLabel,
      NoSuchThingException e, EventHandler eventHandler) {
    if (e instanceof NoSuchTargetException) {
      NoSuchTargetException nste = (NoSuchTargetException) e;
      if (depLabel.equals(nste.getLabel())) {
        eventHandler.handle(Event.error(TargetUtils.getLocationMaybe(target),
            TargetUtils.formatMissingEdge(target, depLabel, e)));
      }
    } else if (e instanceof NoSuchPackageException) {
      NoSuchPackageException nspe = (NoSuchPackageException) e;
      if (nspe.getPackageId().equals(depLabel.getPackageIdentifier())) {
        eventHandler.handle(Event.error(TargetUtils.getLocationMaybe(target),
            TargetUtils.formatMissingEdge(target, depLabel, e)));
      }
    }
  }

  enum LoadTargetResultsType {
    VALUES_MISSING,
    TARGET_AND_ERROR_IF_ANY
  }

  interface LoadTargetResults {
    LoadTargetResultsType getType();
  }

  private static class ValuesMissing implements LoadTargetResults {

    private static final ValuesMissing INSTANCE = new ValuesMissing();

    private ValuesMissing() {}

    @Override
    public LoadTargetResultsType getType() {
      return LoadTargetResultsType.VALUES_MISSING;
    }
  }

  interface TargetAndErrorIfAny {

    boolean isPackageLoadedSuccessfully();

    @Nullable NoSuchTargetException getErrorLoadingTarget();

    Target getTarget();
  }

  private static class TargetAndErrorIfAnyImpl implements TargetAndErrorIfAny, LoadTargetResults {

    private final boolean packageLoadedSuccessfully;
    @Nullable private final NoSuchTargetException errorLoadingTarget;
    private final Target target;

    private TargetAndErrorIfAnyImpl(boolean packageLoadedSuccessfully,
        @Nullable NoSuchTargetException errorLoadingTarget, Target target) {
      this.packageLoadedSuccessfully = packageLoadedSuccessfully;
      this.errorLoadingTarget = errorLoadingTarget;
      this.target = target;
    }

    @Override
    public LoadTargetResultsType getType() {
      return LoadTargetResultsType.TARGET_AND_ERROR_IF_ANY;
    }

    @Override
    public boolean isPackageLoadedSuccessfully() {
      return packageLoadedSuccessfully;
    }

    @Override
    @Nullable
    public NoSuchTargetException getErrorLoadingTarget() {
      return errorLoadingTarget;
    }

    @Override
    public Target getTarget() {
      return target;
    }
  }

  private static LoadTargetResults loadTarget(Environment env, Label label)
      throws NoSuchTargetException, NoSuchPackageException {
    SkyKey packageKey = PackageValue.key(label.getPackageIdentifier());
    SkyKey targetKey = TargetMarkerValue.key(label);

    boolean packageLoadedSuccessfully;
    Target target;
    NoSuchTargetException errorLoadingTarget = null;
    try {
      TargetMarkerValue targetValue = (TargetMarkerValue) env.getValueOrThrow(targetKey,
          NoSuchTargetException.class, NoSuchPackageException.class);
      if (targetValue == null) {
        return ValuesMissing.INSTANCE;
      }
      PackageValue packageValue = (PackageValue) env.getValueOrThrow(packageKey,
          NoSuchPackageException.class);
      if (packageValue == null) {
        return ValuesMissing.INSTANCE;
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
        throw e;
      }

      // We know that a Target may be extracted, but we need to get it out of the Package
      // (which is known to be in error).
      Package pkg;
      try {
        PackageValue packageValue = (PackageValue) env.getValueOrThrow(packageKey,
            NoSuchPackageException.class);
        if (packageValue == null) {
          return ValuesMissing.INSTANCE;
        }
        throw new IllegalStateException(
            "Expected bad package: " + label.getPackageIdentifier());
      } catch (NoSuchPackageException nsp) {
        pkg = Preconditions.checkNotNull(nsp.getPackage(), label.getPackageIdentifier());
      }
      try {
        target = pkg.getTarget(label.getName());
      } catch (NoSuchTargetException nste) {
        throw new IllegalStateException("Expected target to exist", nste);
      }

      errorLoadingTarget = e;
      packageLoadedSuccessfully = false;
    }
    return new TargetAndErrorIfAnyImpl(packageLoadedSuccessfully, errorLoadingTarget, target);
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link TransitiveTraversalFunction#compute}.
   */
  static class TransitiveBaseTraversalFunctionException extends SkyFunctionException {
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
