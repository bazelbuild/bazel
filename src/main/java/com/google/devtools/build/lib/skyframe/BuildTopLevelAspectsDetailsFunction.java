// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Objects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.analysis.AspectCollection;
import com.google.devtools.build.lib.analysis.AspectCollection.AspectCycleOnPathException;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.AspectsListBuilder;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.StarlarkAspect;
import com.google.devtools.build.lib.packages.StarlarkAspectClass;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.server.FailureDetails.Analysis;
import com.google.devtools.build.lib.server.FailureDetails.Analysis.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.AspectValueKey.AspectKey;
import com.google.devtools.build.lib.skyframe.LoadStarlarkAspectFunction.StarlarkAspectLoadingKey;
import com.google.devtools.build.lib.skyframe.LoadStarlarkAspectFunction.StarlarkAspectLoadingValue;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * SkyFunction to load top level aspects, build the dependency relation between them based on the
 * aspects required by the top level aspects and the aspect providers they require and advertise
 * using {@link AspectCollection}.
 *
 * <p>This is needed to compute the relationship between top-level aspects once for all top-level
 * targets in the command. The {@link SkyValue} of this function contains a list of {@link
 * AspectDetails} objects which contain the aspect descriptor and a list of the used aspects this
 * aspect depends on. Then {@link ToplevelStarlarkAspectFunction} adds the target information to
 * create {@link AspectKey}.
 */
public class BuildTopLevelAspectsDetailsFunction implements SkyFunction {
  BuildTopLevelAspectsDetailsFunction() {}

  private static final Interner<BuildTopLevelAspectsDetailsKey>
      buildTopLevelAspectsDetailsKeyInterner = BlazeInterners.newWeakInterner();

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws BuildTopLevelAspectsDetailsFunctionException, InterruptedException {

    BuildTopLevelAspectsDetailsKey topLevelAspectsDetailsKey =
        (BuildTopLevelAspectsDetailsKey) skyKey.argument();
    ImmutableList<AspectClass> topLevelAspectsClasses =
        topLevelAspectsDetailsKey.getTopLevelAspectsClasses();

    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }
    boolean buildTopLevelAspectsDependency =
        starlarkSemantics.getBool(BuildLanguageOptions.INCOMPATIBLE_TOP_LEVEL_ASPECTS_DEPENDENCY);
    if (!buildTopLevelAspectsDependency) {
      // If building a relation between top-level aspects is not required, then we can remove
      // duplicate aspects by keeping the first occurrence of each aspect.
      topLevelAspectsClasses = ImmutableSet.copyOf(topLevelAspectsClasses).asList();

      // Then return a list of indenpendent aspects to be applied on the top-level targets
      ImmutableList<AspectDetails> aspectsDetailsList =
          getIndependentTopLevelAspects(env, topLevelAspectsClasses);
      if (aspectsDetailsList == null) {
        return null; // some aspects are not loaded
      }
      return new BuildTopLevelAspectsDetailsValue(aspectsDetailsList);
    }

    ImmutableList<Aspect> topLevelAspects = getTopLevelAspects(env, topLevelAspectsClasses);

    if (topLevelAspects == null) {
      return null; // some aspects are not loaded
    }

    AspectCollection aspectCollection;
    try {
      aspectCollection = AspectCollection.create(topLevelAspects);
    } catch (AspectCycleOnPathException e) {
      // This exception should never happen because aspects duplicates are not allowed in top-level
      // aspects and their existence should have been caught and reported by `getTopLevelAspects()`.
      env.getListener().handle(Event.error(e.getMessage()));
      throw new BuildTopLevelAspectsDetailsFunctionException(
          new TopLevelAspectsDetailsBuildFailedException(e.getMessage()));
    }
    return new BuildTopLevelAspectsDetailsValue(getTopLevelAspectsDetails(aspectCollection));
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  @Nullable
  private static Map<SkyKey, ValueOrException<AspectCreationException>> loadAspects(
      Environment env, ImmutableList<AspectClass> topLevelAspectsClasses)
      throws InterruptedException {
    ImmutableList.Builder<StarlarkAspectLoadingKey> aspectLoadingKeys = ImmutableList.builder();
    for (AspectClass aspectClass : topLevelAspectsClasses) {
      if (aspectClass instanceof StarlarkAspectClass) {
        aspectLoadingKeys.add(
            LoadStarlarkAspectFunction.createStarlarkAspectLoadingKey(
                (StarlarkAspectClass) aspectClass));
      }
    }

    Map<SkyKey, ValueOrException<AspectCreationException>> loadedAspects =
        env.getValuesOrThrow(aspectLoadingKeys.build(), AspectCreationException.class);
    if (env.valuesMissing()) {
      return null;
    }
    return loadedAspects;
  }

  @Nullable
  private static ImmutableList<AspectDetails> getIndependentTopLevelAspects(
      Environment env, ImmutableList<AspectClass> topLevelAspectsClasses)
      throws InterruptedException, BuildTopLevelAspectsDetailsFunctionException {
    Map<SkyKey, ValueOrException<AspectCreationException>> loadedAspects =
        loadAspects(env, topLevelAspectsClasses);
    if (loadedAspects == null) {
      return null;
    }
    ImmutableList.Builder<AspectDetails> aspectDetailsList = ImmutableList.builder();

    try {
      for (AspectClass aspectClass : topLevelAspectsClasses) {
        if (aspectClass instanceof StarlarkAspectClass) {
          ValueOrException<AspectCreationException> valueOrException =
              loadedAspects.get(
                  LoadStarlarkAspectFunction.createStarlarkAspectLoadingKey(
                      (StarlarkAspectClass) aspectClass));
          StarlarkAspectLoadingValue aspectLoadingValue =
              (StarlarkAspectLoadingValue) valueOrException.get();
          StarlarkAspect starlarkAspect = aspectLoadingValue.getAspect();
          aspectDetailsList.add(
              new AspectDetails(
                  ImmutableList.of(), new AspectDescriptor(starlarkAspect.getAspectClass())));
        } else {
          aspectDetailsList.add(
              new AspectDetails(ImmutableList.of(), new AspectDescriptor(aspectClass)));
        }
      }
    } catch (AspectCreationException e) {
      env.getListener().handle(Event.error(e.getMessage()));
      throw new BuildTopLevelAspectsDetailsFunctionException(
          new TopLevelAspectsDetailsBuildFailedException(e.getMessage()));
    }

    return aspectDetailsList.build();
  }

  @Nullable
  private static ImmutableList<Aspect> getTopLevelAspects(
      Environment env, ImmutableList<AspectClass> topLevelAspectsClasses)
      throws InterruptedException, BuildTopLevelAspectsDetailsFunctionException {
    AspectsListBuilder aspectsList = new AspectsListBuilder();
    Map<SkyKey, ValueOrException<AspectCreationException>> loadedAspects =
        loadAspects(env, topLevelAspectsClasses);
    if (loadedAspects == null) {
      return null;
    }

    for (AspectClass aspectClass : topLevelAspectsClasses) {
      if (aspectClass instanceof StarlarkAspectClass) {
        try {
          ValueOrException<AspectCreationException> valueOrException =
              loadedAspects.get(
                  LoadStarlarkAspectFunction.createStarlarkAspectLoadingKey(
                      (StarlarkAspectClass) aspectClass));
          StarlarkAspectLoadingValue aspectLoadingValue =
              (StarlarkAspectLoadingValue) valueOrException.get();
          StarlarkAspect starlarkAspect = aspectLoadingValue.getAspect();

          starlarkAspect.attachToAspectsList(
              /** baseAspectName= */
              null,
              aspectsList,
              /** inheritedRequiredProviders= */
              ImmutableList.of(),
              /** inheritedAttributeAspects= */
              ImmutableList.of(),
              /** allowAspectsParameters= */
              false);
        } catch (EvalException | AspectCreationException e) {
          env.getListener().handle(Event.error(e.getMessage()));
          throw new BuildTopLevelAspectsDetailsFunctionException(
              new TopLevelAspectsDetailsBuildFailedException(e.getMessage()));
        }
      } else {
        try {
          aspectsList.addAspect((NativeAspectClass) aspectClass);
        } catch (AssertionError ex) {
          env.getListener().handle(Event.error(ex.getMessage()));
          throw new BuildTopLevelAspectsDetailsFunctionException(
              new TopLevelAspectsDetailsBuildFailedException(ex.getMessage()));
        }
      }
    }
    return aspectsList.buildAspects();
  }

  private static Collection<AspectDetails> getTopLevelAspectsDetails(
      AspectCollection aspectCollection) {
    Map<AspectDescriptor, AspectDetails> result = new HashMap<>();
    for (AspectCollection.AspectDeps aspectDeps : aspectCollection.getUsedAspects()) {
      buildAspectDetails(aspectDeps, result);
    }
    return result.values();
  }

  private static AspectDetails buildAspectDetails(
      AspectCollection.AspectDeps aspectDeps, Map<AspectDescriptor, AspectDetails> result) {
    if (result.containsKey(aspectDeps.getAspect())) {
      return result.get(aspectDeps.getAspect());
    }

    ImmutableList.Builder<AspectDetails> dependentAspects = ImmutableList.builder();
    for (AspectCollection.AspectDeps path : aspectDeps.getUsedAspects()) {
      dependentAspects.add(buildAspectDetails(path, result));
    }

    AspectDetails aspectDetails =
        new AspectDetails(dependentAspects.build(), aspectDeps.getAspect());
    result.put(aspectDetails.getAspectDescriptor(), aspectDetails);
    return aspectDetails;
  }

  public static BuildTopLevelAspectsDetailsKey createBuildTopLevelAspectsDetailsKey(
      ImmutableList<AspectClass> aspectClasses) {
    return BuildTopLevelAspectsDetailsKey.createInternal(aspectClasses);
  }

  /** Exceptions thrown from BuildTopLevelAspectsDetailsFunction. */
  public static class BuildTopLevelAspectsDetailsFunctionException extends SkyFunctionException {
    public BuildTopLevelAspectsDetailsFunctionException(
        TopLevelAspectsDetailsBuildFailedException cause) {
      super(cause, Transience.PERSISTENT);
    }
  }

  static final class TopLevelAspectsDetailsBuildFailedException extends Exception
      implements SaneAnalysisException {
    private final DetailedExitCode detailedExitCode;

    private TopLevelAspectsDetailsBuildFailedException(String errorMessage) {
      super(errorMessage);
      this.detailedExitCode =
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(errorMessage)
                  .setAnalysis(Analysis.newBuilder().setCode(Code.ASPECT_CREATION_FAILED))
                  .build());
    }

    @Override
    public DetailedExitCode getDetailedExitCode() {
      return detailedExitCode;
    }
  }

  /**
   * Details of the top-level aspects including the {@link AspectDescriptor} and a list of the
   * aspects it depends on. This is used to build the {@link AspectKey} when combined with
   * configured target details.
   */
  public static final class AspectDetails {
    private final ImmutableList<AspectDetails> usedAspects;
    private final AspectDescriptor aspectDescriptor;

    AspectDetails(ImmutableList<AspectDetails> usedAspects, AspectDescriptor aspectDescriptor) {
      this.usedAspects = usedAspects;
      this.aspectDescriptor = aspectDescriptor;
    }

    public AspectDescriptor getAspectDescriptor() {
      return aspectDescriptor;
    }

    public ImmutableList<AspectDetails> getUsedAspects() {
      return usedAspects;
    }
  }

  /** SkyKey for building top-level aspects details. */
  public static final class BuildTopLevelAspectsDetailsKey implements SkyKey {
    private final ImmutableList<AspectClass> topLevelAspectsClasses;
    private final int hashCode;

    @AutoCodec.Instantiator
    @AutoCodec.VisibleForSerialization
    static BuildTopLevelAspectsDetailsKey createInternal(
        ImmutableList<AspectClass> topLevelAspectsClasses) {
      return buildTopLevelAspectsDetailsKeyInterner.intern(
          new BuildTopLevelAspectsDetailsKey(
              topLevelAspectsClasses, java.util.Objects.hashCode(topLevelAspectsClasses)));
    }

    private BuildTopLevelAspectsDetailsKey(
        ImmutableList<AspectClass> topLevelAspectsClasses, int hashCode) {
      this.topLevelAspectsClasses = topLevelAspectsClasses;
      this.hashCode = hashCode;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.BUILD_TOP_LEVEL_ASPECTS_DETAILS;
    }

    ImmutableList<AspectClass> getTopLevelAspectsClasses() {
      return topLevelAspectsClasses;
    }

    @Override
    public int hashCode() {
      return hashCode;
    }

    @Override
    public boolean equals(Object o) {
      if (o == this) {
        return true;
      }
      if (!(o instanceof BuildTopLevelAspectsDetailsKey)) {
        return false;
      }
      BuildTopLevelAspectsDetailsKey that = (BuildTopLevelAspectsDetailsKey) o;
      return hashCode == that.hashCode
          && Objects.equal(topLevelAspectsClasses, that.topLevelAspectsClasses);
    }
  }

  /**
   * SkyValue for {@code BuildTopLevelAspectsDetailsKey} wraps a list of the {@code AspectDetails}
   * of the top level aspects.
   */
  public static final class BuildTopLevelAspectsDetailsValue implements SkyValue {
    private final ImmutableList<AspectDetails> aspectsDetails;

    private BuildTopLevelAspectsDetailsValue(Collection<AspectDetails> aspectsDetails) {
      this.aspectsDetails = ImmutableList.copyOf(aspectsDetails);
    }

    public ImmutableList<AspectDetails> getAspectsDetails() {
      return aspectsDetails;
    }
  }
}
