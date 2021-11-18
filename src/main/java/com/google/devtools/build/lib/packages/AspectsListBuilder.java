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

package com.google.devtools.build.lib.packages;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import java.util.HashMap;
import java.util.LinkedHashMap;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/**
 * AspectsList represents the list of aspects specified via --aspects command line option or
 * declared in attribute aspects list. The class is responsible for wrapping the information
 * necessary for constructing those aspects.
 */
public final class AspectsListBuilder {

  private final HashMap<String, AspectDetails<?>> aspects = new LinkedHashMap<>();

  public AspectsListBuilder() {}

  public AspectsListBuilder(ImmutableList<AspectDetails<?>> aspectsList) {
    for (AspectDetails<?> aspect : aspectsList) {
      aspects.put(aspect.getName(), aspect);
    }
  }

  /** Returns a list of the collected aspects details. */
  public ImmutableList<AspectDetails<?>> getAspectsDetails() {
    return ImmutableList.copyOf(aspects.values());
  }

  /**
   * Returns a list of Aspect objects for top level aspects.
   *
   * <p>Since top level aspects do not have parameters, a rule is not required to create their
   * Aspect objects.
   */
  public ImmutableList<Aspect> buildAspects() {
    ImmutableList.Builder<Aspect> aspectsList = ImmutableList.builder();
    for (AspectDetails<?> aspect : aspects.values()) {
      aspectsList.add(aspect.getAspect(null));
    }
    return aspectsList.build();
  }

  /** Wraps the information necessary to construct an Aspect. */
  @VisibleForSerialization
  abstract static class AspectDetails<C extends AspectClass> {
    final C aspectClass;
    final Function<Rule, AspectParameters> parametersExtractor;
    final String baseAspectName;

    private AspectDetails(C aspectClass, Function<Rule, AspectParameters> parametersExtractor) {
      this.aspectClass = aspectClass;
      this.parametersExtractor = parametersExtractor;
      this.baseAspectName = null;
    }

    private AspectDetails(
        C aspectClass,
        Function<Rule, AspectParameters> parametersExtractor,
        String baseAspectName) {
      this.aspectClass = aspectClass;
      this.parametersExtractor = parametersExtractor;
      this.baseAspectName = baseAspectName;
    }

    String getName() {
      return this.aspectClass.getName();
    }

    ImmutableSet<String> getRequiredParameters() {
      return ImmutableSet.of();
    }

    protected abstract Aspect getAspect(@Nullable Rule rule);

    C getAspectClass() {
      return aspectClass;
    }
  }

  private static class NativeAspectDetails extends AspectDetails<NativeAspectClass> {
    NativeAspectDetails(
        NativeAspectClass aspectClass, Function<Rule, AspectParameters> parametersExtractor) {
      super(aspectClass, parametersExtractor);
    }

    NativeAspectDetails(
        NativeAspectClass aspectClass,
        Function<Rule, AspectParameters> parametersExtractor,
        String baseAspectName) {
      super(aspectClass, parametersExtractor, baseAspectName);
    }

    @Override
    public Aspect getAspect(Rule rule) {
      AspectParameters params;
      if (rule == null) {
        params = AspectParameters.EMPTY;
      } else {
        params = parametersExtractor.apply(rule);
      }
      return params == null ? null : Aspect.forNative(aspectClass, params);
    }
  }

  @VisibleForSerialization
  @AutoCodec
  static class StarlarkAspectDetails extends AspectDetails<StarlarkAspectClass> {
    private final StarlarkDefinedAspect aspect;

    @VisibleForSerialization
    StarlarkAspectDetails(StarlarkDefinedAspect aspect, String baseAspectName) {
      super(aspect.getAspectClass(), aspect.getDefaultParametersExtractor(), baseAspectName);
      this.aspect = aspect;
    }

    @Override
    public ImmutableSet<String> getRequiredParameters() {
      return aspect.getParamAttributes();
    }

    @Override
    public Aspect getAspect(Rule rule) {
      AspectParameters params;
      if (rule == null) {
        params = AspectParameters.EMPTY;
      } else {
        params = parametersExtractor.apply(rule);
      }
      return Aspect.forStarlark(aspectClass, aspect.getDefinition(params), params);
    }
  }

  /** Aspect details that just wrap a pre-existing Aspect that doesn't vary with the Rule. */
  private static class PredefinedAspectDetails extends AspectDetails<AspectClass> {
    private final Aspect aspect;

    PredefinedAspectDetails(Aspect aspect) {
      super(aspect.getAspectClass(), null);
      this.aspect = aspect;
    }

    @Override
    public Aspect getAspect(Rule rule) {
      return aspect;
    }
  }

  @SerializationConstant @AutoCodec.VisibleForSerialization
  static final Function<Rule, AspectParameters> EMPTY_FUNCTION = input -> AspectParameters.EMPTY;

  /**
   * Adds a native aspect with its parameters extraction function to the aspects list.
   *
   * @param aspect the native aspect to be added
   * @param evaluator function that extracts aspect parameters from rule.
   */
  public void addAspect(NativeAspectClass aspect, Function<Rule, AspectParameters> evaluator) {
    NativeAspectDetails nativeAspectDetails = new NativeAspectDetails(aspect, evaluator);
    AspectDetails<?> oldAspect =
        this.aspects.put(nativeAspectDetails.getName(), nativeAspectDetails);
    if (oldAspect != null) {
      throw new AssertionError(
          String.format("Aspect %s has already been added", oldAspect.getName()));
    }
  }

  /**
   * Adds a native aspect that does not need a parameters extractor to the aspects list.
   *
   * @param aspect the native aspect to be added
   */
  public void addAspect(NativeAspectClass aspect) {
    addAspect(aspect, EMPTY_FUNCTION);
  }

  /**
   * Adds a starlark defined aspect to the aspects list with its base aspect (the aspect that
   * required it).
   *
   * @param starlarkAspect the starlark defined aspect to be added
   * @param baseAspectName is the name of the base aspect requiring this aspect, can be {@code null}
   *     if the aspect is directly listed in the aspects list
   */
  public void addAspect(StarlarkDefinedAspect starlarkAspect, @Nullable String baseAspectName)
      throws EvalException {
    boolean needsToAdd = checkAndUpdateExistingAspects(starlarkAspect.getName(), baseAspectName);
    if (needsToAdd) {
      StarlarkAspectDetails starlarkAspectDetails =
          new StarlarkAspectDetails(starlarkAspect, baseAspectName);
      this.aspects.put(starlarkAspect.getName(), starlarkAspectDetails);
    }
  }

  /**
   * Adds a native aspect to the aspects list with its base aspect (the aspect that required it).
   *
   * @param nativeAspect the native aspect to be added
   * @param baseAspectName is the name of the base aspect requiring this aspect, can be {@code null}
   *     if the aspect is directly listed in the aspects list
   */
  public void addAspect(StarlarkNativeAspect nativeAspect, @Nullable String baseAspectName)
      throws EvalException {
    boolean needsToAdd = checkAndUpdateExistingAspects(nativeAspect.getName(), baseAspectName);
    if (needsToAdd) {
      NativeAspectDetails nativeAspectDetails =
          new NativeAspectDetails(
              nativeAspect, nativeAspect.getDefaultParametersExtractor(), baseAspectName);
      this.aspects.put(nativeAspect.getName(), nativeAspectDetails);
    }
  }

  /** Should only be used for deserialization. */
  public void addAspect(final Aspect aspect) {
    PredefinedAspectDetails predefinedAspectDetails = new PredefinedAspectDetails(aspect);
    AspectDetails<?> oldAspect =
        this.aspects.put(predefinedAspectDetails.getName(), predefinedAspectDetails);
    if (oldAspect != null) {
      throw new AssertionError(
          String.format("Aspect %s has already been added", oldAspect.getName()));
    }
  }

  private boolean checkAndUpdateExistingAspects(String aspectName, @Nullable String baseAspectName)
      throws EvalException {

    AspectDetails<?> oldAspect = this.aspects.get(aspectName);

    if (oldAspect != null) {
      if (baseAspectName != null) {
        // If the aspect to be added already exists and it is required by another aspect, no need to
        // add it again.
        return false;
      } else {
        // If the aspect to be added is not required by another aspect, then we should throw error
        String oldAspectBaseAspectName = oldAspect.baseAspectName;
        if (oldAspectBaseAspectName != null) {
          throw Starlark.errorf(
              "aspect %s was added before as a required aspect of aspect %s",
              oldAspect.getName(), oldAspectBaseAspectName);
        }
        throw Starlark.errorf("aspect %s added more than once", oldAspect.getName());
      }
    }

    return true; // we need to add the new aspect
  }
}
