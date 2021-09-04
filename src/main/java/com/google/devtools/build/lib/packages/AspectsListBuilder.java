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
import java.util.HashMap;
import java.util.LinkedHashMap;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/**
 * AspectsList represents the list of aspects specified via --aspects command line option or
 * declared in attribute aspects list. The class is responsible for wrapping the information
 * necessary for constructing those aspects including the inherited information for aspects required
 * by other aspects via `requires` attribute.
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
    private static final ImmutableList<String> ALL_ATTR_ASPECTS = ImmutableList.of("*");

    final C aspectClass;
    final Function<Rule, AspectParameters> parametersExtractor;

    String baseAspectName;
    ImmutableList.Builder<ImmutableSet<StarlarkProviderIdentifier>> inheritedRequiredProviders;
    ImmutableList.Builder<String> inheritedAttributeAspects;
    boolean inheritedAllProviders = false;
    boolean inheritedAllAttributes = false;

    private AspectDetails(C aspectClass, Function<Rule, AspectParameters> parametersExtractor) {
      this.aspectClass = aspectClass;
      this.parametersExtractor = parametersExtractor;
      this.inheritedRequiredProviders = ImmutableList.builder();
      this.inheritedAttributeAspects = ImmutableList.builder();
    }

    private AspectDetails(
        C aspectClass,
        Function<Rule, AspectParameters> parametersExtractor,
        String baseAspectName,
        ImmutableList<ImmutableSet<StarlarkProviderIdentifier>> inheritedRequiredProviders,
        ImmutableList<String> inheritedAttributeAspects) {
      this.aspectClass = aspectClass;
      this.parametersExtractor = parametersExtractor;
      this.baseAspectName = baseAspectName;
      this.inheritedRequiredProviders = null;
      this.inheritedAttributeAspects = null;
      if (baseAspectName != null) {
        if (inheritedRequiredProviders == null) {
          // Should only happen during deserialization
          inheritedAllProviders = true;
        } else {
          updateInheritedRequiredProviders(inheritedRequiredProviders);
        }
        if (inheritedAttributeAspects == null) {
          // Should only happen during deserialization
          inheritedAllAttributes = true;
        } else {
          updateInheritedAttributeAspects(inheritedAttributeAspects);
        }
      }
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

    void updateInheritedRequiredProviders(
        ImmutableList<ImmutableSet<StarlarkProviderIdentifier>> requiredProviders) {
      if (!inheritedAllProviders && !requiredProviders.isEmpty()) {
        if (inheritedRequiredProviders == null) {
          inheritedRequiredProviders = ImmutableList.builder();
        }
        inheritedRequiredProviders.addAll(requiredProviders);
      } else {
        inheritedAllProviders = true;
        inheritedRequiredProviders = null;
      }
    }

    void updateInheritedAttributeAspects(ImmutableList<String> attributeAspects) {
      if (!inheritedAllAttributes && !ALL_ATTR_ASPECTS.equals(attributeAspects)) {
        if (inheritedAttributeAspects == null) {
          inheritedAttributeAspects = ImmutableList.builder();
        }
        inheritedAttributeAspects.addAll(attributeAspects);
      } else {
        inheritedAllAttributes = true;
        inheritedAttributeAspects = null;
      }
    }

    RequiredProviders buildInheritedRequiredProviders() {
      if (baseAspectName == null) {
        return RequiredProviders.acceptNoneBuilder().build();
      } else if (inheritedAllProviders) {
        return RequiredProviders.acceptAnyBuilder().build();
      } else {
        ImmutableList<ImmutableSet<StarlarkProviderIdentifier>> inheritedRequiredProvidersList =
            inheritedRequiredProviders.build();
        RequiredProviders.Builder inheritedRequiredProvidersBuilder =
            RequiredProviders.acceptAnyBuilder();
        for (ImmutableSet<StarlarkProviderIdentifier> providerSet :
            inheritedRequiredProvidersList) {
          if (!providerSet.isEmpty()) {
            inheritedRequiredProvidersBuilder.addStarlarkSet(providerSet);
          }
        }
        return inheritedRequiredProvidersBuilder.build();
      }
    }

    @Nullable
    ImmutableSet<String> buildInheritedAttributeAspects() {
      if (baseAspectName == null) {
        return ImmutableSet.of();
      } else if (inheritedAllAttributes) {
        return null;
      } else {
        return ImmutableSet.copyOf(inheritedAttributeAspects.build());
      }
    }

    @VisibleForSerialization
    public ImmutableList<ImmutableSet<StarlarkProviderIdentifier>>
        getInheritedRequiredProvidersList() {
      return inheritedRequiredProviders == null ? null : inheritedRequiredProviders.build();
    }

    @VisibleForSerialization
    public ImmutableList<String> getInheritedAttributeAspectsList() {
      return inheritedAttributeAspects == null ? null : inheritedAttributeAspects.build();
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
        String baseAspectName,
        ImmutableList<ImmutableSet<StarlarkProviderIdentifier>> inheritedRequiredProvidersList,
        ImmutableList<String> inheritedAttributeAspectsList) {
      super(
          aspectClass,
          parametersExtractor,
          baseAspectName,
          inheritedRequiredProvidersList,
          inheritedAttributeAspectsList);
    }

    @Override
    public Aspect getAspect(Rule rule) {
      AspectParameters params;
      if (rule == null) {
        params = AspectParameters.EMPTY;
      } else {
        params = parametersExtractor.apply(rule);
      }
      return params == null
          ? null
          : Aspect.forNative(
              aspectClass,
              params,
              buildInheritedRequiredProviders(),
              buildInheritedAttributeAspects());
    }
  }

  @VisibleForSerialization
  @AutoCodec
  static class StarlarkAspectDetails extends AspectDetails<StarlarkAspectClass> {
    private final StarlarkDefinedAspect aspect;

    @VisibleForSerialization
    StarlarkAspectDetails(
        StarlarkDefinedAspect aspect,
        String baseAspectName,
        ImmutableList<ImmutableSet<StarlarkProviderIdentifier>> inheritedRequiredProvidersList,
        ImmutableList<String> inheritedAttributeAspectsList) {
      super(
          aspect.getAspectClass(),
          aspect.getDefaultParametersExtractor(),
          baseAspectName,
          inheritedRequiredProvidersList,
          inheritedAttributeAspectsList);
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
      return Aspect.forStarlark(
          aspectClass,
          aspect.getDefinition(params),
          params,
          buildInheritedRequiredProviders(),
          buildInheritedAttributeAspects());
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

  @AutoCodec @AutoCodec.VisibleForSerialization
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
   * required it), its inherited required providers and its inherited propagation atttributes if
   * any.
   *
   * @param starlarkAspect the starlark defined aspect to be added
   * @param baseAspectName is the name of the base aspect requiring this aspect, can be {@code null}
   *     if the aspect is directly listed in the aspects list
   * @param inheritedRequiredProviders is the list of required providers inherited from the aspect
   *     parent aspects
   * @param inheritedAttributeAspects is the list of attribute aspects inherited from the aspect
   *     parent aspects
   */
  public void addAspect(
      StarlarkDefinedAspect starlarkAspect,
      @Nullable String baseAspectName,
      ImmutableList<ImmutableSet<StarlarkProviderIdentifier>> inheritedRequiredProviders,
      ImmutableList<String> inheritedAttributeAspects)
      throws EvalException {
    boolean needsToAdd =
        checkAndUpdateExistingAspects(
            starlarkAspect.getName(),
            baseAspectName,
            inheritedRequiredProviders,
            inheritedAttributeAspects);
    if (needsToAdd) {
      StarlarkAspectDetails starlarkAspectDetails =
          new StarlarkAspectDetails(
              starlarkAspect,
              baseAspectName,
              inheritedRequiredProviders,
              inheritedAttributeAspects);
      this.aspects.put(starlarkAspect.getName(), starlarkAspectDetails);
    }
  }

  /**
   * Adds a native aspect to the aspects list with its base aspect (the aspect that required it),
   * its inherited required providers and its inherited propagation atttributes if any.
   *
   * @param nativeAspect the native aspect to be added
   * @param baseAspectName is the name of the base aspect requiring this aspect, can be {@code null}
   *     if the aspect is directly listed in the aspects list
   * @param inheritedRequiredProviders is the list of required providers inherited from the aspect
   *     parent aspects
   * @param inheritedAttributeAspects is the list of attribute aspects inherited from the aspect
   *     parent aspects
   */
  public void addAspect(
      StarlarkNativeAspect nativeAspect,
      @Nullable String baseAspectName,
      ImmutableList<ImmutableSet<StarlarkProviderIdentifier>> inheritedRequiredProviders,
      ImmutableList<String> inheritedAttributeAspects)
      throws EvalException {
    boolean needsToAdd =
        checkAndUpdateExistingAspects(
            nativeAspect.getName(),
            baseAspectName,
            inheritedRequiredProviders,
            inheritedAttributeAspects);
    if (needsToAdd) {
      NativeAspectDetails nativeAspectDetails =
          new NativeAspectDetails(
              nativeAspect,
              nativeAspect.getDefaultParametersExtractor(),
              baseAspectName,
              inheritedRequiredProviders,
              inheritedAttributeAspects);
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

  private boolean checkAndUpdateExistingAspects(
      String aspectName,
      String baseAspectName,
      ImmutableList<ImmutableSet<StarlarkProviderIdentifier>> inheritedRequiredProviders,
      ImmutableList<String> inheritedAttributeAspects)
      throws EvalException {

    AspectDetails<?> oldAspect = this.aspects.get(aspectName);

    if (oldAspect != null) {
      // If the aspect to be added is required by another aspect, i.e. {@code baseAspectName} is
      // not null, then we need to update its inherited required providers and propgation
      // attributes.
      if (baseAspectName != null) {
        oldAspect.baseAspectName = baseAspectName;
        oldAspect.updateInheritedRequiredProviders(inheritedRequiredProviders);
        oldAspect.updateInheritedAttributeAspects(inheritedAttributeAspects);
        return false; // no need to add the new aspect
      } else {
        // If the aspect to be added is not required by another aspect, then we
        // should throw an error
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
