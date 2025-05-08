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

import com.google.common.base.MoreObjects;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.AspectsList;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.StarlarkAspect;
import com.google.devtools.build.lib.packages.StarlarkAspectClass;
import com.google.devtools.build.lib.packages.StarlarkDefinedAspect;
import com.google.devtools.build.lib.server.FailureDetails.Analysis.Code;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/** {@link SkyFunction} to load top level aspects and assign their parameters. */
final class LoadTopLevelAspectsFunction implements SkyFunction {

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws LoadTopLevelAspectsFunctionException, InterruptedException {

    LoadTopLevelAspectsKey topLevelAspectsDetailsKey = (LoadTopLevelAspectsKey) skyKey.argument();

    ImmutableList<Aspect> topLevelAspects =
        getTopLevelAspects(
            env,
            topLevelAspectsDetailsKey.getTopLevelAspectsClasses(),
            topLevelAspectsDetailsKey.getTopLevelAspectsParameters());

    if (topLevelAspects == null) {
      return null; // some aspects are not loaded
    }

    return new LoadTopLevelAspectsValue(topLevelAspects);
  }

  @Nullable
  private static StarlarkDefinedAspect loadStarlarkAspect(
      Environment env, StarlarkAspectClass aspectClass)
      throws InterruptedException, LoadTopLevelAspectsFunctionException {
    StarlarkDefinedAspect starlarkAspect;
    try {
      BzlLoadValue bzlLoadValue =
          (BzlLoadValue)
              env.getValueOrThrow(
                  AspectFunction.bzlLoadKeyForStarlarkAspect(aspectClass),
                  BzlLoadFailedException.class);
      if (bzlLoadValue == null) {
        return null;
      }
      starlarkAspect = AspectFunction.loadAspectFromBzl(aspectClass, bzlLoadValue);
    } catch (BzlLoadFailedException | AspectCreationException e) {
      env.getListener().handle(Event.error(e.getMessage()));
      throw new LoadTopLevelAspectsFunctionException(
          new TopLevelAspectsDetailsBuildFailedException(
              e.getMessage(), Code.ASPECT_CREATION_FAILED));
    }
    return starlarkAspect;
  }

  @Nullable
  private static ImmutableList<Aspect> getTopLevelAspects(
      Environment env,
      ImmutableList<AspectClass> topLevelAspectsClasses,
      ImmutableMap<String, String> topLevelAspectsParameters)
      throws InterruptedException, LoadTopLevelAspectsFunctionException {
    AspectsList.Builder builder = new AspectsList.Builder();

    for (AspectClass aspectClass : topLevelAspectsClasses) {
      if (aspectClass instanceof StarlarkAspectClass starlarkAspectClass) {
        StarlarkAspect starlarkAspect = loadStarlarkAspect(env, starlarkAspectClass);
        if (starlarkAspect == null) {
          return null;
        }
        try {
          builder.addAspect(starlarkAspect);
        } catch (EvalException e) {
          env.getListener().handle(Event.error(e.getInnermostLocation(), e.getMessageWithStack()));
          throw new LoadTopLevelAspectsFunctionException(
              new TopLevelAspectsDetailsBuildFailedException(
                  e.getMessage(), Code.ASPECT_CREATION_FAILED));
        }
      } else {
        try {
          builder.addAspect((NativeAspectClass) aspectClass);
        } catch (AssertionError e) {
          env.getListener().handle(Event.error(e.getMessage()));
          throw new LoadTopLevelAspectsFunctionException(
              new TopLevelAspectsDetailsBuildFailedException(
                  e.getMessage(), Code.ASPECT_CREATION_FAILED));
        }
      }
    }

    AspectsList aspectsList = builder.build();
    try {
        aspectsList.validateTopLevelAspectsParameters(topLevelAspectsParameters);
        return aspectsList.buildAspects(topLevelAspectsParameters);
    } catch (EvalException e) {
      env.getListener().handle(Event.error(e.getInnermostLocation(), e.getMessageWithStack()));
      throw new LoadTopLevelAspectsFunctionException(
          new TopLevelAspectsDetailsBuildFailedException(
              e.getMessage(), Code.ASPECT_CREATION_FAILED));
    }
  }

  private static final class LoadTopLevelAspectsFunctionException extends SkyFunctionException {
    LoadTopLevelAspectsFunctionException(TopLevelAspectsDetailsBuildFailedException cause) {
      super(cause, Transience.PERSISTENT);
    }
  }

  /** {@link SkyKey} for building top-level aspects details. */
  @AutoCodec
  static final class LoadTopLevelAspectsKey implements SkyKey {
    private static final SkyKeyInterner<LoadTopLevelAspectsKey> interner = SkyKey.newInterner();

    private final ImmutableList<AspectClass> topLevelAspectsClasses;
    private final ImmutableMap<String, String> topLevelAspectsParameters;
    private final int hashCode;

    static LoadTopLevelAspectsKey create(
        ImmutableList<AspectClass> topLevelAspectsClasses,
        ImmutableMap<String, String> topLevelAspectsParameters) {
      return interner.intern(
          new LoadTopLevelAspectsKey(
              topLevelAspectsClasses,
              topLevelAspectsParameters,
              Objects.hashCode(topLevelAspectsClasses, topLevelAspectsParameters)));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static LoadTopLevelAspectsKey intern(LoadTopLevelAspectsKey key) {
      return interner.intern(key);
    }

    private LoadTopLevelAspectsKey(
        ImmutableList<AspectClass> topLevelAspectsClasses,
        @Nullable ImmutableMap<String, String> topLevelAspectsParameters,
        int hashCode) {
      Preconditions.checkArgument(!topLevelAspectsClasses.isEmpty(), "No aspects");
      this.topLevelAspectsClasses = topLevelAspectsClasses;
      this.topLevelAspectsParameters = topLevelAspectsParameters;
      this.hashCode = hashCode;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.BUILD_TOP_LEVEL_ASPECTS_DETAILS;
    }

    ImmutableList<AspectClass> getTopLevelAspectsClasses() {
      return topLevelAspectsClasses;
    }

    ImmutableMap<String, String> getTopLevelAspectsParameters() {
      return topLevelAspectsParameters;
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
      if (!(o instanceof LoadTopLevelAspectsKey that)) {
        return false;
      }
      return hashCode == that.hashCode
          && topLevelAspectsClasses.equals(that.topLevelAspectsClasses)
          && topLevelAspectsParameters.equals(that.topLevelAspectsParameters);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("topLevelAspectsClasses", topLevelAspectsClasses)
          .add("topLevelAspectsParameters", topLevelAspectsParameters)
          .toString();
    }

    @Override
    public SkyKeyInterner<LoadTopLevelAspectsKey> getSkyKeyInterner() {
      return interner;
    }
  }

  /**
   * {@link SkyValue} for {@code LoadTopLevelAspectsKey} wraps a list of the {@code Aspect} of the
   * top level aspects.
   */
  static final class LoadTopLevelAspectsValue implements SkyValue {
    private final ImmutableList<Aspect> aspects;

    private LoadTopLevelAspectsValue(Collection<Aspect> aspects) {
      this.aspects = ImmutableList.copyOf(aspects);
    }

    public ImmutableList<Aspect> getAspects() {
      return aspects;
    }
  }
}
