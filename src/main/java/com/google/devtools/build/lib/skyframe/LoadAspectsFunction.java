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
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/** {@link SkyFunction} to load top level aspects and assign their parameters. */
final class LoadAspectsFunction implements SkyFunction {

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws LoadAspectsFunctionException, InterruptedException {

    LoadAspectsKey topLevelAspectsDetailsKey = (LoadAspectsKey) skyKey.argument();

    ImmutableList<Aspect> topLevelAspects =
        getTopLevelAspects(
            env,
            topLevelAspectsDetailsKey.getTopLevelAspectsClasses(),
            topLevelAspectsDetailsKey.getTopLevelAspectsParameters());

    if (topLevelAspects == null) {
      return null; // some aspects are not loaded
    }

    return new LoadAspectsValue(topLevelAspects);
  }

  @Nullable
  private static StarlarkDefinedAspect loadStarlarkAspect(
      Environment env, StarlarkAspectClass aspectClass)
      throws InterruptedException, LoadAspectsFunctionException {
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
      throw new LoadAspectsFunctionException(
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
      throws InterruptedException, LoadAspectsFunctionException {
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
          throw new LoadAspectsFunctionException(
              new TopLevelAspectsDetailsBuildFailedException(
                  e.getMessage(), Code.ASPECT_CREATION_FAILED));
        }
      } else {
        try {
          builder.addAspect((NativeAspectClass) aspectClass);
        } catch (AssertionError e) {
          env.getListener().handle(Event.error(e.getMessage()));
          throw new LoadAspectsFunctionException(
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
      throw new LoadAspectsFunctionException(
          new TopLevelAspectsDetailsBuildFailedException(
              e.getMessage(), Code.ASPECT_CREATION_FAILED));
    }
  }

  private static final class LoadAspectsFunctionException extends SkyFunctionException {
    LoadAspectsFunctionException(TopLevelAspectsDetailsBuildFailedException cause) {
      super(cause, Transience.PERSISTENT);
    }
  }
}
