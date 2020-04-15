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

import com.google.devtools.build.lib.causes.LabelCause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.SkylarkAspect;
import com.google.devtools.build.lib.skyframe.AspectValue.SkylarkAspectLoadingKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * SkyFunction to load aspects from Starlark extensions and calculate their values.
 *
 * <p>Used for loading top-level aspects. At top level, in {@link
 * com.google.devtools.build.lib.analysis.BuildView}, we cannot invoke two SkyFunctions one after
 * another, so BuildView calls this function to do the work.
 */
public class ToplevelSkylarkAspectFunction implements SkyFunction {
  ToplevelSkylarkAspectFunction() {}

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws LoadSkylarkAspectFunctionException, InterruptedException {
    SkylarkAspectLoadingKey aspectLoadingKey = (SkylarkAspectLoadingKey) skyKey.argument();
    String skylarkValueName = aspectLoadingKey.getSkylarkValueName();
    Label skylarkFileLabel = aspectLoadingKey.getSkylarkFileLabel();

    SkylarkAspect skylarkAspect;
    try {
      skylarkAspect = AspectFunction.loadSkylarkAspect(env, skylarkFileLabel, skylarkValueName);
      if (skylarkAspect == null) {
        return null;
      }
      if (!skylarkAspect.getParamAttributes().isEmpty()) {
        String msg = "Cannot instantiate parameterized aspect " + skylarkAspect.getName()
            + " at the top level.";
        throw new AspectCreationException(msg, new LabelCause(skylarkFileLabel, msg));
      }
    } catch (AspectCreationException e) {
      throw new LoadSkylarkAspectFunctionException(e);
    }
    SkyKey aspectKey = aspectLoadingKey.toAspectKey(skylarkAspect.getAspectClass());

    return env.getValue(aspectKey);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Exceptions thrown from ToplevelSkylarkAspectFunction.
   */
  public class LoadSkylarkAspectFunctionException extends SkyFunctionException {
    public LoadSkylarkAspectFunctionException(AspectCreationException cause) {
      super(cause, Transience.PERSISTENT);
    }
  }
}
