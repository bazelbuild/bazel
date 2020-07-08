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
import com.google.devtools.build.lib.packages.StarlarkAspect;
import com.google.devtools.build.lib.skyframe.AspectValueKey.StarlarkAspectLoadingKey;
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
public class ToplevelStarlarkAspectFunction implements SkyFunction {
  ToplevelStarlarkAspectFunction() {}

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws LoadStarlarkAspectFunctionException, InterruptedException {
    StarlarkAspectLoadingKey aspectLoadingKey = (StarlarkAspectLoadingKey) skyKey.argument();
    String starlarkValueName = aspectLoadingKey.getStarlarkValueName();
    Label starlarkFileLabel = aspectLoadingKey.getStarlarkFileLabel();

    StarlarkAspect starlarkAspect;
    try {
      starlarkAspect = AspectFunction.loadStarlarkAspect(env, starlarkFileLabel, starlarkValueName);
      if (starlarkAspect == null) {
        return null;
      }
      if (!starlarkAspect.getParamAttributes().isEmpty()) {
        String msg = "Cannot instantiate parameterized aspect " + starlarkAspect.getName()
            + " at the top level.";
        throw new AspectCreationException(msg, new LabelCause(starlarkFileLabel, msg));
      }
    } catch (AspectCreationException e) {
      throw new LoadStarlarkAspectFunctionException(e);
    }
    SkyKey aspectKey = aspectLoadingKey.toAspectKey(starlarkAspect.getAspectClass());

    return env.getValue(aspectKey);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /** Exceptions thrown from ToplevelStarlarkAspectFunction. */
  public static class LoadStarlarkAspectFunctionException extends SkyFunctionException {
    public LoadStarlarkAspectFunctionException(AspectCreationException cause) {
      super(cause, Transience.PERSISTENT);
    }
  }
}
