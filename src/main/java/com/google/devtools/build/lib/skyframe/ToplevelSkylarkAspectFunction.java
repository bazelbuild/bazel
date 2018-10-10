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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.causes.LabelCause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.SkylarkAspect;
import com.google.devtools.build.lib.skyframe.AspectFunction.AspectCreationException;
import com.google.devtools.build.lib.skyframe.AspectValue.SkylarkAspectLoadingKey;
import com.google.devtools.build.lib.syntax.SkylarkImport;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * SkyFunction to load aspects from Skylark extensions and calculate their values.
 *
 * Used for loading top-level aspects. At top level, in
 * {@link com.google.devtools.build.lib.analysis.BuildView}, we cannot invoke two SkyFunctions
 * one after another, so BuildView calls this function to do the work.
 */
public class ToplevelSkylarkAspectFunction implements SkyFunction {

  @Nullable private final SkylarkImportLookupFunction skylarkImportLookupFunctionForInlining;

  ToplevelSkylarkAspectFunction(
      @Nullable SkylarkImportLookupFunction skylarkImportLookupFunctionForInlining) {
    this.skylarkImportLookupFunctionForInlining = skylarkImportLookupFunctionForInlining;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws LoadSkylarkAspectFunctionException, InterruptedException {
    SkylarkAspectLoadingKey aspectLoadingKey = (SkylarkAspectLoadingKey) skyKey.argument();
    String skylarkValueName = aspectLoadingKey.getSkylarkValueName();
    SkylarkImport extensionFile = aspectLoadingKey.getSkylarkImport();
    
    // Find label corresponding to skylark file, if one exists.
    ImmutableMap<String, Label> labelLookupMap =
        SkylarkImportLookupFunction.getLabelsForLoadStatements(
            ImmutableList.of(extensionFile),
            Label.parseAbsoluteUnchecked("//:empty"));

    SkylarkAspect skylarkAspect;
    Label extensionFileLabel = Iterables.getOnlyElement(labelLookupMap.values());
    try {
      skylarkAspect =
          AspectFunction.loadSkylarkAspect(
              env, extensionFileLabel, skylarkValueName, skylarkImportLookupFunctionForInlining);
      if (skylarkAspect == null) {
        return null;
      }
      if (!skylarkAspect.getParamAttributes().isEmpty()) {
        String msg = "Cannot instantiate parameterized aspect " + skylarkAspect.getName()
            + " at the top level.";
        throw new AspectCreationException(msg, new LabelCause(extensionFileLabel, msg));
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
