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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.rules.SkylarkRuleClassFunctions.SkylarkAspect;
import com.google.devtools.build.lib.skyframe.AspectValue.SkylarkAspectLoadingKey;
import com.google.devtools.build.lib.skyframe.SkylarkImportLookupFunction.SkylarkImportFailedException;
import com.google.devtools.build.lib.syntax.Type.ConversionException;
import com.google.devtools.build.lib.vfs.PathFragment;
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

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws LoadSkylarkAspectFunctionException, InterruptedException {
    SkylarkAspectLoadingKey aspectLoadingKey = (SkylarkAspectLoadingKey) skyKey.argument();
    String skylarkValueName = aspectLoadingKey.getSkylarkValueName();
    PathFragment extensionFile = aspectLoadingKey.getExtensionFile();
    
    // Find label corresponding to skylark file, if one exists.
    ImmutableMap<PathFragment, Label> labelLookupMap;
    try {
      labelLookupMap =
          SkylarkImportLookupFunction.labelsForAbsoluteImports(ImmutableSet.of(extensionFile), env);
    } catch (SkylarkImportFailedException e) {
      throw new LoadSkylarkAspectFunctionException(e, skyKey);
    }
    if (labelLookupMap == null) {
      return null;
    }

    SkylarkAspect skylarkAspect = null;
    try {
      skylarkAspect = AspectFunction.loadSkylarkAspect(
          env, labelLookupMap.get(extensionFile), skylarkValueName);
    } catch (ConversionException e) {
      throw new LoadSkylarkAspectFunctionException(e, skyKey);
    }
    if (skylarkAspect == null) {
      return null;
    }
    SkyKey aspectKey =
        AspectValue.key(
            aspectLoadingKey.getTargetLabel(),
            aspectLoadingKey.getTargetConfiguration(),
            skylarkAspect.getAspectClass(),
            AspectParameters.EMPTY);

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
    public LoadSkylarkAspectFunctionException(Exception cause, SkyKey childKey) {
      super(cause, childKey);
    }

    public LoadSkylarkAspectFunctionException(Exception cause) {
      super(cause, Transience.PERSISTENT);
    }
  }
}
