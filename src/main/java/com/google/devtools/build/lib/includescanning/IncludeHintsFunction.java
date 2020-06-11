// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.includescanning;

import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.includescanning.IncludeParser.Hints;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.skyframe.ContainingPackageLookupValue;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * Creates a {@link IncludeParser.HintsRules} object. Done in Skyframe to track dependence on
 * INCLUDE_HINTS file.
 */
public class IncludeHintsFunction implements SkyFunction {
  // TODO(b/111722810): the action cache is not sensitive to changes in the INCLUDE_HINTS file, so
  //  even though Skyframe handles changes, we may still not re-execute an affected action.
  @AutoCodec
  public static final SkyKey INCLUDE_HINTS_KEY =
      (SkyKey) () -> IncludeScanningSkyFunctions.INCLUDE_HINTS;

  private final PathFragment hintsFile;

  public IncludeHintsFunction(PathFragment hintsFile) {
    this.hintsFile = hintsFile;
  }

  @Nullable
  @Override
  public IncludeParser.HintsRules compute(SkyKey skyKey, Environment env)
      throws IncludeHintsFunctionException, InterruptedException {
    Root hintsPackageRoot;
    try {
      ContainingPackageLookupValue hintsLookupValue =
          (ContainingPackageLookupValue) env.getValueOrThrow(ContainingPackageLookupValue.key(
              PackageIdentifier.createInMainRepo(hintsFile.getParentDirectory())),
              IOException.class, BuildFileNotFoundException.class);
      if (env.valuesMissing()) {
        return null;
      }
      if (!hintsLookupValue.hasContainingPackage()) {
        String reasonForNoContainingPackage = hintsLookupValue.getReasonForNoContainingPackage();
        throw new IncludeHintsFunctionException(
            new EnvironmentalExecException(
                "INCLUDE_HINTS file "
                    + hintsFile
                    + " was not in a package"
                    + (reasonForNoContainingPackage != null
                        ? ": " + reasonForNoContainingPackage
                        : "")));
      }
      hintsPackageRoot = hintsLookupValue.getContainingPackageRoot();
      env.getValueOrThrow(FileValue.key(RootedPath.toRootedPath(hintsPackageRoot, hintsFile)),
          IOException.class);
    } catch (IOException | BuildFileNotFoundException e) {
      throw new IncludeHintsFunctionException(new EnvironmentalExecException(
          "could not read INCLUDE_HINTS file", e));
    }
    if (env.valuesMissing()) {
      return null;
    }
    try {
      return Hints.getRules(hintsPackageRoot.getRelative(hintsFile));
    } catch (IOException e) {
      throw new IncludeHintsFunctionException(new EnvironmentalExecException(
          "could not read INCLUDE_HINTS file", e));
    }
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Used to declare the exception type that can be wrapped in the exception thrown by
   * {@link IncludeHintsFunction#compute}.
   */
  private static final class IncludeHintsFunctionException extends SkyFunctionException {
    IncludeHintsFunctionException(EnvironmentalExecException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
