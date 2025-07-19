// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.packages.MacroInstance;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchPackagePieceException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.PackagePiece;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * A SkyFunction that looks up a {@link com.google.devtools.build.lib.packages.MacroInstance} in a
 * {@link com.google.devtools.build.lib.packages.PackagePiece}, producing a {@link
 * MacroInstanceValue}.
 */
final class MacroInstanceFunction implements SkyFunction {
  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws MacroInstanceFunctionException, InterruptedException {
    MacroInstanceValue.Key key = (MacroInstanceValue.Key) skyKey.argument();
    @Nullable PackagePieceValue packagePieceValue;
    try {
      packagePieceValue =
          (PackagePieceValue)
              env.getValueOrThrow(
                  key.packagePieceId(),
                  NoSuchPackageException.class,
                  NoSuchPackagePieceException.class);
    } catch (NoSuchPackageException e) {
      throw new MacroInstanceFunctionException(e);
    } catch (NoSuchPackagePieceException e) {
      throw new MacroInstanceFunctionException(e);
    }
    if (packagePieceValue == null) {
      return null;
    }

    PackagePiece packagePiece = packagePieceValue.getPackagePiece();
    @Nullable MacroInstance macroInstance = packagePiece.getMacroByName(key.macroInstanceName());
    if (macroInstance == null) {
      throw new MacroInstanceFunctionException(new NoSuchMacroInstanceException(key, packagePiece));
    }
    return new MacroInstanceValue(macroInstance);
  }

  public static final class MacroInstanceFunctionException extends SkyFunctionException {
    MacroInstanceFunctionException(NoSuchPackageException cause) {
      super(cause, Transience.PERSISTENT);
    }

    MacroInstanceFunctionException(NoSuchPackagePieceException cause) {
      super(cause, Transience.PERSISTENT);
    }

    MacroInstanceFunctionException(NoSuchMacroInstanceException cause) {
      super(cause, Transience.PERSISTENT);
    }
  }

  public static final class NoSuchMacroInstanceException extends NoSuchThingException {
    NoSuchMacroInstanceException(MacroInstanceValue.Key key, PackagePiece packagePiece) {
      super(
          String.format(
              "Macro instance '%s' not found in %s",
              key.macroInstanceName(), packagePiece.getShortDescription()));
    }
  }
}
