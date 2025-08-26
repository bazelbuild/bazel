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

import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchPackagePieceException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackagePieceIdentifier;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * A SkyFunction that looks up a {@link Package.Declarations} in a {@link
 * com.google.devtools.build.lib.packages.PackagePiece.ForBuildFile}, producing a {@link
 * PackageDeclarationsValue}.
 */
final class PackageDeclarationsFunction implements SkyFunction {
  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws PackageDeclarationsFunctionException, InterruptedException {
    PackageDeclarationsValue.Key key = (PackageDeclarationsValue.Key) skyKey.argument();
    @Nullable PackagePieceValue.ForBuildFile packagePieceValue;
    try {
      packagePieceValue =
          (PackagePieceValue.ForBuildFile)
              env.getValueOrThrow(
                  new PackagePieceIdentifier.ForBuildFile(key.packageId()),
                  NoSuchPackageException.class,
                  NoSuchPackagePieceException.class);
    } catch (NoSuchPackageException e) {
      throw new PackageDeclarationsFunctionException(e);
    } catch (NoSuchPackagePieceException e) {
      throw new PackageDeclarationsFunctionException(e);
    }
    if (packagePieceValue == null) {
      return null;
    }

    return new PackageDeclarationsValue(
        packagePieceValue.getPackagePiece().getMetadata(),
        packagePieceValue.getPackagePiece().getDeclarations(),
        packagePieceValue.starlarkSemantics(),
        packagePieceValue.mainRepositoryMapping());
  }

  public static final class PackageDeclarationsFunctionException extends SkyFunctionException {
    PackageDeclarationsFunctionException(NoSuchPackageException cause) {
      super(cause, Transience.PERSISTENT);
    }

    PackageDeclarationsFunctionException(NoSuchPackagePieceException cause) {
      super(cause, Transience.PERSISTENT);
    }
  }
}
