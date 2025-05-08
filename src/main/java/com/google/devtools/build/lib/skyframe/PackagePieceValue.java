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

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.PackagePiece;
import com.google.devtools.build.lib.packages.Packageoid;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/**
 * A Skyframe value representing a package piece.
 *
 * <p>The corresponding {@link com.google.devtools.build.skyframe.SkyKey} is {@link
 * com.google.devtools.build.lib.packages.PackagePieceIdentifier}. Note that different subclasses of
 * PackagePieceIdentifier are evaluated by different SkyFunctions.
 */
public interface PackagePieceValue extends PackageoidValue {
  /**
   * Returns the package piece. This package piece may contain errors, in which case the caller
   * should throw an appropriate subclass of {@link
   * com.google.devtools.build.lib.packages.NoSuchPackagePieceException} if an error-free package
   * piece is needed.
   */
  PackagePiece getPackagePiece();

  @Override
  public default Packageoid getPackageoid() {
    return getPackagePiece();
  }

  /**
   * A Skyframe value representing a package piece obtained by evaluating a BUILD file without
   * expanding any symbolic macros.
   *
   * <p>The corresponding {@link com.google.devtools.build.skyframe.SkyKey} is {@link
   * com.google.devtools.build.lib.packages.PackagePieceIdentifier.ForBuildFile}.
   */
  @AutoCodec
  @Immutable
  @ThreadSafe
  public static final class ForBuildFile implements PackagePieceValue {
    private final PackagePiece.ForBuildFile forBuildFile;

    @Override
    public PackagePiece.ForBuildFile getPackagePiece() {
      return forBuildFile;
    }

    @Override
    public String toString() {
      return String.format(
          "<PackagePieceValue.ForBuildFile name=%s>",
          forBuildFile.getIdentifier().getCanonicalFormName());
    }

    ForBuildFile(PackagePiece.ForBuildFile forBuildFile) {
      this.forBuildFile = forBuildFile;
    }
  }

  /** A Skyframe value representing a package piece obtained by evaluating one symbolic macro. */
  @AutoCodec
  @Immutable
  @ThreadSafe
  public static final class ForMacro implements PackagePieceValue {
    private final PackagePiece.ForMacro forMacro;

    @Override
    public PackagePiece.ForMacro getPackagePiece() {
      return forMacro;
    }

    @Override
    public String toString() {
      return String.format(
          "<PackagePieceValue.ForMacro name=%s defined_by=%s>",
          forMacro.getIdentifier().getCanonicalFormName(),
          forMacro.getIdentifier().getCanonicalFormDefinedBy());
    }

    ForMacro(PackagePiece.ForMacro forMacro) {
      this.forMacro = forMacro;
    }
  }
}
