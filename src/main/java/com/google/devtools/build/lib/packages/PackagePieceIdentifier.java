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

package com.google.devtools.build.lib.packages;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.util.HashCodes;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Objects;

/** A unique identifier for a {@link PackagePiece}. */
public sealed interface PackagePieceIdentifier extends SkyKey
    permits PackagePieceIdentifier.ForBuildFile, PackagePieceIdentifier.ForMacro {
  /**
   * The canonical form of the package name if this is an identifier for a {@link
   * PackagePiece.ForBuildFile}, or the canonical form of the macro instance name if this is an
   * identifier for a {@link PackagePiece.ForMacro}.
   *
   * <p>In tha case of a {@link PackagePiece.ForMacro}, the string is not unique, since multiple
   * macro instances can have the same name. Intended to be used in combination with {@link
   * PackagePiece#getCanonicalFormDefinedBy}.
   */
  public abstract String getCanonicalFormName();

  /** Returns the package identifier of the package to which this package piece belong . */
  public PackageIdentifier getPackageIdentifier();

  /**
   * A unique identifier for a {@link PackagePiece.ForBuildFile}.
   *
   * <p>This class does not add any new fields to {@link PackagePieceIdentifier}; it exists as a
   * sibling class of {@link PackagePieceIdentifier.ForMacro} only to reduce the potential for
   * confusion when used as sky keys.
   */
  public static final class ForBuildFile implements PackagePieceIdentifier {
    private final PackageIdentifier packageIdentifier;

    @Override
    public PackageIdentifier getPackageIdentifier() {
      return packageIdentifier;
    }

    @Override
    public String getCanonicalFormName() {
      return packageIdentifier.getCanonicalForm();
    }

    @Override
    public String toString() {
      return String.format("<PackagePieceIdentifier.ForBuildFile pkg=%s>", getCanonicalFormName());
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PACKAGE;
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }
      return other instanceof PackagePieceIdentifier.ForBuildFile that
          && Objects.equals(this.packageIdentifier, that.packageIdentifier);
    }

    @Override
    public int hashCode() {
      return HashCodes.hashObject(packageIdentifier);
    }

    public ForBuildFile(PackageIdentifier packageIdentifier) {
      this.packageIdentifier = checkNotNull(packageIdentifier);
    }
  }

  /** A unique identifier for a {@link PackagePiece.ForMacro}. */
  public static final class ForMacro implements PackagePieceIdentifier {
    private final PackageIdentifier packageIdentifier;
    private final PackagePieceIdentifier parentIdentifier;
    private final String instanceName;

    /** Returns the name attribute of the macro instance. */
    public String getInstanceName() {
      return instanceName;
    }

    @Override
    public PackageIdentifier getPackageIdentifier() {
      return packageIdentifier;
    }

    /** Returns the identifier of the package piece in which this macro instance was defined. */
    public PackagePieceIdentifier getParentIdentifier() {
      return parentIdentifier;
    }

    @Override
    public String getCanonicalFormName() {
      return String.format("%s:%s", packageIdentifier.getCanonicalForm(), getInstanceName());
    }

    @Override
    public String toString() {
      return String.format(
          "<PackagePieceIdentifier.ForMacro name=%s declared_in=%s>",
          getCanonicalFormName(), parentIdentifier);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.EVAL_MACRO;
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }
      return other instanceof PackagePieceIdentifier.ForMacro that
          && Objects.equals(this.packageIdentifier, that.packageIdentifier)
          && Objects.equals(this.instanceName, that.instanceName)
          && Objects.equals(this.parentIdentifier, that.parentIdentifier);
    }

    @Override
    public int hashCode() {
      return HashCodes.hashObjects(packageIdentifier, instanceName, parentIdentifier);
    }

    public ForMacro(
        PackageIdentifier packageIdentifier,
        PackagePieceIdentifier parentIdentifier,
        String instanceName) {
      this.packageIdentifier = checkNotNull(packageIdentifier);
      checkArgument(
          checkNotNull(parentIdentifier).getPackageIdentifier().equals(packageIdentifier));
      this.parentIdentifier = parentIdentifier;
      this.instanceName = checkNotNull(instanceName);
    }
  }
}
