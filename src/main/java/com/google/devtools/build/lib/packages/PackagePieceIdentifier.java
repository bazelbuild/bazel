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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.util.HashCodes;

/** A unique identifier for a {@link PackagePiece}. */
public abstract sealed class PackagePieceIdentifier
    permits PackagePieceIdentifier.ForBuildFile, PackagePieceIdentifier.ForMacro {
  protected final PackageIdentifier packageIdentifier;
  // BUILD file label for a {@link PackagePiece.ForBuildFile}, or the label of the macro class
  // definition for a {@link PackagePiece.ForMacro}.
  protected final Label definingLabel;

  /**
   * The canonical form of the package name if this is an identifier for a {@link
   * PackagePiece.ForBuildFile}, or the canonical form of the macro instance name if this is an
   * identifier for a {@link PackagePiece.ForMacro}.
   *
   * <p>In tha case of a {@link PackagePiece.ForMacro}, the string is not unique, since multiple
   * macro instances can have the same name. Intended to be used in combination with {@link
   * #getCanonicalFormDefinedBy}.
   */
  public abstract String getCanonicalFormName();

  public abstract String getCanonicalFormDefinedBy();

  /**
   * BUILD file label for a {@link PackagePiece.ForBuildFile}, or the label of the macro class
   * definition for a {@link PackagePiece.ForMacro}.
   */
  public Label getDefiningLabel() {
    return definingLabel;
  }

  public PackageIdentifier getPackageIdentifier() {
    return packageIdentifier;
  }

  @Override
  public String toString() {
    return String.format("%s defined by %s", getCanonicalFormName(), getCanonicalFormDefinedBy());
  }

  protected PackagePieceIdentifier(PackageIdentifier packageIdentifier, Label definingLabel) {
    this.packageIdentifier = checkNotNull(packageIdentifier);
    this.definingLabel = checkNotNull(definingLabel);
  }

  /**
   * A unique identifier for a {@link PackagePiece.ForBuildFile}.
   *
   * <p>This class does not add any new fields to {@link PackagePieceIdentifier}; it exists as a
   * sibling class of {@link PackagePieceIdentifier.ForMacro} only to reduce the potential for
   * confusion when used as sky keys.
   */
  public static final class ForBuildFile extends PackagePieceIdentifier {
    @Override
    public String getCanonicalFormName() {
      return packageIdentifier.getCanonicalForm();
    }

    @Override
    public String getCanonicalFormDefinedBy() {
      return definingLabel.getCanonicalForm();
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }
      if (!(other instanceof PackagePieceIdentifier.ForBuildFile that)) {
        return false;
      }
      return this.packageIdentifier.equals(that.packageIdentifier)
          && this.definingLabel.equals(that.definingLabel);
    }

    @Override
    public int hashCode() {
      return HashCodes.hashObjects(packageIdentifier, definingLabel);
    }

    public ForBuildFile(PackageIdentifier packageIdentifier, Label definingLabel) {
      super(packageIdentifier, definingLabel);
    }
  }

  /** A unique identifier for a {@link PackagePiece.ForMacro}. */
  public static final class ForMacro extends PackagePieceIdentifier {
    private final String definingSymbol;
    private final String instanceName;

    /** Returns the name of the macro class symbol. */
    public String getDefiningSymbol() {
      return definingSymbol;
    }

    /** Returns the name attribute of the macro instance. */
    public String getInstanceName() {
      return instanceName;
    }

    @Override
    public String getCanonicalFormName() {
      return String.format("%s:%s", packageIdentifier.getCanonicalForm(), getInstanceName());
    }

    @Override
    public String getCanonicalFormDefinedBy() {
      return String.format("%s%%%s", definingLabel.getCanonicalForm(), getDefiningSymbol());
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }
      if (!(other instanceof PackagePieceIdentifier.ForMacro that)) {
        return false;
      }
      return this.packageIdentifier.equals(that.packageIdentifier)
          && this.definingLabel.equals(that.definingLabel)
          // defining symbol and instance name are non-null
          && this.definingSymbol.equals(that.definingSymbol)
          && this.instanceName.equals(that.instanceName);
    }

    @Override
    public int hashCode() {
      return HashCodes.hashObjects(packageIdentifier, definingLabel, definingSymbol, instanceName);
    }

    public ForMacro(
        PackageIdentifier packageIdentifier,
        Label definingLabel,
        String definingSymbol,
        String instanceName) {
      super(packageIdentifier, definingLabel);
      this.definingSymbol = checkNotNull(definingSymbol);
      this.instanceName = checkNotNull(instanceName);
    }
  }
}
