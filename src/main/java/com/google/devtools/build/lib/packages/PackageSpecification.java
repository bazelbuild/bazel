// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Verify;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * A class that represents some packages that are included in the visibility list of a rule.
 */
public abstract class PackageSpecification {
  private static final String PACKAGE_LABEL = "__pkg__";
  private static final String SUBTREE_LABEL = "__subpackages__";
  private static final String ALL_BENEATH_SUFFIX = "/...";
  public static final PackageSpecification EVERYTHING = new PackageSpecification() {
    @Override
    public boolean containsPackage(PackageIdentifier packageName) {
      return true;
    }
  };

  public abstract boolean containsPackage(PackageIdentifier packageName);

  @Override
  public int hashCode() {
    return toString().hashCode();
  }

  @Override
  public boolean equals(Object that) {
    if (this == that) {
      return true;
    }

    if (!(that instanceof PackageSpecification)) {
      return false;
    }

    return this.toString().equals(that.toString());
  }

  /**
   * Parses a string as a visibility specification.
   * Throws {@link InvalidPackageSpecificationException} if the label cannot be parsed.
   *
   * <p>Note that these strings are different from what {@link #fromLabel} understands.
   */
  public static PackageSpecification fromString(Label context, final String spec)
      throws InvalidPackageSpecificationException {
    String result = spec;
    boolean allBeneath = false;
    if (result.endsWith(ALL_BENEATH_SUFFIX)) {
      allBeneath = true;
      result = result.substring(0, result.length() - ALL_BENEATH_SUFFIX.length());
      if (result.equals("/")) {
        // Special case: //... will not end in /...
        return EVERYTHING;
      }
    }

    if (!spec.startsWith("//")) {
      throw new InvalidPackageSpecificationException("invalid package name '" + spec
          + "': must start with '//'");
    }

    PackageIdentifier packageId;
    try {
      packageId = PackageIdentifier.parse(result);
    } catch (LabelSyntaxException e) {
      throw new InvalidPackageSpecificationException(
          "invalid package name '" + spec + "': " + e.getMessage());
    }

    Verify.verify(packageId.getRepository().isDefault());
    packageId = PackageIdentifier.create(
        context.getPackageIdentifier().getRepository(), packageId.getPackageFragment());

    return allBeneath ?
        new AllPackagesBeneath(packageId) :
        new SinglePackage(packageId);
  }

  /**
   * Parses a label as a visibility specification. returns null if the label cannot be parsed.
   *
   * <p>Note that these strings are different from what {@link #fromString} understands.
   */
  public static PackageSpecification fromLabel(Label label) {
    if (label.getName().equals(PACKAGE_LABEL)) {
      return new SinglePackage(label.getPackageIdentifier());
    } else if (label.getName().equals(SUBTREE_LABEL)) {
      return new AllPackagesBeneath(label.getPackageIdentifier());
    } else {
      return null;
    }
  }

  private static class SinglePackage extends PackageSpecification {
    private PackageIdentifier singlePackageName;

    public SinglePackage(PackageIdentifier packageName) {
      this.singlePackageName = packageName;
    }

    @Override
    public boolean containsPackage(PackageIdentifier packageName) {
      return this.singlePackageName.equals(packageName);
    }

    @Override
    public String toString() {
      return singlePackageName.toString();
    }
  }

  private static class AllPackagesBeneath extends PackageSpecification {
    private PackageIdentifier prefix;

    public AllPackagesBeneath(PackageIdentifier prefix) {
      this.prefix = prefix;
    }

    @Override
    public boolean containsPackage(PackageIdentifier packageName) {
      return packageName.getRepository().equals(prefix.getRepository())
          && packageName.getPackageFragment().startsWith(prefix.getPackageFragment());
    }

    @Override
    public String toString() {
      return prefix.equals(new PathFragment("")) ? "..." : prefix + "/...";
    }
  }

  /**
   * Exception class to be thrown when a specification cannot be parsed.
   */
  public static class InvalidPackageSpecificationException extends Exception {
    public InvalidPackageSpecificationException(String message) {
      super(message);
    }
  }
}
