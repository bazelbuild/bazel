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
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Represents one of the following:
 *
 * <ul>
 * <li>A single package (e.g. "//foo/bar")
 * <li>All transitive subpackages of a package, inclusive (e.g. "//foo/bar/...", which includes
 *     "//foo/bar")
 * <li>All packages (i.e. "//...")
 * </ul>
 *
 * <p>Typically (exclusively?) used for package visibility, as part of a {@link PackageGroup}
 * target.
 *
 * <p>A package specification is specific to a single {@link RepositoryName} unless it is the "all
 * packages" specification.
 */
public abstract class PackageSpecification {
  private static final String PACKAGE_LABEL = "__pkg__";
  private static final String SUBTREE_LABEL = "__subpackages__";
  private static final String ALL_BENEATH_SUFFIX = "/...";

  /** Returns {@code true} if the package spec includes the provided {@code packageName}. */
  public abstract boolean containsPackage(PackageIdentifier packageName);

  /**
   * Returns a {@link String} representation of the {@link PackageSpecification} of the same format
   * accepted by {@link #fromString}.
   *
   * <p>The returned {@link String} is insensitive to the {@link RepositoryName} associated with the
   * {@link PackageSpecification}.
   */
  public abstract String toStringWithoutRepository();

  /**
   * Parses the provided {@link String} into a {@link PackageSpecification}.
   *
   * <p>The {@link String} must have one of the following forms:
   *
   * <ul>
   * <li>The full name of a single package, without repository qualification, prefixed with "//"
   *     (e.g. "//foo/bar"). This results in a {@link PackageSpecification} that contains exactly
   *     the named package.
   * <li>The full name of a single package, without repository qualification, prefixed with "//",
   *     and suffixed with "/..." (e.g. "//foo/bar/...") This results in a {@link
   *     PackageSpecification} that contains all transitive subpackages of that package, inclusive.
   * <li>Exactly "//...". This results in a {@link PackageSpecification} that contains all packages.
   * </ul>
   *
   * <p>If and only if the {@link String} is one of the first two forms, the returned {@link
   * PackageSpecification} is specific to the provided {@link RepositoryName}. Note that it is not
   * possible to construct a repository-specific {@link PackageSpecification} for all transitive
   * subpackages of the root package (i.e. a repository-specific "//...").
   *
   * <p>Throws {@link InvalidPackageSpecificationException} if the {@link String} cannot be parsed.
   */
  public static PackageSpecification fromString(RepositoryName repositoryName, String spec)
      throws InvalidPackageSpecificationException {
    String result = spec;
    boolean allBeneath = false;
    if (result.endsWith(ALL_BENEATH_SUFFIX)) {
      allBeneath = true;
      result = result.substring(0, result.length() - ALL_BENEATH_SUFFIX.length());
      if (result.equals("/")) {
        // spec was "//...".
        return AllPackages.EVERYTHING;
      }
    }

    if (!spec.startsWith("//")) {
      throw new InvalidPackageSpecificationException(
          String.format("invalid package name '%s': must start with '//'", spec));
    }

    PackageIdentifier packageId;
    try {
      packageId = PackageIdentifier.parse(result);
    } catch (LabelSyntaxException e) {
      throw new InvalidPackageSpecificationException(
          String.format("invalid package name '%s': %s", spec, e.getMessage()));
    }
    Verify.verify(packageId.getRepository().isDefault());

    PackageIdentifier packageIdForSpecifiedRepository =
        PackageIdentifier.create(repositoryName, packageId.getPackageFragment());
    return allBeneath
        ? new AllPackagesBeneath(packageIdForSpecifiedRepository)
        : new SinglePackage(packageIdForSpecifiedRepository);
  }

  /**
   * Parses the provided {@link Label} into a {@link PackageSpecification} specific to the {@link
   * RepositoryName} associated with the label.
   *
   * <p>If {@code label.getName.equals("__pkg__")} then this results in a {@link
   * PackageSpecification} that contains exactly the named package.
   *
   * <p>If {@code label.getName.equals("__subpackages__")} then this results in a {@link
   * PackageSpecification} that contains all transitive subpackages of that package, inclusive.
   *
   * <p>If the label's name is neither "__pkg__" nor "__subpackages__", this returns {@code null}.
   *
   * <p>Note that there is no {@link Label} associated with the {@link RepositoryName}-agnostic "all
   * packages" specification (corresponding to {@code #fromString(null, "//...")}).
   */
  static PackageSpecification fromLabel(Label label) {
    if (label.getName().equals(PACKAGE_LABEL)) {
      return new SinglePackage(label.getPackageIdentifier());
    } else if (label.getName().equals(SUBTREE_LABEL)) {
      return new AllPackagesBeneath(label.getPackageIdentifier());
    } else {
      return null;
    }
  }

  public static PackageSpecification everything() {
    return AllPackages.EVERYTHING;
  }

  private static class SinglePackage extends PackageSpecification {
    private PackageIdentifier singlePackageName;

    private SinglePackage(PackageIdentifier packageName) {
      this.singlePackageName = packageName;
    }

    @Override
    public boolean containsPackage(PackageIdentifier packageName) {
      return this.singlePackageName.equals(packageName);
    }

    @Override
    public String toStringWithoutRepository() {
      return "//" + singlePackageName.getPackageFragment().getPathString();
    }

    @Override
    public String toString() {
      return singlePackageName.toString();
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof SinglePackage)) {
        return false;
      }
      SinglePackage that = (SinglePackage) o;
      return singlePackageName.equals(that.singlePackageName);
    }

    @Override
    public int hashCode() {
      return singlePackageName.hashCode();
    }
  }

  private static class AllPackagesBeneath extends PackageSpecification {
    private PackageIdentifier prefix;

    private AllPackagesBeneath(PackageIdentifier prefix) {
      this.prefix = prefix;
    }

    @Override
    public boolean containsPackage(PackageIdentifier packageName) {
      return packageName.getRepository().equals(prefix.getRepository())
          && packageName.getPackageFragment().startsWith(prefix.getPackageFragment());
    }

    @Override
    public String toStringWithoutRepository() {
      return "//" + prefix.getPackageFragment().getPathString() + ALL_BENEATH_SUFFIX;
    }

    @Override
    public String toString() {
      if (prefix.getPackageFragment().equals(PathFragment.EMPTY_FRAGMENT)) {
        return "//...";
      }
      return prefix + "/...";
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof AllPackagesBeneath)) {
        return false;
      }
      AllPackagesBeneath that = (AllPackagesBeneath) o;
      return prefix.equals(that.prefix);
    }

    @Override
    public int hashCode() {
      return prefix.hashCode();
    }
  }

  private static class AllPackages extends PackageSpecification {

    private static final PackageSpecification EVERYTHING = new AllPackages();

    @Override
    public boolean containsPackage(PackageIdentifier packageName) {
      return true;
    }

    @Override
    public String toStringWithoutRepository() {
      return "//...";
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof AllPackages;
    }

    @Override
    public int hashCode() {
      return "//...".hashCode();
    }

    @Override
    public String toString() {
      return "//...";
    }
  }

  /** Exception class to be thrown when a specification cannot be parsed. */
  static class InvalidPackageSpecificationException extends Exception {
    private InvalidPackageSpecificationException(String message) {
      super(message);
    }
  }
}
