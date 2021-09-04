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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.LinkedHashMap;
import java.util.stream.Stream;

/**
 * Represents one of the following:
 *
 * <ul>
 *   <li>A single package (e.g. "//foo/bar")
 *   <li>All transitive subpackages of a package, inclusive (e.g. "//foo/bar/...", which includes
 *       "//foo/bar")
 *   <li>All packages (i.e. "//...")
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
  private static final String NEGATIVE_PREFIX = "-";

  /** Returns {@code true} if the package spec includes the provided {@code packageName}. */
  protected abstract boolean containsPackage(PackageIdentifier packageName);

  /**
   * Returns a {@link String} representation of the {@link PackageSpecification} of the same format
   * accepted by {@link #fromString}.
   *
   * <p>The returned {@link String} is insensitive to the {@link RepositoryName} associated with the
   * {@link PackageSpecification}.
   */
  protected abstract String toStringWithoutRepository();

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
    boolean negative = false;
    if (result.startsWith(NEGATIVE_PREFIX)) {
      negative = true;
      result = result.substring(NEGATIVE_PREFIX.length());
    }
    PackageSpecification packageSpecification = fromStringPositive(repositoryName, result);
    return negative ? new NegativePackageSpecification(packageSpecification) : packageSpecification;
  }

  private static PackageSpecification fromStringPositive(RepositoryName repositoryName, String spec)
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

  @AutoCodec
  @VisibleForSerialization
  static final class SinglePackage extends PackageSpecification {
    private final PackageIdentifier singlePackageName;

    @VisibleForSerialization
    SinglePackage(PackageIdentifier singlePackageName) {
      this.singlePackageName = singlePackageName;
    }

    @Override
    protected boolean containsPackage(PackageIdentifier packageName) {
      return this.singlePackageName.equals(packageName);
    }

    @Override
    protected String toStringWithoutRepository() {
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

  @AutoCodec
  @VisibleForSerialization
  static final class AllPackagesBeneath extends PackageSpecification {
    private final PackageIdentifier prefix;

    @VisibleForSerialization
    AllPackagesBeneath(PackageIdentifier prefix) {
      this.prefix = prefix;
    }

    @Override
    protected boolean containsPackage(PackageIdentifier packageName) {
      return packageName.getRepository().equals(prefix.getRepository())
          && packageName.getPackageFragment().startsWith(prefix.getPackageFragment());
    }

    @Override
    protected String toStringWithoutRepository() {
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

  /** A package specification for a negative match, e.g. {@code -//pkg/sub/...}. */
  @AutoCodec
  @VisibleForSerialization
  static final class NegativePackageSpecification extends PackageSpecification {
    private final PackageSpecification delegate;

    NegativePackageSpecification(PackageSpecification delegate) {
      this.delegate = delegate;
    }

    @Override
    protected boolean containsPackage(PackageIdentifier packageName) {
      return delegate.containsPackage(packageName);
    }

    @Override
    protected String toStringWithoutRepository() {
      return "-" + delegate.toStringWithoutRepository();
    }

    @Override
    public int hashCode() {
      return delegate.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      return obj instanceof NegativePackageSpecification
          && delegate.equals(((NegativePackageSpecification) obj).delegate);
    }

    @Override
    public String toString() {
      return "-" + delegate;
    }
  }

  @AutoCodec
  @VisibleForSerialization
  static final class AllPackages extends PackageSpecification {
    private static final PackageSpecification EVERYTHING = new AllPackages();

    @Override
    protected boolean containsPackage(PackageIdentifier packageName) {
      return true;
    }

    @Override
    protected String toStringWithoutRepository() {
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

  /**
   * A collection of {@link PackageSpecification}s from a {@code package_group}, which supports
   * testing a given package for containment (see {@link #containedPackages()}}.
   */
  @Immutable
  @AutoCodec
  public static final class PackageGroupContents {
    private final ImmutableMap<PackageIdentifier, PackageSpecification> singlePackages;
    private final ImmutableList<PackageSpecification> negativePackageSpecifications;
    private final ImmutableList<PackageSpecification> allSpecifications;

    @VisibleForSerialization
    PackageGroupContents(
        ImmutableMap<PackageIdentifier, PackageSpecification> singlePackages,
        ImmutableList<PackageSpecification> negativePackageSpecifications,
        ImmutableList<PackageSpecification> allSpecifications) {

      this.singlePackages = singlePackages;
      this.negativePackageSpecifications = negativePackageSpecifications;
      this.allSpecifications = allSpecifications;
    }

    /**
     * Creates a {@link PackageGroupContents} representing a collection of {@link
     * PackageSpecification}s.
     */
    public static PackageGroupContents create(
        ImmutableList<PackageSpecification> packageSpecifications) {
      LinkedHashMap<PackageIdentifier, PackageSpecification> singlePackageBuilder =
          new LinkedHashMap<>();
      ImmutableList.Builder<PackageSpecification> negativePackageSpecificationsBuilder =
          ImmutableList.builder();
      ImmutableList.Builder<PackageSpecification> allSpecificationsBuilder =
          ImmutableList.builder();

      for (PackageSpecification packageSpecification : packageSpecifications) {
        if (packageSpecification instanceof SinglePackage) {
          singlePackageBuilder.put(
              ((SinglePackage) packageSpecification).singlePackageName, packageSpecification);
        } else if (packageSpecification instanceof NegativePackageSpecification) {
          negativePackageSpecificationsBuilder.add(packageSpecification);
        } else {
          allSpecificationsBuilder.add(packageSpecification);
          if (!(packageSpecification instanceof AllPackages)
              && !(packageSpecification instanceof AllPackagesBeneath)) {
            throw new IllegalStateException(
                "Instance of unhandled class " + packageSpecification.getClass());
          }
        }
      }
      return new PackageGroupContents(
          ImmutableMap.copyOf(singlePackageBuilder),
          negativePackageSpecificationsBuilder.build(),
          allSpecificationsBuilder.build());
    }

    /**
     * Returns {@code true} if the package specifications include the provided {@code packageName}.
     * That is, at least one positive package specification matches, and no negative package
     * specifications match.
     */
    public boolean containsPackage(PackageIdentifier packageIdentifier) {
      // DO NOT use streams or iterators here as they create excessive garbage.

      // if some negative matches, returns false immediately.
      for (int i = 0; i < negativePackageSpecifications.size(); i++) {
        if (negativePackageSpecifications.get(i).containsPackage(packageIdentifier)) {
          return false;
        }
      }

      if (singlePackages.containsKey(packageIdentifier)) {
        return true;
      }

      for (int i = 0; i < allSpecifications.size(); i++) {
        if (allSpecifications.get(i).containsPackage(packageIdentifier)) {
          return true;
        }
      }
      return false;
    }

    /**
     * Returns {@link String} representations of the component {@link PackageSpecification}s of the
     * same format accepted by {@link #fromString}.
     */
    public Stream<String> containedPackages() {
      return getStream().map(PackageSpecification::toString);
    }

    /**
     * Returns {@link String} representations of the component {@link PackageSpecification}s of the
     * same format accepted by {@link #fromString}.
     *
     * <p>The returned {@link String}s are insensitive to the {@link RepositoryName} associated with
     * the {@link PackageSpecification}.
     */
    public Stream<String> containedPackagesWithoutRepository() {
      return getStream().map(PackageSpecification::toStringWithoutRepository);
    }

    private Stream<PackageSpecification> getStream() {
      return Stream.concat(
          Stream.concat(allSpecifications.stream(), negativePackageSpecifications.stream()),
          singlePackages.values().stream());
    }
  }
}
