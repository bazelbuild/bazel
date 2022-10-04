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
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

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
   * Returns a string representation of this package spec.
   *
   * <p>The repository is included, unless it is the main repository, in which case there will be no
   * leading {@literal @}. For instance, {@code "@somerepo//pkg/subpkg"} and {@code
   * "//otherpkg/..."} are both valid outputs.
   *
   * <p>Note that since {@link #fromString} does not accept label strings with repositories, this
   * representation is not guaranteed to be round-trippable.
   *
   * <p>If {@code includeDoubleSlash} is false, then in the case of the main repository, the leading
   * {@code //} will also be omitted, so that the output looks like {@code otherpkg/...}. This form
   * is deprecated.
   */
  // TODO(b/77598306): Remove the parameter after switching all callers to pass true.
  protected abstract String asString(boolean includeDoubleSlash);

  /**
   * Returns a string representation of this package spec without the repository, and which is
   * round-trippable through {@link #fromString}.
   *
   * <p>For instance, {@code @somerepo//pkg/subpkg/...} turns into {@code "//pkg/subpkg/..."}.
   *
   * <p>Omitting the repository means that the returned strings are ambiguous in the absence of
   * additional context. But, for instance, if interpreted with respect to a {@code package_group}'s
   * {@code packages} attribute, the strings always have the same repository as the package group.
   */
  // TODO(brandjon): This method's main benefit is that it's round-trippable. We could eliminate
  // it in favor of asString() if we provided a public variant of fromString() that tolerates
  // repositories.
  protected abstract String asStringWithoutRepository();

  @Override
  public String toString() {
    return asString(/*includeDoubleSlash=*/ false);
  }

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
    Verify.verify(packageId.getRepository().isMain());

    PackageIdentifier packageIdForSpecifiedRepository =
        PackageIdentifier.create(repositoryName, packageId.getPackageFragment());
    return allBeneath
        ? new AllPackagesBeneath(packageIdForSpecifiedRepository)
        : new SinglePackage(packageIdForSpecifiedRepository);
  }

  /**
   * Parses a string to a {@code PackageSpecification} for use with .bzl visibility.
   *
   * <p>This rejects negative package patterns, and translates the exception type into {@code
   * EvalException}.
   */
  // TODO(b/22193153): Support negatives too.
  public static PackageSpecification fromStringForBzlVisibility(
      RepositoryName repositoryName, String spec) throws EvalException {
    PackageSpecification result;
    try {
      result = fromString(repositoryName, spec);
    } catch (InvalidPackageSpecificationException e) {
      throw new EvalException(e.getMessage());
    }
    if (result instanceof NegativePackageSpecification) {
      throw Starlark.errorf("Cannot use negative package patterns here");
    }
    return result;
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
  @Nullable
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

  private static final class SinglePackage extends PackageSpecification {
    private final PackageIdentifier singlePackageName;

    SinglePackage(PackageIdentifier singlePackageName) {
      this.singlePackageName = singlePackageName;
    }

    @Override
    protected boolean containsPackage(PackageIdentifier packageName) {
      return this.singlePackageName.equals(packageName);
    }

    @Override
    protected String asString(boolean includeDoubleSlash) {
      if (includeDoubleSlash) {
        return singlePackageName.getCanonicalForm();
      } else {
        // PackageIdentifier#toString implements the legacy behavior of omitting the double slash
        // for the main repo.
        return singlePackageName.toString();
      }
    }

    @Override
    protected String asStringWithoutRepository() {
      return "//" + singlePackageName.getPackageFragment().getPathString();
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

  private static final class AllPackagesBeneath extends PackageSpecification {
    private final PackageIdentifier prefix;

    AllPackagesBeneath(PackageIdentifier prefix) {
      this.prefix = prefix;
    }

    @Override
    protected boolean containsPackage(PackageIdentifier packageName) {
      return packageName.getRepository().equals(prefix.getRepository())
          && packageName.getPackageFragment().startsWith(prefix.getPackageFragment());
    }

    @Override
    protected String asString(boolean includeDoubleSlash) {
      if (prefix.getPackageFragment().equals(PathFragment.EMPTY_FRAGMENT)) {
        // Special case: Emit "//..." rather than suffixing "/...", which would yield "/...".
        // Make sure not to strip the repo in the case of "@repo//...".
        //
        // Note that "//..." is the desired result, not "...", even under the legacy behavior of
        // includeDoubleSlash=false.
        return prefix.getCanonicalForm() + "...";
      }
      if (includeDoubleSlash) {
        return prefix.getCanonicalForm() + ALL_BENEATH_SUFFIX;
      } else {
        // PackageIdentifier#toString implements the legacy behavior of omitting the double slash
        // for the main repo.
        return prefix.toString() + ALL_BENEATH_SUFFIX;
      }
    }

    @Override
    protected String asStringWithoutRepository() {
      return "//" + prefix.getPackageFragment().getPathString() + ALL_BENEATH_SUFFIX;
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
  private static final class NegativePackageSpecification extends PackageSpecification {
    private final PackageSpecification delegate;

    NegativePackageSpecification(PackageSpecification delegate) {
      this.delegate = delegate;
    }

    @Override
    protected boolean containsPackage(PackageIdentifier packageName) {
      return delegate.containsPackage(packageName);
    }

    @Override
    protected String asString(boolean includeDoubleSlash) {
      return "-" + delegate.asString(includeDoubleSlash);
    }

    @Override
    protected String asStringWithoutRepository() {
      return "-" + delegate.asStringWithoutRepository();
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
  }

  @VisibleForSerialization
  static final class AllPackages extends PackageSpecification {
    @SerializationConstant @VisibleForSerialization
    static final PackageSpecification EVERYTHING = new AllPackages();

    @Override
    protected boolean containsPackage(PackageIdentifier packageName) {
      return true;
    }

    @Override
    protected String asString(boolean includeDoubleSlash) {
      // Note that "//..." is the desired result, not "...", even under the legacy behavior of
      // includeDoubleSlash=false.
      return "//...";
    }

    @Override
    protected String asStringWithoutRepository() {
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
  }

  /** Exception class to be thrown when a specification cannot be parsed. */
  static class InvalidPackageSpecificationException extends Exception {
    private InvalidPackageSpecificationException(String message) {
      super(message);
    }
  }

  /**
   * A collection of {@link PackageSpecification}s logically corresponding to a single {@code
   * package_group}'s {@code packages} attribute.
   *
   * <p>Supports testing whether a given package is contained, taking into account negative specs.
   *
   * <p>Duplicate specs (e.g., ["//foo", "//foo"]) may or may not be deduplicated. Iteration order
   * may vary from the order in which specs were provided, but is guaranteed to be deterministic.
   *
   * <p>For modeling a {@code package_group}'s transitive contents (i.e., via the {@code includes}
   * attribute), see {@link PackageSpecificationProvider}.
   */
  @Immutable
  public static final class PackageGroupContents {
    // We separate PackageSpecifications by type.
    //   - Single positive specs are separate so that we can look them up quickly by package name,
    //     without requiring a linear search for a satisfying containsPackage().
    //   - Negative specs need to be separate because their semantics are different (they overrule
    //     any positive spec).
    // We don't bother separating out single negative specs. Negatives are pretty rare anyway.
    private final ImmutableMap<PackageIdentifier, PackageSpecification> singlePositives;
    private final ImmutableList<PackageSpecification> otherPositives;
    private final ImmutableList<PackageSpecification> negatives;

    private PackageGroupContents(
        ImmutableMap<PackageIdentifier, PackageSpecification> singlePositives,
        ImmutableList<PackageSpecification> otherPositives,
        ImmutableList<PackageSpecification> negatives) {
      this.singlePositives = singlePositives;
      this.otherPositives = otherPositives;
      this.negatives = negatives;
    }

    /**
     * Creates a {@link PackageGroupContents} representing a collection of {@link
     * PackageSpecification}s.
     */
    public static PackageGroupContents create(
        ImmutableList<PackageSpecification> packageSpecifications) {
      ImmutableMap.Builder<PackageIdentifier, PackageSpecification> singlePositives =
          ImmutableMap.builder();
      ImmutableList.Builder<PackageSpecification> otherPositives = ImmutableList.builder();
      ImmutableList.Builder<PackageSpecification> negatives = ImmutableList.builder();

      for (PackageSpecification spec : packageSpecifications) {
        if (spec instanceof SinglePackage) {
          singlePositives.put(((SinglePackage) spec).singlePackageName, spec);
        } else if (spec instanceof AllPackages || spec instanceof AllPackagesBeneath) {
          otherPositives.add(spec);
        } else if (spec instanceof NegativePackageSpecification) {
          negatives.add(spec);
        } else {
          throw new IllegalStateException(
              "Unhandled PackageSpecification subclass " + spec.getClass());
        }
      }
      return new PackageGroupContents(
          singlePositives.buildKeepingLast(), otherPositives.build(), negatives.build());
    }

    /**
     * Returns true if the given package matches at least one of this {@code PackageGroupContents}'
     * positive specifications and none of its negative specifications.
     */
    public boolean containsPackage(PackageIdentifier packageIdentifier) {
      // DO NOT use streams or iterators here as they create excessive garbage.

      // Check negatives first. If there's a match we get to bail out early. If not, we'd still have
      // to check all the negatives anyway.
      for (int i = 0; i < negatives.size(); i++) {
        if (negatives.get(i).containsPackage(packageIdentifier)) {
          return false;
        }
      }
      // Check the map in hopes of passing without having to do a linear scan over all other
      // positive specs.
      if (singlePositives.containsKey(packageIdentifier)) {
        return true;
      }
      // Oh well.
      for (int i = 0; i < otherPositives.size(); i++) {
        if (otherPositives.get(i).containsPackage(packageIdentifier)) {
          return true;
        }
      }
      return false;
    }

    /**
     * Maps {@link PackageSpecification#asString} to the component package specs.
     *
     * <p>Note that strings for specs that cross repositories can't be reparsed using {@link
     * PackageSpecification#fromString}.
     */
    public Stream<String> streamPackageStrings(boolean includeDoubleSlash) {
      return streamSpecs().map(spec -> spec.asString(includeDoubleSlash));
    }

    /**
     * Maps {@link PackageSpecification#asStringWithoutRepository} to the component package specs.
     *
     * <p>Note that this is ambiguous w.r.t. specs that reference other repositories.
     */
    public Stream<String> streamPackageStringsWithoutRepository() {
      return streamSpecs().map(PackageSpecification::asStringWithoutRepository);
    }

    private Stream<PackageSpecification> streamSpecs() {
      return Streams.concat(
          otherPositives.stream(), negatives.stream(), singlePositives.values().stream());
    }
  }
}
