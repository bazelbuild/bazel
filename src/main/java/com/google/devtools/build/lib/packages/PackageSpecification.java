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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.vfs.PathFragment;
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
// TODO(b/279784354): Delete PackageSpecification; reimplement bzl visibility using something else.
public abstract class PackageSpecification {
  private static final String PUBLIC_VISIBILITY = "public";
  private static final String PRIVATE_VISIBILITY = "private";
  private static final String ALL_BENEATH_SUFFIX = "/...";
  private static final String NEGATIVE_PREFIX = "-";

  // Used for interpreting `visibility` labels.
  private static final String PACKAGE_LABEL = "__pkg__";
  private static final String SUBTREE_LABEL = "__subpackages__";

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

  @Override
  public String toString() {
    return asString(/*includeDoubleSlash=*/ false);
  }

  /**
   * Parses the string {@code spec} into a {@link PackageSpecification}, within the context of the
   * given repository name.
   *
   * <p>{@code spec} may have the following forms:
   *
   * <ol>
   *   <li>The full name of a single package, without repository qualification, prefixed with "//"
   *       (e.g. "//foo/bar"). The resulting specification contains exactly that package.
   *   <li>The same, but suffixed with "/..." for a non-root package ("//foo/bar/...") or "..." for
   *       the root package ("//..."). The resulting specification contains that package and all its
   *       subpackages.
   *   <li>The string constants "public" or "private". The resulting specification contains either
   *       all packages or no packages, respectively.
   * </ol>
   *
   * In the first two cases, the repository of the given package name is taken to be {@code
   * repositoryName}. In the third case the repository name is ignored.
   *
   * <p>In the first two cases, {@code spec} may also be prefixed by a "-". The resulting
   * specification contains the same set of packages but is marked as being negated. (Negation logic
   * is applied at the level of {@link PackageGroupContents}.)
   *
   * <p>Setting {@code allowPublicPrivate} to false disallows the string constants "public" and
   * "private". Note that if {@link #asString} is called with {@code includeDoubleSlash} set to
   * false, the stringification of "public" and "//public" is ambiguous (likewise for private),
   * hence why it might be appropriate to prohibit these forms.
   *
   * <p>Setting {@code repoRootMeansCurrentRepo} to false restores the following legacy behavior: In
   * the specific case where {@code spec} is "//..." (or its negation), the package specification
   * contains <i>all</i> packages (possibly marked as negated) rather than just those packages in
   * {@code repositoryName}. In other words, "//..." behaves the same as "public". However, "//"
   * still represents just the root package of {@code repositoryName}.
   *
   * <p>To protect against requiring users to update to a disallowed syntax, it is illegal to
   * specify {@code repoRootMeansCurrentRepo} without also specifying {@code allowPublicPrivate}.
   *
   * @throws InvalidPackageSpecificationException if the string does not fit one of these forms
   */
  // TODO(#16365): Remove allowPublicPrivate.
  // TODO(#16324): Remove legacy behavior and repoRootMeansCurrentRepo param.
  public static PackageSpecification fromString(
      RepositoryName repositoryName,
      String spec,
      boolean allowPublicPrivate,
      boolean repoRootMeansCurrentRepo)
      throws InvalidPackageSpecificationException {
    if (repoRootMeansCurrentRepo && !allowPublicPrivate) {
      throw new InvalidPackageSpecificationException(
          "Cannot use new \"//...\" meaning without allowing new \"public\" syntax. Try enabling"
              + " --incompatible_package_group_has_public_syntax or disabling"
              + " --incompatible_fix_package_group_reporoot_syntax.");
    }
    if (!allowPublicPrivate
        && (spec.equals(PUBLIC_VISIBILITY) || spec.equals(PRIVATE_VISIBILITY))) {
      throw new InvalidPackageSpecificationException(
          String.format(
              "Use of \"%s\" package specification requires enabling"
                  + " --incompatible_package_group_has_public_syntax",
              spec));
    }
    boolean negative = false;
    if (spec.startsWith(NEGATIVE_PREFIX)) {
      negative = true;
      spec = spec.substring(NEGATIVE_PREFIX.length());
      if (spec.equals(PUBLIC_VISIBILITY) || spec.equals(PRIVATE_VISIBILITY)) {
        throw new InvalidPackageSpecificationException(
            String.format("Cannot negate \"%s\" package specification", spec));
      }
    }
    PackageSpecification packageSpecification =
        fromStringPositive(repositoryName, spec, repoRootMeansCurrentRepo);
    return negative ? new NegativePackageSpecification(packageSpecification) : packageSpecification;
  }

  private static PackageSpecification fromStringPositive(
      RepositoryName repositoryName, String spec, boolean repoRootMeansCurrentRepo)
      throws InvalidPackageSpecificationException {
    if (spec.equals(PUBLIC_VISIBILITY)) {
      return AllPackages.INSTANCE;
    } else if (spec.equals(PRIVATE_VISIBILITY)) {
      return NoPackages.INSTANCE;
    }
    if (!spec.startsWith("//")) {
      throw new InvalidPackageSpecificationException(
          String.format(
              "invalid package name '%s': must start with '//' or be 'public' or 'private'", spec));
    }

    String pkgPath;
    boolean allBeneath = false;
    if (spec.endsWith(ALL_BENEATH_SUFFIX)) {
      allBeneath = true;
      pkgPath = spec.substring(0, spec.length() - ALL_BENEATH_SUFFIX.length());
      if (pkgPath.equals("/")) {
        // spec was "//...".
        if (repoRootMeansCurrentRepo) {
          pkgPath = "//";
        } else {
          // Legacy behavior: //... is "public".
          return AllPackages.INSTANCE;
        }
      }
    } else {
      pkgPath = spec;
    }

    PackageIdentifier unqualifiedPkgId;
    try {
      unqualifiedPkgId = PackageIdentifier.parse(pkgPath);
    } catch (LabelSyntaxException e) {
      throw new InvalidPackageSpecificationException(
          String.format("invalid package name '%s': %s", spec, e.getMessage()));
    }
    Verify.verify(unqualifiedPkgId.getRepository().isMain());

    PackageIdentifier pkgId =
        PackageIdentifier.create(repositoryName, unqualifiedPkgId.getPackageFragment());
    return allBeneath ? new AllPackagesBeneath(pkgId) : new SinglePackage(pkgId);
  }

  /**
   * Parses a string to a {@code PackageSpecification} for use with .bzl load visibility.
   *
   * <p>This rejects negative package patterns, and translates the exception type into {@code
   * EvalException}.
   *
   * <p>Note that load visibility package specifications always behave as if {@code
   * --incompatible_package_group_has_public_syntax} and {@code
   * --incompatible_fix_package_group_reporoot_syntax} are enabled.
   */
  public static PackageSpecification fromStringForBzlVisibility(
      RepositoryName repositoryName, String spec) throws EvalException {
    PackageSpecification result;
    try {
      result =
          fromString(
              repositoryName,
              spec,
              /*allowPublicPrivate=*/ true,
              /*repoRootMeansCurrentRepo=*/ true);
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
   * <p>Note that there is no {@link Label} associated with the {@link RepositoryName}-agnostic
   * "public" specification ("//..." under legacy semantics).
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
    return AllPackages.INSTANCE;
  }

  public static PackageSpecification nothing() {
    return NoPackages.INSTANCE;
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
      return PackageGroupContents.stringForSinglePackage(singlePackageName, includeDoubleSlash);
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
      return PackageGroupContents.stringForAllPackagesBeneath(prefix, includeDoubleSlash);
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
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      return obj instanceof NegativePackageSpecification
          && delegate.equals(((NegativePackageSpecification) obj).delegate);
    }

    @Override
    public int hashCode() {
      return NegativePackageSpecification.class.hashCode() ^ delegate.hashCode();
    }
  }

  @VisibleForSerialization
  static final class AllPackages extends PackageSpecification {
    @SerializationConstant @VisibleForSerialization
    static final PackageSpecification INSTANCE = new AllPackages();

    @Override
    protected boolean containsPackage(PackageIdentifier packageName) {
      return true;
    }

    @Override
    protected String asString(boolean includeDoubleSlash) {
      return PackageGroupContents.stringForAllPackages(includeDoubleSlash);
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof AllPackages;
    }

    @Override
    public int hashCode() {
      return AllPackages.class.hashCode();
    }
  }

  @VisibleForSerialization
  static final class NoPackages extends PackageSpecification {
    @SerializationConstant @VisibleForSerialization
    static final PackageSpecification INSTANCE = new NoPackages();

    @Override
    protected boolean containsPackage(PackageIdentifier packageName) {
      return false;
    }

    @Override
    protected String asString(boolean includeDoubleSlash) {
      return "private";
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof NoPackages;
    }

    @Override
    public int hashCode() {
      return NoPackages.class.hashCode();
    }
  }

  /** Exception class to be thrown when a specification cannot be parsed. */
  static class InvalidPackageSpecificationException extends Exception {
    private InvalidPackageSpecificationException(String message) {
      super(message);
    }
  }

  /**
   * Represents a collection of {@link PackageSpecification}s logically corresponding to a single
   * {@code package_group}'s {@code packages} attribute.
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
    // This class is optimized for memory and cpu.
    // TODO(b/279784354): Further improvements are possible.
    //
    // PackageGroupContents instances are retained through and after the analysis phase. Therefore,
    // in order to save memory, we don't retain PackageSpecification instances but instead unroll
    // them, storing just lists and sets of PackageIdentifier instances.
    //
    // To save cpu, we store the PackageIdentifier instances corresponding to positive/negative
    // single package specifications in sets so we get a fast-path hit on them. Also, we check
    // negatives first since package_group semantics require we check all negatives no matter what.
    // Both of these cpu optimizations combine well in practice since there are some package_group
    // targets with very large lists of negative single package specifications.

    private final ImmutableSet<PackageIdentifier> singlePackagePositives;
    private final ImmutableList<PackageIdentifier> allPackagesBeneathPositives;

    private final boolean hasPositiveAllPackages;

    private final ImmutableSet<PackageIdentifier> singlePackageNegatives;
    private final ImmutableList<PackageIdentifier> allPackagesBeneathNegatives;

    private final boolean hasPrivate;
    private final boolean hasNegativeAllPackages;

    private PackageGroupContents(ImmutableList<PackageSpecification> packageSpecifications) {
      var singlePackagePositivesBuilder = ImmutableSet.<PackageIdentifier>builder();
      var allPackagesBeneathPositivesBuilder = ImmutableList.<PackageIdentifier>builder();
      boolean hasPositiveAllPackages = false;
      var singlePackageNegativesBuilder = ImmutableSet.<PackageIdentifier>builder();
      var allPackagesBeneathNegativesBuilder = ImmutableList.<PackageIdentifier>builder();
      boolean hasPrivate = false;
      boolean hasNegativeAllPackages = false;

      for (PackageSpecification spec : packageSpecifications) {
        if (spec instanceof SinglePackage) {
          singlePackagePositivesBuilder.add(((SinglePackage) spec).singlePackageName);
          continue;
        }

        if (spec instanceof AllPackagesBeneath) {
          allPackagesBeneathPositivesBuilder.add(((AllPackagesBeneath) spec).prefix);
          continue;
        }

        if (spec instanceof AllPackages) {
          hasPositiveAllPackages = true;
          continue;
        }

        if (spec instanceof NoPackages) {
          // We can't drop NoPackages because it still needs to be serialized, e.g. in bazel query
          // output.
          hasPrivate = true;
          continue;
        }

        PackageSpecification delegate = ((NegativePackageSpecification) spec).delegate;
        if (delegate instanceof SinglePackage) {
          singlePackageNegativesBuilder.add(((SinglePackage) delegate).singlePackageName);
          continue;
        }

        if (delegate instanceof AllPackagesBeneath) {
          allPackagesBeneathNegativesBuilder.add(((AllPackagesBeneath) delegate).prefix);
          continue;
        }

        if (delegate instanceof AllPackages) {
          hasNegativeAllPackages = true;
          continue;
        }

        throw new IllegalStateException(spec.toString());
      }

      this.singlePackagePositives = singlePackagePositivesBuilder.build();
      this.allPackagesBeneathPositives = allPackagesBeneathPositivesBuilder.build();
      this.hasPositiveAllPackages = hasPositiveAllPackages;
      this.singlePackageNegatives = singlePackageNegativesBuilder.build();
      this.allPackagesBeneathNegatives = allPackagesBeneathNegativesBuilder.build();
      this.hasPrivate = hasPrivate;
      this.hasNegativeAllPackages = hasNegativeAllPackages;
    }

    /**
     * Creates a {@link PackageGroupContents} representing a collection of {@link
     * PackageSpecification}s.
     */
    public static PackageGroupContents create(
        ImmutableList<PackageSpecification> packageSpecifications) {
      return new PackageGroupContents(packageSpecifications);
    }

    /**
     * Returns true if the given package matches at least one of this {@code PackageGroupContents}'
     * positive specifications and none of its negative specifications.
     */
    public boolean containsPackage(PackageIdentifier packageIdentifier) {
      // DO NOT use streams or iterators here as they create excessive garbage.
      if (hasNegativeAllPackages) {
        return false;
      }
      if (singlePackageNegatives.contains(packageIdentifier)) {
        return false;
      }
      for (int i = 0; i < allPackagesBeneathNegatives.size(); i++) {
        if (matchesAllPackagesBeneath(packageIdentifier, allPackagesBeneathNegatives.get(i))) {
          return false;
        }
      }
      if (hasPositiveAllPackages) {
        return true;
      }
      if (singlePackagePositives.contains(packageIdentifier)) {
        return true;
      }
      for (int i = 0; i < allPackagesBeneathPositives.size(); i++) {
        if (matchesAllPackagesBeneath(packageIdentifier, allPackagesBeneathPositives.get(i))) {
          return true;
        }
      }
      return false;
    }

    private static boolean matchesAllPackagesBeneath(
        PackageIdentifier pkgId, PackageIdentifier prefix) {
      return pkgId.getRepository().equals(prefix.getRepository())
          && pkgId.getPackageFragment().startsWith(prefix.getPackageFragment());
    }

    /**
     * Does the equivalent of mapping {@link PackageSpecification#asString} to the component package
     * specs.
     *
     * <p>Note that strings for specs that cross repositories can't be reparsed using {@link
     * PackageSpecification#fromString}.
     *
     * <p>The special public constant will serialize as {@code "public"} if {@code
     * includeDoubleSlash} is true, and {@code "//..."} otherwise. The private constant will always
     * serialize as {@code "private"},
     */
    public ImmutableList<String> packageStrings(boolean includeDoubleSlash) {
      ImmutableList.Builder<String> resultBuilder = ImmutableList.builder();
      for (PackageIdentifier pkgId : singlePackagePositives) {
        resultBuilder.add(stringForSinglePackage(pkgId, includeDoubleSlash));
      }
      for (PackageIdentifier pkgId : allPackagesBeneathPositives) {
        resultBuilder.add(stringForAllPackagesBeneath(pkgId, includeDoubleSlash));
      }
      for (PackageIdentifier pkgId : singlePackageNegatives) {
        resultBuilder.add("-" + stringForSinglePackage(pkgId, includeDoubleSlash));
      }
      for (PackageIdentifier pkgId : allPackagesBeneathNegatives) {
        resultBuilder.add("-" + stringForAllPackagesBeneath(pkgId, includeDoubleSlash));
      }
      if (hasPositiveAllPackages) {
        resultBuilder.add(stringForAllPackages(includeDoubleSlash));
      }
      if (hasPrivate) {
        resultBuilder.add("private");
      }
      if (hasNegativeAllPackages) {
        resultBuilder.add("-" + stringForAllPackages(includeDoubleSlash));
      }
      return resultBuilder.build();
    }

    private static String stringForAllPackages(boolean includeDoubleSlash) {
      // Under legacy formatting rules, use legacy syntax. This avoids ambiguity between "public"
      // and "//public", and ensures that AllPackages is round-trippable when the value of
      // includeDoubleSlash matches allowPublicPrivate.
      return includeDoubleSlash ? "public" : "//...";
    }

    private static String stringForSinglePackage(
        PackageIdentifier pkgId, boolean includeDoubleSlash) {
      if (includeDoubleSlash) {
        return pkgId.getCanonicalForm();
      } else {
        // PackageIdentifier#toString implements the legacy behavior of omitting the double slash
        // for the main repo.
        return pkgId.toString();
      }
    }

    private static String stringForAllPackagesBeneath(
        PackageIdentifier pkgId, boolean includeDoubleSlash) {
      if (pkgId.getPackageFragment().equals(PathFragment.EMPTY_FRAGMENT)) {
        // Special case: Emit "//..." rather than suffixing "/...", which would yield "/...".
        // Make sure not to strip the repo in the case of "@repo//...".
        //
        // Note that "//..." is the desired result, not "...", even under the legacy behavior of
        // includeDoubleSlash=false.
        return pkgId.getCanonicalForm() + "...";
      }
      if (includeDoubleSlash) {
        return pkgId.getCanonicalForm() + ALL_BENEATH_SUFFIX;
      } else {
        // PackageIdentifier#toString implements the legacy behavior of omitting the double slash
        // for the main repo.
        return pkgId.toString() + ALL_BENEATH_SUFFIX;
      }
    }

    /**
     * /** Returns a string representation of this package spec without the repository, and which is
     * round-trippable through {@link #fromString}.
     *
     * <p>For instance, {@code @somerepo//pkg/subpkg/...} turns into {@code "//pkg/subpkg/..."}.
     *
     * <p>Omitting the repository means that the returned strings are ambiguous in the absence of
     * additional context. But, for instance, if interpreted with respect to a {@code
     * package_group}'s {@code packages} attribute, the strings always have the same repository as
     * the package group.
     *
     * <p>Note that this is ambiguous w.r.t. specs that reference other repositories.
     *
     * <p>The special public and private constants will serialize as {@code "public"} and {@code
     * "private"} respectively.
     */
    public ImmutableList<String> packageStringsWithDoubleSlashAndWithoutRepository() {
      ImmutableList.Builder<String> resultBuilder = ImmutableList.builder();
      for (PackageIdentifier pkgId : singlePackagePositives) {
        resultBuilder.add(stringForSinglePackageWithDoubleSlashAndWithoutRepository(pkgId));
      }
      for (PackageIdentifier pkgId : allPackagesBeneathPositives) {
        resultBuilder.add(stringForAllPackagesBeneathWithDoubleSlashAndWithoutRepository(pkgId));
      }
      for (PackageIdentifier pkgId : singlePackageNegatives) {
        resultBuilder.add("-" + stringForSinglePackageWithDoubleSlashAndWithoutRepository(pkgId));
      }
      for (PackageIdentifier pkgId : allPackagesBeneathNegatives) {
        resultBuilder.add(
            "-" + stringForAllPackagesBeneathWithDoubleSlashAndWithoutRepository(pkgId));
      }
      if (hasPositiveAllPackages) {
        resultBuilder.add("public");
      }
      if (hasPrivate) {
        resultBuilder.add("private");
      }
      return resultBuilder.build();
    }

    private static String stringForSinglePackageWithDoubleSlashAndWithoutRepository(
        PackageIdentifier pkgId) {
      return "//" + pkgId.getPackageFragment().getPathString();
    }

    private static String stringForAllPackagesBeneathWithDoubleSlashAndWithoutRepository(
        PackageIdentifier pkgId) {
      PathFragment pathFragment = pkgId.getPackageFragment();
      return pathFragment.equals(PathFragment.EMPTY_FRAGMENT)
          ? "//..."
          : "//" + pathFragment.getPathString() + ALL_BENEATH_SUFFIX;
    }
  }
}
