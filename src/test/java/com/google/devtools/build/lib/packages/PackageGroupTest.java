// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.packages.util.TargetDataSubject.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for PackageGroup. */
@RunWith(JUnit4.class)
public class PackageGroupTest extends PackageLoadingTestCase {

  @Test
  public void testDoesNotFailHorribly() throws Exception {
    scratch.file(
        "fruits/BUILD",
        """
        package_group(
            name = "apple",
            packages = ["//random"],
        )
        """);
    // Note that, for our purposes, the packages listed in the package_group need not exist.

    getPackageGroup("fruits", "apple");
  }

  // Regression test for: "Package group with empty name causes Blaze exception"
  @Test
  public void testEmptyPackageGroupNameDoesNotThrow() throws Exception {
    scratch.file(
        "strawberry/BUILD",
        """
        package_group(
            name = "",
            packages = [],
        )
        """);

    reporter.removeHandler(failFastHandler);
    // Call getTarget() directly since getPackageGroup() requires a name.
    getTarget("//strawberry:BUILD");
    assertContainsEvent("package group has invalid name");
  }

  @Test
  public void testAbsolutePackagesWork() throws Exception {
    scratch.file(
        "fruits/BUILD",
        """
        package_group(
            name = "apple",
            packages = ["//vegetables"],
        )
        """);

    PackageGroup grp = getPackageGroup("fruits", "apple");
    assertThat(grp.contains(pkgId("vegetables"))).isTrue();
    assertThat(grp.contains(pkgId("fruits/vegetables"))).isFalse();
  }

  @Test
  public void testPackagesWithoutDoubleSlashDoNotWork() throws Exception {
    scratch.file(
        "fruits/BUILD",
        """
        package_group(
            name = "apple",
            packages = ["vegetables"],
        )
        """);

    reporter.removeHandler(failFastHandler);
    getPackageGroup("fruits", "apple");
    assertContainsEvent("invalid package name 'vegetables'");
  }

  @Test
  public void testPackagesWithRepositoryDoNotWork() throws Exception {
    scratch.file(
        "fruits/BUILD",
        """
        package_group(
            name = "banana",
            packages = ["@veggies//:cucumber"],
        )
        """);

    reporter.removeHandler(failFastHandler);
    getPackageGroup("fruits", "banana");
    assertContainsEvent("invalid package name '@veggies//:cucumber'");
  }

  @Test
  public void testAllPackagesInMainRepositoryDoesNotWork() throws Exception {
    scratch.file(
        "fruits/BUILD",
        """
        package_group(
            name = "apple",
            packages = ["@//..."],
        )
        """);

    reporter.removeHandler(failFastHandler);
    getPackageGroup("fruits", "apple");
    assertContainsEvent("invalid package name '@//...'");
  }

  // TODO(brandjon): It'd be nice to include a test here that you can cross repositories via
  // `includes`: if package_group //:A includes package_group @repo//:B that has "//foo" in its
  // `packages`, then //:A admits package @repo//foo. Unfortunately PackageLoadingTestCase doesn't
  // support resolving repos, but similar functionality is tested in
  // BzlLoadFunctionTest#testBzlVisibility_enumeratedPackagesMultipleRepos.

  @Test
  public void testTargetNameAsPackageDoesNotWork1() throws Exception {
    scratch.file(
        "fruits/BUILD",
        """
        package_group(
            name = "apple",
            packages = ["//vegetables:carrot"],
        )
        """);

    reporter.removeHandler(failFastHandler);
    getPackageGroup("fruits", "apple");
    assertContainsEvent("invalid package name '//vegetables:carrot'");
  }

  @Test
  public void testTargetNameAsPackageDoesNotWork2() throws Exception {
    scratch.file(
        "fruits/BUILD",
        """
        package_group(
            name = "apple",
            packages = [":carrot"],
        )
        """);

    reporter.removeHandler(failFastHandler);
    getPackageGroup("fruits", "apple");
    assertContainsEvent("invalid package name ':carrot'");
  }

  @Test
  public void testAllBeneathSpecificationWorks() throws Exception {
    scratch.file(
        "fruits/BUILD",
        """
        package_group(
            name = "maracuja",
            packages = ["//tropics/..."],
        )
        """);

    getPackageGroup("fruits", "maracuja");
  }

  @Test
  public void testNegative() throws Exception {
    scratch.file(
        "test/BUILD",
        """
        package_group(
            name = "packages",
            packages = [
                "-//four",
                "-//three",
                "//one",
                "//two",
            ],
        )
        """);

    PackageGroup grp = getPackageGroup("test", "packages");
    assertThat(grp.contains(pkgId("one"))).isTrue();
    assertThat(grp.contains(pkgId("two"))).isTrue();
    assertThat(grp.contains(pkgId("three"))).isFalse();
    assertThat(grp.contains(pkgId("four"))).isFalse();
  }

  @Test
  public void testNegative_noSubpackages() throws Exception {
    scratch.file(
        "test/BUILD",
        """
        package_group(
            name = "packages",
            packages = [
                "-//pkg/one",
                "//pkg/...",
            ],
        )
        """);

    PackageGroup grp = getPackageGroup("test", "packages");
    assertThat(grp.contains(pkgId("pkg"))).isTrue();
    assertThat(grp.contains(pkgId("pkg/one"))).isFalse();
    assertThat(grp.contains(pkgId("pkg/one/two"))).isTrue();
  }

  @Test
  public void testNegative_subpackages() throws Exception {
    scratch.file(
        "test/BUILD",
        """
        package_group(
            name = "packages",
            packages = [
                "-//pkg/one/...",
                "//pkg/...",
            ],
        )
        """);

    PackageGroup grp = getPackageGroup("test", "packages");
    assertThat(grp.contains(pkgId("pkg"))).isTrue();
    assertThat(grp.contains(pkgId("pkg/one"))).isFalse();
    assertThat(grp.contains(pkgId("pkg/one/two"))).isFalse();
  }

  @Test
  public void testEverythingSpecificationWorks() throws Exception {
    setBuildLanguageOptions("--incompatible_package_group_has_public_syntax=true");

    scratch.file(
        "fruits/BUILD",
        """
        package_group(
            name = "mango",
            packages = ["public"],
        )
        """);
    PackageGroup grp = getPackageGroup("fruits", "mango");

    // Assert that we're using the right package spec.
    assertThat(grp.getContainedPackages(/*includeDoubleSlash=*/ true)).containsExactly("public");
    // Assert that this package spec contains packages from both inside and outside the main repo.
    assertThat(grp.contains(pkgId("pkg"))).isTrue();
    assertThat(grp.contains(pkgId("somerepo", "pkg"))).isTrue();
  }

  @Test
  public void testNothingSpecificationWorks() throws Exception {
    setBuildLanguageOptions("--incompatible_package_group_has_public_syntax=true");

    scratch.file(
        "fruits/BUILD",
        """
        package_group(
            name = "mango",
            packages = ["private"],
        )
        """);
    PackageGroup grp = getPackageGroup("fruits", "mango");

    // Assert that we're using the right package spec.
    assertThat(grp.getContainedPackages(/*includeDoubleSlash=*/ true)).containsExactly("private");
    assertThat(grp.contains(pkgId("anything"))).isFalse();
  }

  @Test
  public void testPublicPrivateAreNotAccessibleWithoutFlag() throws Exception {
    setBuildLanguageOptions(
        // Flag being tested
        "--incompatible_package_group_has_public_syntax=false",
        // Must also be disabled in order to disable the above
        "--incompatible_fix_package_group_reporoot_syntax=false");

    scratch.file(
        "foo/BUILD",
        """
        package_group(
            name = "grp1",
            packages = ["public"],
        )
        """);
    scratch.file(
        "bar/BUILD",
        """
        package_group(
            name = "grp2",
            packages = ["private"],
        )
        """);

    reporter.removeHandler(failFastHandler);
    getPackageGroup("foo", "grp1");
    assertContainsEvent(
        "Use of \"public\" package specification requires enabling"
            + " --incompatible_package_group_has_public_syntax");
    getPackageGroup("bar", "grp2");
    assertContainsEvent(
        "Use of \"private\" package specification requires enabling"
            + " --incompatible_package_group_has_public_syntax");
  }

  @Test
  public void testRepoRootSubpackagesIsPublic_withoutFlag() throws Exception {
    setBuildLanguageOptions("--incompatible_fix_package_group_reporoot_syntax=false");

    scratch.file(
        "fruits/BUILD",
        """
        package_group(
            name = "mango",
            packages = ["//..."],
        )
        """);
    PackageGroup grp = getPackageGroup("fruits", "mango");

    // Use includeDoubleSlash=true to make package spec stringification distinguish AllPackages from
    // AllPackagesBeneath with empty package path.
    assertThat(grp.getContainedPackages(/*includeDoubleSlash=*/ true))
        // Assert that "//..." gave us AllPackages.
        .containsExactly("public");
    assertThat(grp.contains(pkgId("pkg"))).isTrue();
    assertThat(grp.contains(pkgId("somerepo", "pkg"))).isTrue();
  }

  @Test
  public void testRepoRootSubpackagesIsNotPublic_withFlag() throws Exception {
    setBuildLanguageOptions(
        "--incompatible_package_group_has_public_syntax=true",
        "--incompatible_fix_package_group_reporoot_syntax=true");

    scratch.file(
        "fruits/BUILD",
        """
        package_group(
            name = "mango",
            packages = ["//..."],
        )
        """);
    PackageGroup grp = getPackageGroup("fruits", "mango");

    // Use includeDoubleSlash=true to make package spec stringification distinguish AllPackages from
    // AllPackagesBeneath with empty package path.
    assertThat(grp.getContainedPackages(/*includeDoubleSlash=*/ true))
        // Assert that "//..." gave us AllPackagesBeneath.
        .containsExactly("//...");
    assertThat(grp.contains(pkgId("pkg"))).isTrue();
    assertThat(grp.contains(pkgId("somerepo", "pkg"))).isFalse();
  }

  @Test
  public void testCannotUseNewRepoRootSyntaxWithoutPublicSyntax() throws Exception {
    setBuildLanguageOptions(
        "--incompatible_package_group_has_public_syntax=false",
        "--incompatible_fix_package_group_reporoot_syntax=true");

    scratch.file(
        "fruits/BUILD",
        """
        package_group(
            name = "mango",
            packages = ["//something"],
        )
        """);

    reporter.removeHandler(failFastHandler);
    getPackageGroup("fruits", "mango");
    assertContainsEvent("Cannot use new \"//...\" meaning without allowing new \"public\" syntax.");
  }

  @Test
  public void testNegative_repoRootSubpackages() throws Exception {
    scratch.file(
        "test/BUILD",
        """
        package_group(
            name = "packages",
            packages = [
                "-//...",
                "//pkg/one",
            ],
        )
        """);

    PackageGroup grp = getPackageGroup("test", "packages");
    assertThat(grp.contains(pkgId("pkg"))).isFalse();
    assertThat(grp.contains(pkgId("pkg/one"))).isFalse();
    assertThat(grp.contains(pkgId("pkg/one/two"))).isFalse();
  }

  @Test
  public void testNegative_public() throws Exception {
    setBuildLanguageOptions("--incompatible_package_group_has_public_syntax=true");

    scratch.file(
        "fruits/BUILD",
        """
        package_group(
            name = "apple",
            packages = ["-public"],
        )
        """);

    reporter.removeHandler(failFastHandler);
    getPackageGroup("fruits", "apple");
    assertContainsEvent("Cannot negate \"public\" package specification");
  }

  @Test
  public void testNegative_private() throws Exception {
    setBuildLanguageOptions("--incompatible_package_group_has_public_syntax=true");

    scratch.file(
        "fruits/BUILD",
        """
        package_group(
            name = "apple",
            packages = ["-private"],
        )
        """);

    reporter.removeHandler(failFastHandler);
    getPackageGroup("fruits", "apple");
    assertContainsEvent("Cannot negate \"private\" package specification");
  }

  @Test
  public void testDuplicatePackage() throws Exception {
    scratch.file(
        "test/BUILD",
        """
        package_group(
            name = "packages",
            packages = ["//one/two"],
        )
        """);

    PackageGroup grp = getPackageGroup("test", "packages");
    assertThat(grp.contains(pkgId("one/two"))).isTrue();
  }

  @Test
  public void testStringification() throws Exception {
    RepositoryName main = RepositoryName.MAIN;
    RepositoryName other = RepositoryName.create("other");
    PackageGroupContents contents =
        PackageGroupContents.create(
            ImmutableList.of(
                pkgSpec(main, "//a"),
                pkgSpec(main, "//a/b/..."),
                pkgSpec(main, "-//c"),
                pkgSpec(main, "-//c/d/..."),
                pkgSpec(main, "//..."),
                pkgSpec(main, "-//..."),
                pkgSpec(main, "//"),
                pkgSpec(main, "-//"),
                pkgSpec(other, "//z"),
                pkgSpec(other, "//..."),
                pkgSpec(main, "public"),
                pkgSpec(main, "private")));
    assertThat(contents.packageStrings(/* includeDoubleSlash= */ false))
        .containsExactly(
            "a",
            "",
            "@@other//z",
            "a/b/...",
            "//...",
            "@@other//...",
            "-c",
            "-",
            "-c/d/...",
            "-//...",
            "//...", // legacy syntax for public
            "private");
    assertThat(contents.packageStrings(/* includeDoubleSlash= */ true))
        .containsExactly(
            "//a",
            "//a/b/...",
            "-//c",
            "-//c/d/...",
            "//...",
            "-//...",
            "//",
            "-//",
            "@@other//z",
            "@@other//...",
            "public",
            "private");
    assertThat(contents.packageStringsWithDoubleSlashAndWithoutRepository())
        .containsExactly(
            "//a",
            "//a/b/...",
            "-//c",
            "-//c/d/...",
            "//...",
            "-//...",
            "//",
            "-//",
            "//z",
            "//...",
            "public",
            "private");
  }

  @Test
  public void testReduceForSerialization() throws Exception {
    setBuildLanguageOptions("--incompatible_package_group_has_public_syntax=true");

    scratch.file(
        "fruits/BUILD",
        """
        package_group(
            name = "apple",
            packages = ["//vegetables"],
        )

        package_group(
            name = "mango",
            packages = ["public"],
        )
        """);
    PackageGroup grp = getPackageGroup("fruits", "apple");
    assertThat(grp).hasSamePropertiesAs(grp.reduceForSerialization());

    grp = getPackageGroup("fruits", "mango");
    assertThat(grp).hasSamePropertiesAs(grp.reduceForSerialization());
  }

  /** Convenience method for obtaining a PackageSpecification. */
  private PackageSpecification pkgSpec(RepositoryName repository, String spec) throws Exception {
    return PackageSpecification.fromString(
        repository, spec, /*allowPublicPrivate=*/ true, /*repoRootMeansCurrentRepo=*/ true);
  }

  /** Convenience method for obtaining a PackageIdentifier. */
  private PackageIdentifier pkgId(String packageName) throws Exception {
    return PackageIdentifier.createUnchecked(/*repository=*/ "", packageName);
  }

  /** Convenience method for obtaining a PackageIdentifier outside the main repo. */
  private PackageIdentifier pkgId(String repoName, String packageName) throws Exception {
    return PackageIdentifier.createUnchecked(repoName, packageName);
  }

  /** Evaluates and returns the requested package_group target. */
  private PackageGroup getPackageGroup(String pkg, String name) throws Exception {
    return (PackageGroup) getTarget("//" + pkg + ":" + name);
  }
}
