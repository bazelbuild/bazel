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

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
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
        "package_group(",
        "    name = 'apple',",
        "    packages = ['//random'],",
        ")");
    // Note that, for our purposes, the packages listed in the package_group need not exist.

    getPackageGroup("fruits", "apple");
  }

  // Regression test for: "Package group with empty name causes Blaze exception"
  @Test
  public void testEmptyPackageGroupNameDoesNotThrow() throws Exception {
    scratch.file(
        "strawberry/BUILD", //
        "package_group(",
        "    name = '',",
        "    packages=[],",
        ")");

    reporter.removeHandler(failFastHandler);
    // Call getTarget() directly since getPackageGroup() requires a name.
    getTarget("//strawberry:BUILD");
    assertContainsEvent("package group has invalid name");
  }

  @Test
  public void testAbsolutePackagesWork() throws Exception {
    scratch.file(
        "fruits/BUILD",
        "package_group(",
        "    name = 'apple',",
        "    packages = ['//vegetables'],",
        ")");

    PackageGroup grp = getPackageGroup("fruits", "apple");
    assertThat(grp.contains(pkgId("vegetables"))).isTrue();
    assertThat(grp.contains(pkgId("fruits/vegetables"))).isFalse();
  }

  @Test
  public void testPackagesWithoutDoubleSlashDoNotWork() throws Exception {
    scratch.file(
        "fruits/BUILD",
        "package_group(",
        "    name = 'apple',",
        "    packages = ['vegetables'],",
        ")");

    reporter.removeHandler(failFastHandler);
    getPackageGroup("fruits", "apple");
    assertContainsEvent("invalid package name 'vegetables'");
  }

  @Test
  public void testPackagesWithRepositoryDoNotWork() throws Exception {
    scratch.file(
        "fruits/BUILD",
        "package_group(",
        "    name = 'banana',",
        "    packages = ['@veggies//:cucumber'],",
        ")");

    reporter.removeHandler(failFastHandler);
    getPackageGroup("fruits", "banana");
    assertContainsEvent("invalid package name '@veggies//:cucumber'");
  }

  @Test
  public void testAllPackagesInMainRepositoryDoesNotWork() throws Exception {
    scratch.file(
        "fruits/BUILD", //
        "package_group(",
        "    name = 'apple',",
        "    packages = ['@//...'],",
        ")");

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
        "package_group(",
        "    name = 'apple',",
        "    packages = ['//vegetables:carrot'],",
        ")");

    reporter.removeHandler(failFastHandler);
    getPackageGroup("fruits", "apple");
    assertContainsEvent("invalid package name '//vegetables:carrot'");
  }

  @Test
  public void testTargetNameAsPackageDoesNotWork2() throws Exception {
    scratch.file(
        "fruits/BUILD",
        "package_group(",
        "    name = 'apple',",
        "    packages = [':carrot'],",
        ")");

    reporter.removeHandler(failFastHandler);
    getPackageGroup("fruits", "apple");
    assertContainsEvent("invalid package name ':carrot'");
  }

  @Test
  public void testAllBeneathSpecificationWorks() throws Exception {
    scratch.file(
        "fruits/BUILD",
        "package_group(",
        "    name = 'maracuja',",
        "    packages = ['//tropics/...'],",
        ")");

    getPackageGroup("fruits", "maracuja");
  }

  @Test
  public void testNegative() throws Exception {
    scratch.file(
        "test/BUILD",
        "package_group(",
        "    name = 'packages',",
        "    packages = [",
        "        '//one',",
        "        '//two',",
        "        '-//three',",
        "        '-//four',",
        "    ],",
        ")");

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
        "package_group(",
        "    name = 'packages',",
        "    packages = [",
        "        '//pkg/...',",
        "        '-//pkg/one',",
        "    ],",
        ")");

    PackageGroup grp = getPackageGroup("test", "packages");
    assertThat(grp.contains(pkgId("pkg"))).isTrue();
    assertThat(grp.contains(pkgId("pkg/one"))).isFalse();
    assertThat(grp.contains(pkgId("pkg/one/two"))).isTrue();
  }

  @Test
  public void testNegative_subpackages() throws Exception {
    scratch.file(
        "test/BUILD",
        "package_group(",
        "    name = 'packages',",
        "    packages = [",
        "        '//pkg/...',",
        "        '-//pkg/one/...',",
        "    ],",
        ")");

    PackageGroup grp = getPackageGroup("test", "packages");
    assertThat(grp.contains(pkgId("pkg"))).isTrue();
    assertThat(grp.contains(pkgId("pkg/one"))).isFalse();
    assertThat(grp.contains(pkgId("pkg/one/two"))).isFalse();
  }

  @Test
  public void testNegative_everything() throws Exception {
    scratch.file(
        "test/BUILD",
        "package_group(",
        "    name = 'packages',",
        "    packages = [",
        "        '//pkg/one',",
        "        '-//...',",
        "    ],",
        ")");

    PackageGroup grp = getPackageGroup("test", "packages");
    assertThat(grp.contains(pkgId("pkg"))).isFalse();
    assertThat(grp.contains(pkgId("pkg/one"))).isFalse();
    assertThat(grp.contains(pkgId("pkg/one/two"))).isFalse();
  }

  @Test
  public void testEverythingSpecificationWorks() throws Exception {
    scratch.file(
        "fruits/BUILD", //
        "package_group(",
        "    name = 'mango',",
        "    packages = ['//...'],",
        ")");
    PackageGroup grp = getPackageGroup("fruits", "mango");

    // Assert that we're actually using the "everything" package specification, and assert that this
    // means we include packages from the main repo but also from other repos.
    assertThat(grp.getContainedPackages())
        .containsExactly(PackageSpecification.everything().toString());
    assertThat(grp.contains(pkgId("pkg"))).isTrue();
    assertThat(grp.contains(pkgId("somerepo", "pkg"))).isTrue();
  }

  @Test
  public void testDuplicatePackage() throws Exception {
    scratch.file(
        "test/BUILD",
        "package_group(",
        "    name = 'packages',",
        "    packages = [",
        "        '//one/two',",
        "        '//one/two',",
        "    ],",
        ")");

    PackageGroup grp = getPackageGroup("test", "packages");
    assertThat(grp.contains(pkgId("one/two"))).isTrue();
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
