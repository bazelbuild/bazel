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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for PackageGroup. */
@RunWith(JUnit4.class)
public class PackageGroupTest extends PackageLoadingTestCase {

  @Test
  public void testDoesNotFailHorribly() throws Exception {
    scratch.file("fruits/BUILD", "package_group(name = 'apple', packages = ['//random'])");

    getPackageGroup("fruits", "apple");
  }

  // Regression test for: "Package group with empty name causes Blaze exception"
  @Test
  public void testEmptyPackageGroupNameDoesNotThrow() throws Exception {
    scratch.file("strawberry/BUILD", "package_group(name = '', packages=[])");

    reporter.removeHandler(failFastHandler);
    getPackage("strawberry");
    assertContainsEvent("package group has invalid name");
  }

  @Test
  public void testAbsolutePackagesWork() throws Exception {
    scratch.file(
        "fruits/BUILD",
        "package_group(name = 'apple',",
        "              packages = ['//vegetables'])");

    scratch.file("vegetables/BUILD");
    scratch.file("fruits/vegetables/BUILD");

    PackageGroup grp = getPackageGroup("fruits", "apple");
    assertThat(grp.contains(getPackage("vegetables"))).isTrue();
    assertThat(grp.contains(getPackage("fruits/vegetables"))).isFalse();
  }

  @Test
  public void testPackagesWithoutDoubleSlashDoNotWork() throws Exception {
    scratch.file(
        "fruits/BUILD",
        "package_group(name = 'apple',",
        "              packages = ['vegetables'])");

    scratch.file("vegetables/BUILD");
    scratch.file("fruits/vegetables/BUILD");

    reporter.removeHandler(failFastHandler);
    getPackageGroup("fruits", "apple");
    assertContainsEvent("invalid package name 'vegetables'");
  }

  @Test
  public void testPackagesWithRepositoryDoNotWork() throws Exception {
    scratch.file(
        "fruits/BUILD", "package_group(name = 'banana', packages = ['@veggies//:cucumber'])");

    reporter.removeHandler(failFastHandler);
    getPackageGroup("fruits", "banana");
    assertContainsEvent("invalid package name '@veggies//:cucumber'");
  }

  @Test
  public void testAllPackagesInMainRepositoryDoesNotWork() throws Exception {
    scratch.file("fruits/BUILD", "package_group(name = 'apple', packages = ['@//...'])");

    reporter.removeHandler(failFastHandler);
    getPackageGroup("fruits", "apple");
    assertContainsEvent("invalid package name '@//...'");
  }

  @Test
  public void testTargetNameAsPackageDoesNotWork1() throws Exception {
    scratch.file(
        "fruits/BUILD",
        "package_group(name = 'apple',",
        "              packages = ['//vegetables:carrot'])");

    scratch.file("vegetables/BUILD");
    scratch.file("fruits/vegetables/BUILD");

    reporter.removeHandler(failFastHandler);
    getPackageGroup("fruits", "apple");
    assertContainsEvent("invalid package name '//vegetables:carrot'");
  }

  @Test
  public void testTargetNameAsPackageDoesNotWork2() throws Exception {
    scratch.file("fruits/BUILD", "package_group(name = 'apple', packages = [':carrot'])");
    scratch.file("vegetables/BUILD");
    scratch.file("fruits/vegetables/BUILD");

    reporter.removeHandler(failFastHandler);
    getPackageGroup("fruits", "apple");
    assertContainsEvent("invalid package name ':carrot'");
  }

  @Test
  public void testAllBeneathSpecificationWorks() throws Exception {
    scratch.file("fruits/BUILD", "package_group(name = 'maracuja', packages = ['//tropics/...'])");

    getPackageGroup("fruits", "maracuja");
  }

  @Test
  public void testNegative() throws Exception {
    scratch.file("one/BUILD");
    scratch.file("two/BUILD");
    scratch.file("three/BUILD");
    scratch.file("four/BUILD");
    scratch.file(
        "test/BUILD",
        "package_group(",
        "  name = 'packages',",
        "    packages = [",
        "        '//one',",
        "        '//two',",
        "        '-//three',",
        "        '-//four',",
        "    ],",
        ")");

    PackageGroup grp = getPackageGroup("test", "packages");
    assertThat(grp.contains(getPackage("one"))).isTrue();
    assertThat(grp.contains(getPackage("two"))).isTrue();
    assertThat(grp.contains(getPackage("three"))).isFalse();
    assertThat(grp.contains(getPackage("four"))).isFalse();
  }

  @Test
  public void testNegative_noSubpackages() throws Exception {
    scratch.file("pkg/BUILD");
    scratch.file("pkg/one/BUILD");
    scratch.file("pkg/one/two/BUILD");
    scratch.file(
        "test/BUILD",
        "package_group(",
        "  name = 'packages',",
        "    packages = [",
        "        '//pkg/...',",
        "        '-//pkg/one',",
        "    ],",
        ")");

    PackageGroup grp = getPackageGroup("test", "packages");
    assertThat(grp.contains(getPackage("pkg"))).isTrue();
    assertThat(grp.contains(getPackage("pkg/one"))).isFalse();
    assertThat(grp.contains(getPackage("pkg/one/two"))).isTrue();
  }

  @Test
  public void testNegative_subpackages() throws Exception {
    scratch.file("pkg/BUILD");
    scratch.file("pkg/one/BUILD");
    scratch.file("pkg/one/two/BUILD");
    scratch.file(
        "test/BUILD",
        "package_group(",
        "  name = 'packages',",
        "    packages = [",
        "        '//pkg/...',",
        "        '-//pkg/one/...',",
        "    ],",
        ")");

    PackageGroup grp = getPackageGroup("test", "packages");
    assertThat(grp.contains(getPackage("pkg"))).isTrue();
    assertThat(grp.contains(getPackage("pkg/one"))).isFalse();
    assertThat(grp.contains(getPackage("pkg/one/two"))).isFalse();
  }

  @Test
  public void testNegative_everything() throws Exception {
    scratch.file("pkg/BUILD");
    scratch.file("pkg/one/BUILD");
    scratch.file("pkg/one/two/BUILD");
    scratch.file(
        "test/BUILD",
        "package_group(",
        "  name = 'packages',",
        "    packages = [",
        "        '-//...',",
        "    ],",
        ")");

    PackageGroup grp = getPackageGroup("test", "packages");
    assertThat(grp.contains(getPackage("pkg"))).isFalse();
    assertThat(grp.contains(getPackage("pkg/one"))).isFalse();
    assertThat(grp.contains(getPackage("pkg/one/two"))).isFalse();
  }

  @Test
  public void testEverythingSpecificationWorks() throws Exception {
    scratch.file("fruits/BUILD", "package_group(name = 'mango', packages = ['//...'])");
    PackageGroup packageGroup = getPackageGroup("fruits", "mango");
    assertThat(
            packageGroup.getPackageSpecifications().containedPackages().collect(toImmutableList()))
        .containsExactly(PackageSpecification.everything().toString());
  }

  @Test
  public void testDuplicatePackage() throws Exception {
    scratch.file("pkg/BUILD");
    scratch.file("one/two/BUILD");

    scratch.file(
        "test/BUILD",
        "package_group(",
        "  name = 'packages',",
        "    packages = [",
        "        '//one/two',",
        "        '//one/two',",
        "    ],",
        ")");

    PackageGroup grp = getPackageGroup("test", "packages");
    assertThat(grp.contains(getPackage("one/two"))).isTrue();
  }

  private Package getPackage(String packageName) throws Exception {
    return getTarget("//" + packageName + ":BUILD").getPackage();
  }

  private PackageGroup getPackageGroup(String pkg, String name) throws Exception {
    return (PackageGroup) getTarget("//" + pkg + ":" + name);
  }
}
