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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.packages.util.PackageFactoryApparatus;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for PackageGroup.
 */
@RunWith(JUnit4.class)
public class PackageGroupTest {
  private Scratch scratch = new Scratch("/workspace");
  private EventCollectionApparatus events = new EventCollectionApparatus();
  private PackageFactoryApparatus packages = new PackageFactoryApparatus(events.reporter());

  @Test
  public void testDoesNotFailHorribly() throws Exception {
    scratch.file("fruits/BUILD", "package_group(name = 'apple', packages = ['//random'])");

    getPackageGroup("fruits", "apple");
  }

  // Regression test for: "Package group with empty name causes Blaze exception"
  @Test
  public void testEmptyPackageGroupNameDoesNotThrow() throws Exception {
    scratch.file("strawberry/BUILD", "package_group(name = '', packages=[])");

    events.setFailFast(false);
    getPackage("strawberry");
    events.assertContainsError("package group has invalid name");
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
    assertTrue(grp.contains(getPackage("vegetables")));
    assertFalse(grp.contains(getPackage("fruits/vegetables")));
  }

  @Test
  public void testPackagesWithoutDoubleSlashDoNotWork() throws Exception {
    scratch.file(
        "fruits/BUILD",
        "package_group(name = 'apple',",
        "              packages = ['vegetables'])");

    scratch.file("vegetables/BUILD");
    scratch.file("fruits/vegetables/BUILD");

    events.setFailFast(false);
    getPackageGroup("fruits", "apple");
    events.assertContainsError("invalid package name 'vegetables'");
  }

  @Test
  public void testPackagesWithRepositoryDoNotWork() throws Exception {
    scratch.file(
        "fruits/BUILD",
        "package_group(name = 'banana',",
        "              packages = ['@veggies//:cucumber'])");

    events.setFailFast(false);
    getPackageGroup("fruits", "banana");
    events.assertContainsError("invalid package name '@veggies//:cucumber'");
  }

  @Test
  public void testAllPackagesInMainRepositoryDoesNotWork() throws Exception {
    scratch.file(
        "fruits/BUILD",
        "package_group(name = 'apple',",
        "              packages = ['@//...'])");

    events.setFailFast(false);
    getPackageGroup("fruits", "apple");
    events.assertContainsError("invalid package name '@//...'");
  }

  @Test
  public void testTargetNameAsPackageDoesNotWork1() throws Exception {
    scratch.file(
        "fruits/BUILD",
        "package_group(name = 'apple',",
        "              packages = ['//vegetables:carrot'])");

    scratch.file("vegetables/BUILD");
    scratch.file("fruits/vegetables/BUILD");

    events.setFailFast(false);
    getPackageGroup("fruits", "apple");
    events.assertContainsError("invalid package name '//vegetables:carrot'");
  }

  @Test
  public void testTargetNameAsPackageDoesNotWork2() throws Exception {
    scratch.file(
        "fruits/BUILD", "package_group(name = 'apple',", "              packages = [':carrot'])");

    scratch.file("vegetables/BUILD");
    scratch.file("fruits/vegetables/BUILD");

    events.setFailFast(false);
    getPackageGroup("fruits", "apple");
    events.assertContainsError("invalid package name ':carrot'");
  }

  @Test
  public void testAllBeneathSpecificationWorks() throws Exception {
    scratch.file(
        "fruits/BUILD",
        "package_group(name = 'maracuja',",
        "              packages = ['//tropics/...'])");

    getPackageGroup("fruits", "maracuja");
  }

  @Test
  public void testEverythingSpecificationWorks() throws Exception {
    scratch.file("fruits/BUILD", "package_group(name = 'mango', packages = ['//...'])");
    PackageGroup packageGroup = getPackageGroup("fruits", "mango");
    assertThat(packageGroup.getPackageSpecifications())
        .containsExactlyElementsIn(ImmutableList.of(PackageSpecification.EVERYTHING));
  }

  private Package getPackage(String packageName) throws Exception {
    PathFragment buildFileFragment = new PathFragment(packageName).getRelative("BUILD");

    Path buildFile = scratch.resolve(buildFileFragment.getPathString());
    return packages.createPackage(packageName, buildFile);
  }

  private PackageGroup getPackageGroup(String pkg, String name) throws Exception {
    return (PackageGroup) getPackage(pkg).getTarget(name);
  }
}
