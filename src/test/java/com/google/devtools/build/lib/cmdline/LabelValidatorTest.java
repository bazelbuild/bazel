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

package com.google.devtools.build.lib.cmdline;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.cmdline.LabelValidator.PackageAndTarget;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link LabelValidator}.
 */
@RunWith(JUnit4.class)
public class LabelValidatorTest {

  private PackageAndTarget newFooTarget() {
    return new PackageAndTarget("foo", "foo");
  }

  private PackageAndTarget newBarTarget() {
    return new PackageAndTarget("bar", "bar");
  }

  @Test
  public void testValidatePackageName() throws Exception {
    // OK:
    assertNull(LabelValidator.validatePackageName("foo"));
    assertNull(LabelValidator.validatePackageName("Foo"));
    assertNull(LabelValidator.validatePackageName("FOO"));
    assertNull(LabelValidator.validatePackageName("foO"));
    assertNull(LabelValidator.validatePackageName("foo-bar"));
    assertNull(LabelValidator.validatePackageName("Foo-Bar"));
    assertNull(LabelValidator.validatePackageName("FOO-BAR"));
    assertNull(LabelValidator.validatePackageName("bar.baz"));
    assertNull(LabelValidator.validatePackageName("a/..b"));
    assertNull(LabelValidator.validatePackageName("a/.b"));
    assertNull(LabelValidator.validatePackageName("a/b."));
    assertNull(LabelValidator.validatePackageName("a/b.."));

    // Bad:
    assertEquals(
        "package names may not start with '/'", LabelValidator.validatePackageName("/foo"));
    assertEquals("package names may not end with '/'", LabelValidator.validatePackageName("foo/"));
    assertEquals(LabelValidator.PACKAGE_NAME_ERROR, LabelValidator.validatePackageName("bar baz"));
    assertEquals(LabelValidator.PACKAGE_NAME_ERROR, LabelValidator.validatePackageName("foo:bar"));
    assertEquals(
        LabelValidator.PACKAGE_NAME_ERROR, LabelValidator.validatePackageName("baz@12345"));
    assertEquals(LabelValidator.PACKAGE_NAME_ERROR, LabelValidator.validatePackageName("baz(foo)"));
    assertEquals(LabelValidator.PACKAGE_NAME_ERROR, LabelValidator.validatePackageName("bazfoo)"));

    assertEquals(
        LabelValidator.PACKAGE_NAME_DOT_ERROR, LabelValidator.validatePackageName("bar/../baz"));
    assertEquals(
        LabelValidator.PACKAGE_NAME_DOT_ERROR, LabelValidator.validatePackageName("bar/.."));
    assertEquals(
        LabelValidator.PACKAGE_NAME_DOT_ERROR, LabelValidator.validatePackageName("../bar"));
    assertEquals(
        LabelValidator.PACKAGE_NAME_DOT_ERROR, LabelValidator.validatePackageName("bar/..."));

    assertEquals(
        LabelValidator.PACKAGE_NAME_DOT_ERROR, LabelValidator.validatePackageName("bar/./baz"));
    assertEquals(
        LabelValidator.PACKAGE_NAME_DOT_ERROR, LabelValidator.validatePackageName("bar/."));
    assertEquals(
        LabelValidator.PACKAGE_NAME_DOT_ERROR, LabelValidator.validatePackageName("./bar"));
  }

  @Test
  public void testValidateTargetName() throws Exception {

    assertNull(LabelValidator.validateTargetName("foo"));
    assertNull(LabelValidator.validateTargetName("foo+bar"));
    assertNull(LabelValidator.validateTargetName("foo_bar"));
    assertNull(LabelValidator.validateTargetName("foo=bar"));
    assertNull(LabelValidator.validateTargetName("foo-bar"));
    assertNull(LabelValidator.validateTargetName("foo.bar"));
    assertNull(LabelValidator.validateTargetName("foo@bar"));
    assertNull(LabelValidator.validateTargetName("foo~bar"));
    assertNull(LabelValidator.validateTargetName("foo#bar"));

    assertEquals("target names may not end with '/'",
                 LabelValidator.validateTargetName("foo/"));
    assertEquals("target names may not contain ' '",
                 LabelValidator.validateTargetName("bar baz"));
    assertEquals("target names may not contain ':'",
                 LabelValidator.validateTargetName("bar:baz"));
    assertEquals("target names may not contain ':'",
                 LabelValidator.validateTargetName("bar:"));
    assertEquals("target names may not contain '&'",
                 LabelValidator.validateTargetName("bar&"));
    assertEquals("target names may not contain '$'",
                 LabelValidator.validateTargetName("baz$a"));
    assertEquals("target names may not contain '('",
                 LabelValidator.validateTargetName("baz(foo)"));
    assertEquals("target names may not contain ')'",
                 LabelValidator.validateTargetName("bazfoo)"));
  }

  @Test
  public void testValidateAbsoluteLabel() throws Exception {
    PackageAndTarget emptyPackage = new PackageAndTarget("", "bar");
    assertEquals(emptyPackage, LabelValidator.validateAbsoluteLabel("//:bar"));
    assertEquals(emptyPackage, LabelValidator.validateAbsoluteLabel("@repo//:bar"));
    assertEquals(new PackageAndTarget("foo", "bar"),
        LabelValidator.validateAbsoluteLabel("@repo//foo:bar"));
    assertEquals(new PackageAndTarget("foo", "bar"),
        LabelValidator.validateAbsoluteLabel("@//foo:bar"));

    try {
      LabelValidator.validateAbsoluteLabel("@foo");
      fail("Should not have been able to validate @foo");
    } catch (LabelValidator.BadLabelException expected) {
      assertThat(expected.getMessage()).contains("invalid fully-qualified label");
    }
  }

  @Test
  public void testPackageAndTargetHashCode_distinctButEqualObjects() {
    PackageAndTarget fooTarget1 = newFooTarget();
    PackageAndTarget fooTarget2 = newFooTarget();
    assertNotSame(fooTarget1, fooTarget2);
    assertEquals("Should have same hash code", fooTarget2.hashCode(), fooTarget1.hashCode());
  }

  @Test
  public void testPackageAndTargetEquals_distinctButEqualObjects() {
    PackageAndTarget fooTarget1 = newFooTarget();
    PackageAndTarget fooTarget2 = newFooTarget();
    assertNotSame(fooTarget1, fooTarget2);
    assertEquals("Should be equal", fooTarget2, fooTarget1);
  }

  @Test
  public void testPackageAndTargetEquals_unequalObjects() {
    assertFalse("should be unequal", newFooTarget().equals(newBarTarget()));
  }

  @Test
  public void testPackageAndTargetToString() {
    assertEquals("//foo:foo", newFooTarget().toString());
    assertEquals("//bar:bar", newBarTarget().toString());
  }
}
