// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.cmdline.LabelValidator.PackageAndTarget;

import junit.framework.TestCase;

/**
 * Tests for {@link LabelValidator}.
 */
public class LabelValidatorTest extends TestCase {

  private static final String BAD_PACKAGE_CHARS =
      "package names may contain only A-Z, a-z, 0-9, '/', '-' and '_'";

  private PackageAndTarget newFooTarget() {
    return new PackageAndTarget("foo", "foo");
  }

  private PackageAndTarget newBarTarget() {
    return new PackageAndTarget("bar", "bar");
  }

  public void testValidatePackageName() throws Exception {
    // OK:
    assertNull(LabelValidator.validatePackageName("foo"));
    assertNull(LabelValidator.validatePackageName("Foo"));
    assertNull(LabelValidator.validatePackageName("FOO"));
    assertNull(LabelValidator.validatePackageName("foO"));
    assertNull(LabelValidator.validatePackageName("foo-bar"));
    assertNull(LabelValidator.validatePackageName("Foo-Bar"));
    assertNull(LabelValidator.validatePackageName("FOO-BAR"));

    // Bad:
    assertEquals("package names may not start with '/'",
        LabelValidator.validatePackageName("/foo"));
    assertEquals("package names may not end with '/'",
                 LabelValidator.validatePackageName("foo/"));
    assertEquals(BAD_PACKAGE_CHARS,
                 LabelValidator.validatePackageName("bar baz"));
    assertEquals(BAD_PACKAGE_CHARS,
                 LabelValidator.validatePackageName("foo:bar"));
    assertEquals(BAD_PACKAGE_CHARS,
                 LabelValidator.validatePackageName("baz@12345"));
    assertEquals(BAD_PACKAGE_CHARS,
                 LabelValidator.validatePackageName("baz(foo)"));
    assertEquals(BAD_PACKAGE_CHARS,
                 LabelValidator.validatePackageName("bazfoo)"));
  }

  public void testValidateTargetName() throws Exception {

    assertNull(LabelValidator.validateTargetName("foo"));
    assertNull(LabelValidator.validateTargetName("foo+bar"));
    assertNull(LabelValidator.validateTargetName("foo_bar"));
    assertNull(LabelValidator.validateTargetName("foo=bar"));
    assertNull(LabelValidator.validateTargetName("foo-bar"));
    assertNull(LabelValidator.validateTargetName("foo.bar"));
    assertNull(LabelValidator.validateTargetName("foo@bar"));
    assertNull(LabelValidator.validateTargetName("foo~bar"));

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

  public void testPackageAndTargetHashCode_distinctButEqualObjects() {
    PackageAndTarget fooTarget1 = newFooTarget();
    PackageAndTarget fooTarget2 = newFooTarget();
    assertNotSame(fooTarget1, fooTarget2);
    assertEquals("Should have same hash code", fooTarget2.hashCode(), fooTarget1.hashCode());
  }

  public void testPackageAndTargetEquals_distinctButEqualObjects() {
    PackageAndTarget fooTarget1 = newFooTarget();
    PackageAndTarget fooTarget2 = newFooTarget();
    assertNotSame(fooTarget1, fooTarget2);
    assertEquals("Should be equal", fooTarget2, fooTarget1);
  }

  public void testPackageAndTargetEquals_unequalObjects() {
    assertFalse("should be unequal", newFooTarget().equals(newBarTarget()));
  }

  public void testPackageAndTargetToString() {
    assertEquals("//foo:foo", newFooTarget().toString());
    assertEquals("//bar:bar", newBarTarget().toString());
  }
}
