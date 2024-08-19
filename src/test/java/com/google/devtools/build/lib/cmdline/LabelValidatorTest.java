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
import static com.google.common.truth.Truth.assertWithMessage;

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
    assertThat(LabelValidator.validatePackageName("foo")).isNull();
    assertThat(LabelValidator.validatePackageName("Foo")).isNull();
    assertThat(LabelValidator.validatePackageName("FOO")).isNull();
    assertThat(LabelValidator.validatePackageName("foO")).isNull();
    assertThat(LabelValidator.validatePackageName("foo-bar")).isNull();
    assertThat(LabelValidator.validatePackageName("Foo-Bar")).isNull();
    assertThat(LabelValidator.validatePackageName("FOO-BAR")).isNull();
    assertThat(LabelValidator.validatePackageName("bar.baz")).isNull();
    assertThat(LabelValidator.validatePackageName("a/..b")).isNull();
    assertThat(LabelValidator.validatePackageName("a/.b")).isNull();
    assertThat(LabelValidator.validatePackageName("a/b.")).isNull();
    assertThat(LabelValidator.validatePackageName("a/b..")).isNull();
    assertThat(LabelValidator.validatePackageName("a$( )/b..")).isNull();

    // These are in ascii code order.
    assertThat(LabelValidator.validatePackageName("foo!bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo\"bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo#bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo$bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo%bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo&bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo'bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo(bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo)bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo*bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo+bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo,bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo-bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo.bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo;bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo<bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo=bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo>bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo?bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo@bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo[bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo]bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo^bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo_bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo`bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo{bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo|bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo}bar")).isNull();
    assertThat(LabelValidator.validatePackageName("foo~bar")).isNull();

    // Bad:
    assertThat(LabelValidator.validatePackageName("/foo"))
        .isEqualTo("package names may not start with '/'");
    assertThat(LabelValidator.validatePackageName("foo/"))
        .isEqualTo("package names may not end with '/'");
    assertThat(LabelValidator.validatePackageName("foo:bar"))
        .isEqualTo(LabelValidator.PACKAGE_NAME_ERROR);

    assertThat(LabelValidator.validatePackageName("bar/../baz"))
        .isEqualTo(LabelValidator.PACKAGE_NAME_DOT_ERROR);
    assertThat(LabelValidator.validatePackageName("bar/.."))
        .isEqualTo(LabelValidator.PACKAGE_NAME_DOT_ERROR);
    assertThat(LabelValidator.validatePackageName("../bar"))
        .isEqualTo(LabelValidator.PACKAGE_NAME_DOT_ERROR);
    assertThat(LabelValidator.validatePackageName("bar/..."))
        .isEqualTo(LabelValidator.PACKAGE_NAME_DOT_ERROR);

    assertThat(LabelValidator.validatePackageName("bar/./baz"))
        .isEqualTo(LabelValidator.PACKAGE_NAME_DOT_ERROR);
    assertThat(LabelValidator.validatePackageName("bar/."))
        .isEqualTo(LabelValidator.PACKAGE_NAME_DOT_ERROR);
    assertThat(LabelValidator.validatePackageName("./bar"))
        .isEqualTo(LabelValidator.PACKAGE_NAME_DOT_ERROR);
  }

  @Test
  public void testValidateTargetName() throws Exception {
    assertThat(LabelValidator.validateTargetName("foo")).isNull();
    assertThat(LabelValidator.validateTargetName("foo!bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo\"bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo#bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo$bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo%bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo&bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo'bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo(bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo)bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo*bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo+bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo,bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo-bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo.bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo+bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo;bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo<bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo=bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo>bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo?bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo[bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo]bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo^bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo_bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo`bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo{bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo|bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo}bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo~bar")).isNull();

    assertThat(LabelValidator.validateTargetName("foo/bar")).isNull();
    assertThat(LabelValidator.validateTargetName("foo@bar")).isNull();
    assertThat(LabelValidator.validateTargetName("bar:baz")).isNull();
    assertThat(LabelValidator.validateTargetName("bar:")).isNull();

    assertThat(LabelValidator.validateTargetName("foo/"))
        .isEqualTo("target names may not end with '/'");
    assertThat(LabelValidator.validateTargetName(":foo"))
        .isEqualTo("target names may not start with ':'");
  }

  @Test
  public void testValidateAbsoluteLabel() throws Exception {
    PackageAndTarget emptyPackage = new PackageAndTarget("", "bar");
    assertThat(LabelValidator.validateAbsoluteLabel("//:bar")).isEqualTo(emptyPackage);
    assertThat(LabelValidator.validateAbsoluteLabel("@repo//:bar")).isEqualTo(emptyPackage);
    assertThat(LabelValidator.validateAbsoluteLabel("@repo//foo:bar"))
        .isEqualTo(new PackageAndTarget("foo", "bar"));
    assertThat(LabelValidator.validateAbsoluteLabel("@//foo:bar"))
        .isEqualTo(new PackageAndTarget("foo", "bar"));
    emptyPackage = new PackageAndTarget("", "b$() ar");
    assertThat(LabelValidator.validateAbsoluteLabel("//:b$() ar")).isEqualTo(emptyPackage);
    assertThat(LabelValidator.validateAbsoluteLabel("@repo//:b$() ar")).isEqualTo(emptyPackage);
    assertThat(LabelValidator.validateAbsoluteLabel("@repo//f$( )oo:b$() ar"))
        .isEqualTo(new PackageAndTarget("f$( )oo", "b$() ar"));
    assertThat(LabelValidator.validateAbsoluteLabel("@//f$( )oo:b$() ar"))
        .isEqualTo(new PackageAndTarget("f$( )oo", "b$() ar"));
    assertThat(LabelValidator.validateAbsoluteLabel("//f@oo"))
        .isEqualTo(new PackageAndTarget("f@oo", "f@oo"));
    assertThat(LabelValidator.validateAbsoluteLabel("//@foo"))
        .isEqualTo(new PackageAndTarget("@foo", "@foo"));
    assertThat(LabelValidator.validateAbsoluteLabel("//@foo:@bar"))
        .isEqualTo(new PackageAndTarget("@foo", "@bar"));
    assertThat(LabelValidator.validateAbsoluteLabel("@repo//f$( )oo:b$() :ar"))
        .isEqualTo(new PackageAndTarget("f$( )oo", "b$() :ar"));
  }

  @Test
  public void testPackageAndTargetHashCode_distinctButEqualObjects() {
    PackageAndTarget fooTarget1 = newFooTarget();
    PackageAndTarget fooTarget2 = newFooTarget();
    assertThat(fooTarget2).isNotSameInstanceAs(fooTarget1);
    assertWithMessage("Should have same hash code")
        .that(fooTarget1.hashCode())
        .isEqualTo(fooTarget2.hashCode());
  }

  @Test
  public void testPackageAndTargetEquals_distinctButEqualObjects() {
    PackageAndTarget fooTarget1 = newFooTarget();
    PackageAndTarget fooTarget2 = newFooTarget();
    assertThat(fooTarget2).isNotSameInstanceAs(fooTarget1);
    assertWithMessage("Should be equal").that(fooTarget1).isEqualTo(fooTarget2);
  }

  @Test
  public void testPackageAndTargetEquals_unequalObjects() {
    assertWithMessage("should be unequal").that(newFooTarget().equals(newBarTarget())).isFalse();
  }

  @Test
  public void testPackageAndTargetToString() {
    assertThat(newFooTarget().toString()).isEqualTo("//foo:foo");
    assertThat(newBarTarget().toString()).isEqualTo("//bar:bar");
  }

  @Test
  public void testSlashlessLabel_infersTargetNameFromRepoName() throws Exception {
    assertThat(LabelValidator.parseAbsoluteLabel("@foo").toString()).isEqualTo("//:foo");
  }
}
