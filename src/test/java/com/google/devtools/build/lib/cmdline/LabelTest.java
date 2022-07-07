// Copyright 2015 The Bazel Authors. All Rights Reserved.
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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.regex.Pattern;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link Label}.
 */
@RunWith(JUnit4.class)
public class LabelTest {

  private static final String INVALID_TARGET_NAME = "invalid target name";
  private static final String INVALID_PACKAGE_NAME = "invalid package name";

  @Test
  public void testAbsolute() throws Exception {
    {
      Label l = Label.parseCanonical("//foo/bar:baz");
      assertThat(l.getPackageName()).isEqualTo("foo/bar");
      assertThat(l.getName()).isEqualTo("baz");
    }
    {
      Label l = Label.parseCanonical("//foo/bar");
      assertThat(l.getPackageName()).isEqualTo("foo/bar");
      assertThat(l.getName()).isEqualTo("bar");
    }
    {
      Label l = Label.parseCanonical("//:bar");
      assertThat(l.getPackageName()).isEmpty();
      assertThat(l.getName()).isEqualTo("bar");
    }
    {
      Label l = Label.parseCanonical("@foo");
      assertThat(l.getRepository().getNameWithAt()).isEqualTo("@foo");
      assertThat(l.getPackageName()).isEmpty();
      assertThat(l.getName()).isEqualTo("foo");
    }
    {
      Label l = Label.parseCanonical("//@foo");
      assertThat(l.getRepository().getNameWithAt()).isEqualTo("@");
      assertThat(l.getPackageName()).isEqualTo("@foo");
      assertThat(l.getName()).isEqualTo("@foo");
    }
    {
      Label l = Label.parseCanonical("//xyz/@foo:abc");
      assertThat(l.getRepository().getNameWithAt()).isEqualTo("@");
      assertThat(l.getPackageName()).isEqualTo("xyz/@foo");
      assertThat(l.getName()).isEqualTo("abc");
    }
  }

  @Test
  public void testGetRelativeWithAbsoluteLabel() throws Exception {
    Label base = Label.parseCanonical("//foo/bar:baz");
    Label l = base.getRelativeWithRemapping("//p1/p2:target", ImmutableMap.of());
    assertThat(l.getPackageName()).isEqualTo("p1/p2");
    assertThat(l.getName()).isEqualTo("target");
  }

  @Test
  public void testGetRelativeWithRelativeLabel() throws Exception {
    Label base = Label.parseCanonical("//foo/bar:baz");
    Label l = base.getRelativeWithRemapping(":quux", ImmutableMap.of());
    assertThat(l.getPackageName()).isEqualTo("foo/bar");
    assertThat(l.getName()).isEqualTo("quux");
  }

  @Test
  public void testGetRelativeWithIllegalLabel() throws Exception {
    Label base = Label.parseCanonical("//foo/bar:baz");
    assertThrows(
        LabelSyntaxException.class,
        () -> base.getRelativeWithRemapping("/p1/p2:target", ImmutableMap.of()));
    assertThrows(
        LabelSyntaxException.class,
        () -> base.getRelativeWithRemapping("quux:", ImmutableMap.of()));
    assertThrows(
        LabelSyntaxException.class, () -> base.getRelativeWithRemapping(":", ImmutableMap.of()));
    assertThrows(
        LabelSyntaxException.class, () -> base.getRelativeWithRemapping("::", ImmutableMap.of()));
  }

  @Test
  public void testGetRelativeWithDifferentRepo() throws Exception {
    PackageIdentifier packageId = PackageIdentifier.create("repo", PathFragment.create("foo"));
    Label base = Label.create(packageId, "bar");

    Label relative = base.getRelativeWithRemapping("@remote//x:y", ImmutableMap.of());

    assertThat(relative.getRepository()).isEqualTo(RepositoryName.create("remote"));
    assertThat(relative.getPackageFragment()).isEqualTo(PathFragment.create("x"));
    assertThat(relative.getName()).isEqualTo("y");
  }

  @Test
  public void testGetRelativeWithoutRemappingBaseLabel() throws Exception {
    PackageIdentifier packageId = PackageIdentifier.create("a", PathFragment.create("foo"));
    Label base = Label.create(packageId, "bar");
    ImmutableMap<String, RepositoryName> repoMapping =
        ImmutableMap.of("a", RepositoryName.create("b"));
    Label relative = base.getRelativeWithRemapping(":y", repoMapping);

    // getRelative should only remap repositories passed in the string arg and not
    // make changes to existing Labels
    Label actual = Label.parseAbsoluteUnchecked("@a//foo:y");
    assertThat(relative).isEqualTo(actual);
  }

  @Test
  public void testGetRelativeWithDifferentRepoAndRemapping() throws Exception {
    PackageIdentifier packageId = PackageIdentifier.create("repo", PathFragment.create("foo"));
    Label base = Label.create(packageId, "bar");
    ImmutableMap<String, RepositoryName> repoMapping =
        ImmutableMap.of("a", RepositoryName.create("b"));
    Label relative = base.getRelativeWithRemapping("@a//x:y", repoMapping);

    Label actual = Label.parseAbsoluteUnchecked("@b//x:y");
    assertThat(relative).isEqualTo(actual);
  }

  @Test
  public void testGetRelativeWithRepoLocalAbsoluteLabel() throws Exception {
    PackageIdentifier packageId = PackageIdentifier.create("repo", PathFragment.create("foo"));
    Label base = Label.create(packageId, "bar");

    Label relative = base.getRelativeWithRemapping("//x:y", ImmutableMap.of());

    assertThat(relative.getRepository()).isEqualTo(packageId.getRepository());
    assertThat(relative.getPackageFragment()).isEqualTo(PathFragment.create("x"));
    assertThat(relative.getName()).isEqualTo("y");
  }

  @Test
  public void testGetRelativeWithLocalRepoRelativeLabel() throws Exception {
    PackageIdentifier packageId = PackageIdentifier.create("repo", PathFragment.create("foo"));
    Label base = Label.create(packageId, "bar");

    Label relative = base.getRelativeWithRemapping(":y", ImmutableMap.of());

    assertThat(relative.getRepository()).isEqualTo(packageId.getRepository());
    assertThat(relative.getPackageFragment()).isEqualTo(PathFragment.create("foo"));
    assertThat(relative.getName()).isEqualTo("y");
  }

  @Test
  public void testGetRelativeWithRepoAndReservedPackage() throws Exception {
    PackageIdentifier packageId = PackageIdentifier.create("repo", PathFragment.create("foo"));
    Label base = Label.create(packageId, "bar");

    Label relative =
        base.getRelativeWithRemapping(
            "//conditions:default",
            RepositoryMapping.create(ImmutableMap.of(), RepositoryName.create("repo")));

    PackageIdentifier expected = PackageIdentifier.createInMainRepo("conditions");
    assertThat(relative.getRepository()).isEqualTo(expected.getRepository());
    assertThat(relative.getPackageFragment()).isEqualTo(expected.getPackageFragment());
    assertThat(relative.getName()).isEqualTo("default");
  }

  @Test
  public void testGetRelativeWithRemoteRepoToDefaultRepo() throws Exception {
    PackageIdentifier packageId = PackageIdentifier.create("repo", PathFragment.create("foo"));
    Label base = Label.create(packageId, "bar");

    Label relative = base.getRelativeWithRemapping("@//x:y", ImmutableMap.of());

    assertThat(relative.getRepository()).isEqualTo(RepositoryName.MAIN);
    assertThat(relative.getPackageFragment()).isEqualTo(PathFragment.create("x"));
    assertThat(relative.getName()).isEqualTo("y");
  }

  @Test
  public void testFactory() throws Exception {
    Label l = Label.create("foo/bar", "quux");
    assertThat(l.getPackageName()).isEqualTo("foo/bar");
    assertThat(l.getName()).isEqualTo("quux");
  }

  @Test
  public void testIdentities() throws Exception {

    Label l1 = Label.parseCanonical("//foo/bar:baz");
    Label l2 = Label.parseCanonical("//foo/bar:baz");
    Label l3 = Label.parseCanonical("//foo/bar:quux");

    new EqualsTester()
        .addEqualityGroup(l1, l2)
        .addEqualityGroup(l3)
        .testEquals();
  }

  @Test
  public void testToString() throws Exception {
    {
      String s = "@//foo/bar:baz";
      Label l = Label.parseCanonical(s);
      assertThat(l.toString()).isEqualTo("//foo/bar:baz");
    }
    {
      Label l = Label.parseCanonical("//foo/bar");
      assertThat(l.toString()).isEqualTo("//foo/bar:bar");
    }
    {
      Label l = Label.parseCanonical("@foo");
      assertThat(l.toString()).isEqualTo("@foo//:foo");
    }
  }

  @Test
  public void testToShorthandString() throws Exception {
    {
      Label l = Label.parseCanonical("//bar/baz:baz");
      assertThat(l.toShorthandString()).isEqualTo("//bar/baz");
    }
    {
      Label l = Label.parseCanonical("//bar/baz:bat");
      assertThat(l.toShorthandString()).isEqualTo("//bar/baz:bat");
    }
    {
      Label l = Label.parseCanonical("@foo//bar/baz:baz");
      assertThat(l.toShorthandString()).isEqualTo("@foo//bar/baz");
    }
    {
      Label l = Label.parseCanonical("@foo//bar/baz:bat");
      assertThat(l.toShorthandString()).isEqualTo("@foo//bar/baz:bat");
    }
  }

  @Test
  public void testDotDot() throws Exception {
    Label.parseCanonical("//foo/bar:baz..gif");
  }

  /**
   * Asserts that creating a label throws a SyntaxException.
   * @param label the label to create.
   */
  private static void assertSyntaxError(String expectedError, String label) {
    LabelSyntaxException e =
        assertThrows(
            "Label '" + label + "' did not contain a syntax error, but was expected to",
            LabelSyntaxException.class,
            () -> Label.parseCanonical(label));
    assertThat(e).hasMessageThat().containsMatch(Pattern.quote(expectedError));
  }

  @Test
  public void testBadCharacters() throws Exception {
    assertSyntaxError("target names may not contain ':'",
                      "//foo:bar:baz");
    assertSyntaxError("target names may not contain ':'",
                      "//foo:bar:");
    assertSyntaxError("target names may not contain ':'",
                      "//foo/bar::");
  }

  @Test
  public void testUplevelReferences() throws Exception {
    assertSyntaxError(INVALID_PACKAGE_NAME, "//foo/bar/..:baz");
    assertSyntaxError(INVALID_PACKAGE_NAME, "//foo/../baz:baz");
    assertSyntaxError(INVALID_PACKAGE_NAME, "//../bar/baz:baz");
    assertSyntaxError(INVALID_PACKAGE_NAME, "//..:foo");
    assertSyntaxError(INVALID_TARGET_NAME, "//foo:bar/../baz");
    assertSyntaxError(INVALID_TARGET_NAME, "//foo:../bar/baz");
    assertSyntaxError(INVALID_TARGET_NAME, "//foo:bar/baz/..");
    assertSyntaxError(INVALID_TARGET_NAME, "//foo:..");
  }

  @Test
  public void testDotAsAPathSegment() throws Exception {
    assertSyntaxError(INVALID_PACKAGE_NAME, "//foo/bar/.:baz");
    assertSyntaxError(INVALID_PACKAGE_NAME, "//foo/./baz:baz");
    assertSyntaxError(INVALID_PACKAGE_NAME, "//./bar/baz:baz");
    assertSyntaxError(INVALID_TARGET_NAME, "//foo:bar/./baz");
    assertSyntaxError(INVALID_TARGET_NAME, "//foo:./bar/baz");
    // TODO(bazel-team): enable when we have removed the "Workaround" in Label
    // that rewrites broken Labels by removing the trailing '.'
    //assertSyntaxError(INVALID_PACKAGE_NAME,
    //                  "//foo:bar/baz/.");
    //assertSyntaxError(INVALID_PACKAGE_NAME,
    //                  "//foo:.");
  }

  @Test
  public void testTrailingDotSegment() throws Exception {
    assertThat(Label.parseCanonical("//foo:dir")).isEqualTo(Label.parseCanonical("//foo:dir/."));
  }

  @Test
  public void testSomeOtherBadLabels() throws Exception {
    assertSyntaxError("package names may not end with '/'",
                      "//foo/:bar");
    assertSyntaxError("package names may not start with '/'", "///p:foo");
    assertSyntaxError("package names may not contain '//' path separators",
                      "//a//b:foo");
  }

  @Test
  public void testSomeGoodLabels() throws Exception {
    Label.parseCanonical("//foo:..bar");
    Label.parseCanonical("//Foo:..bar");
    Label.parseCanonical("//-Foo:..bar");
    Label.parseCanonical("//00:..bar");
    Label.parseCanonical("//package:foo+bar");
    Label.parseCanonical("//package:foo_bar");
    Label.parseCanonical("//package:foo=bar");
    Label.parseCanonical("//package:foo-bar");
    Label.parseCanonical("//package:foo.bar");
    Label.parseCanonical("//package:foo@bar");
    Label.parseCanonical("//package:foo~bar");
    Label.parseCanonical("//$( ):$( )");
  }

  @Test
  public void testDoubleSlashPathSeparator() throws Exception {
    assertSyntaxError("package names may not contain '//' path separators",
                      "//foo//bar:baz");
    assertSyntaxError("target names may not contain '//' path separator",
                      "//foo:bar//baz");
  }

  @Test
  public void testNonPrintableCharacters() throws Exception {
    assertSyntaxError(
      "target names may not contain non-printable characters: '\\x02'",
      "//foo:..\002bar");
  }

  /** Make sure that control characters - such as CR - are escaped on output. */
  @Test
  public void testInvalidLineEndings() throws Exception {
    assertSyntaxError("invalid target name '..bar\\r': "
        + "target names may not end with carriage returns", "//foo:..bar\r");
  }

  @Test
  public void testEmptyName() throws Exception {
    assertSyntaxError("invalid target name '': empty target name", "//foo/bar:");
  }

  @Test
  public void testRepoLabel() throws Exception {
    Label label = Label.parseCanonical("@foo//bar/baz:bat/boo");
    assertThat(label.toString()).isEqualTo("@foo//bar/baz:bat/boo");
  }

  @Test
  public void testNoRepo() throws Exception {
    Label label = Label.parseCanonical("//bar/baz:bat/boo");
    assertThat(label.toString()).isEqualTo("//bar/baz:bat/boo");
  }

  @Test
  public void testInvalidRepo() throws Exception {
    LabelSyntaxException e =
        assertThrows(
            LabelSyntaxException.class, () -> Label.parseCanonical("foo//bar/baz:bat/boo"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "invalid package name 'foo//bar/baz': package names may not contain '//' path"
                + " separators");
  }

  @Test
  public void testInvalidRepoWithColon() throws Exception {
    LabelSyntaxException e =
        assertThrows(LabelSyntaxException.class, () -> Label.parseCanonical("@foo:xyz"));
    assertThat(e)
        .hasMessageThat()
        .containsMatch("invalid repository name '@foo:xyz': repo names may contain only");
  }

  @Test
  public void testGetWorkspaceRoot() throws Exception {
    Label label = Label.parseCanonical("//bar/baz");
    assertThat(label.getWorkspaceRootForStarlarkOnly(StarlarkSemantics.DEFAULT)).isEmpty();
    label = Label.parseCanonical("@repo//bar/baz");
    assertThat(label.getWorkspaceRootForStarlarkOnly(StarlarkSemantics.DEFAULT))
        .isEqualTo("external/repo");
  }

  @Test
  public void testGetContainingDirectory() {
    assertThat(Label.getContainingDirectory(Label.parseAbsoluteUnchecked("//a:b")))
        .isEqualTo(PathFragment.create("a"));
    assertThat(Label.getContainingDirectory(Label.parseAbsoluteUnchecked("//a/b:c")))
        .isEqualTo(PathFragment.create("a/b"));
    assertThat(Label.getContainingDirectory(Label.parseAbsoluteUnchecked("//a:b/c")))
        .isEqualTo(PathFragment.create("a/b"));
    assertThat(Label.getContainingDirectory(Label.parseAbsoluteUnchecked("//a/b/c")))
        .isEqualTo(PathFragment.create("a/b/c"));
  }

  @Test
  public void testWorkspaceName() throws Exception {
    assertThat(Label.parseCanonical("@foo//bar:baz").getWorkspaceName()).isEqualTo("foo");
    assertThat(Label.parseCanonical("//bar:baz").getWorkspaceName()).isEmpty();
    assertThat(Label.parseCanonical("@//bar:baz").getWorkspaceName()).isEmpty();
  }

  @Test
  public void testStarlarkStrAndRepr() throws Exception {
    Label label = Label.parseCanonical("//x");
    assertThat(Starlark.str(label)).isEqualTo("//x:x");
    assertThat(Starlark.repr(label)).isEqualTo("Label(\"//x:x\")");
  }
}
