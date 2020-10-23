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
import com.google.devtools.build.lib.testutil.TestUtils;
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
      Label l = Label.parseAbsolute("//foo/bar:baz", ImmutableMap.of());
      assertThat(l.getPackageName()).isEqualTo("foo/bar");
      assertThat(l.getName()).isEqualTo("baz");
    }
    {
      Label l = Label.parseAbsolute("//foo/bar", ImmutableMap.of());
      assertThat(l.getPackageName()).isEqualTo("foo/bar");
      assertThat(l.getName()).isEqualTo("bar");
    }
    {
      Label l = Label.parseAbsolute("//:bar", ImmutableMap.of());
      assertThat(l.getPackageName()).isEmpty();
      assertThat(l.getName()).isEqualTo("bar");
    }
    {
      Label l = Label.parseAbsolute("@foo", ImmutableMap.of());
      assertThat(l.getRepository().getName()).isEqualTo("@foo");
      assertThat(l.getPackageName()).isEmpty();
      assertThat(l.getName()).isEqualTo("foo");
    }
    {
      Label l = Label.parseAbsolute("//@foo", ImmutableMap.of());
      assertThat(l.getRepository().getName()).isEqualTo("@");
      assertThat(l.getPackageName()).isEqualTo("@foo");
      assertThat(l.getName()).isEqualTo("@foo");
    }
    {
      Label l = Label.parseAbsolute("//xyz/@foo:abc", ImmutableMap.of());
      assertThat(l.getRepository().getName()).isEqualTo("@");
      assertThat(l.getPackageName()).isEqualTo("xyz/@foo");
      assertThat(l.getName()).isEqualTo("abc");
    }
  }

  private static String parseCommandLine(String label, String prefix) throws LabelSyntaxException {
    return Label.parseCommandLineLabel(label, PathFragment.create(prefix)).toString();
  }

  @Test
  public void testLabelResolution() throws Exception {
    assertThat(parseCommandLine("//absolute:label", "")).isEqualTo("//absolute:label");
    assertThat(parseCommandLine("//absolute:label", "absolute")).isEqualTo("//absolute:label");
    assertThat(parseCommandLine(":label", "absolute")).isEqualTo("//absolute:label");
    assertThat(parseCommandLine("label", "absolute")).isEqualTo("//absolute:label");
    assertThat(parseCommandLine("absolute:label", "")).isEqualTo("//absolute:label");
    assertThat(parseCommandLine("path:label", "absolute")).isEqualTo("//absolute/path:label");
    assertThat(parseCommandLine("path:label/path", "absolute"))
        .isEqualTo("//absolute/path:label/path");
    assertThat(parseCommandLine("label/path", "absolute")).isEqualTo("//absolute:label/path");
  }

  @Test
  public void testLabelResolutionAbsolutePath() throws Exception {
    assertThrows(
        IllegalArgumentException.class, () -> parseCommandLine("//absolute:label", "/absolute"));
  }

  @Test
  public void testLabelResolutionBadSyntax() throws Exception {
    assertThrows(LabelSyntaxException.class, () -> parseCommandLine("//absolute:A+bad:syntax", ""));
  }

  @Test
  public void testGetRelativeWithAbsoluteLabel() throws Exception {
    Label base = Label.parseAbsolute("//foo/bar:baz", ImmutableMap.of());
    Label l = base.getRelativeWithRemapping("//p1/p2:target", ImmutableMap.of());
    assertThat(l.getPackageName()).isEqualTo("p1/p2");
    assertThat(l.getName()).isEqualTo("target");
  }

  @Test
  public void testGetRelativeWithRelativeLabel() throws Exception {
    Label base = Label.parseAbsolute("//foo/bar:baz", ImmutableMap.of());
    Label l = base.getRelativeWithRemapping(":quux", ImmutableMap.of());
    assertThat(l.getPackageName()).isEqualTo("foo/bar");
    assertThat(l.getName()).isEqualTo("quux");
  }

  @Test
  public void testGetRelativeWithIllegalLabel() throws Exception {
    Label base = Label.parseAbsolute("//foo/bar:baz", ImmutableMap.of());
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
    PackageIdentifier packageId = PackageIdentifier.create("@repo", PathFragment.create("foo"));
    Label base = Label.create(packageId, "bar");

    Label relative = base.getRelativeWithRemapping("@remote//x:y", ImmutableMap.of());

    assertThat(relative.getRepository()).isEqualTo(RepositoryName.create("@remote"));
    assertThat(relative.getPackageFragment()).isEqualTo(PathFragment.create("x"));
    assertThat(relative.getName()).isEqualTo("y");
  }

  @Test
  public void testGetRelativeWithoutRemappingBaseLabel() throws Exception {
    PackageIdentifier packageId = PackageIdentifier.create("@a", PathFragment.create("foo"));
    Label base = Label.create(packageId, "bar");
    ImmutableMap<RepositoryName, RepositoryName> repoMapping =
        ImmutableMap.of(RepositoryName.create("@a"), RepositoryName.create("@b"));
    Label relative = base.getRelativeWithRemapping(":y", repoMapping);

    // getRelative should only remap repositories passed in the string arg and not
    // make changes to existing Labels
    Label actual = Label.parseAbsoluteUnchecked("@a//foo:y");
    assertThat(relative).isEqualTo(actual);
  }

  @Test
  public void testGetRelativeWithDifferentRepoAndRemapping() throws Exception {
    PackageIdentifier packageId = PackageIdentifier.create("@repo", PathFragment.create("foo"));
    Label base = Label.create(packageId, "bar");
    ImmutableMap<RepositoryName, RepositoryName> repoMapping =
        ImmutableMap.of(RepositoryName.create("@a"), RepositoryName.create("@b"));
    Label relative = base.getRelativeWithRemapping("@a//x:y", repoMapping);

    Label actual = Label.parseAbsoluteUnchecked("@b//x:y");
    assertThat(relative).isEqualTo(actual);
  }

  @Test
  public void testGetRelativeWithRepoLocalAbsoluteLabel() throws Exception {
    PackageIdentifier packageId = PackageIdentifier.create("@repo", PathFragment.create("foo"));
    Label base = Label.create(packageId, "bar");

    Label relative = base.getRelativeWithRemapping("//x:y", ImmutableMap.of());

    assertThat(relative.getRepository()).isEqualTo(packageId.getRepository());
    assertThat(relative.getPackageFragment()).isEqualTo(PathFragment.create("x"));
    assertThat(relative.getName()).isEqualTo("y");
  }

  @Test
  public void testGetRelativeWithLocalRepoRelativeLabel() throws Exception {
    PackageIdentifier packageId = PackageIdentifier.create("@repo", PathFragment.create("foo"));
    Label base = Label.create(packageId, "bar");

    Label relative = base.getRelativeWithRemapping(":y", ImmutableMap.of());

    assertThat(relative.getRepository()).isEqualTo(packageId.getRepository());
    assertThat(relative.getPackageFragment()).isEqualTo(PathFragment.create("foo"));
    assertThat(relative.getName()).isEqualTo("y");
  }

  @Test
  public void testGetRelativeWithRepoAndReservedPackage() throws Exception {
    PackageIdentifier packageId = PackageIdentifier.create("@repo", PathFragment.create("foo"));
    Label base = Label.create(packageId, "bar");

    Label relative = base.getRelativeWithRemapping("//conditions:default", ImmutableMap.of());

    PackageIdentifier expected = PackageIdentifier.createInMainRepo("conditions");
    assertThat(relative.getRepository()).isEqualTo(expected.getRepository());
    assertThat(relative.getPackageFragment()).isEqualTo(expected.getPackageFragment());
    assertThat(relative.getName()).isEqualTo("default");
  }

  @Test
  public void testGetRelativeWithRemoteRepoToDefaultRepo() throws Exception {
    PackageIdentifier packageId = PackageIdentifier.create("@repo", PathFragment.create("foo"));
    Label base = Label.create(packageId, "bar");

    Label relative = base.getRelativeWithRemapping("@//x:y", ImmutableMap.of());

    assertThat(relative.getRepository()).isEqualTo(RepositoryName.create("@"));
    assertThat(relative.getPackageFragment()).isEqualTo(PathFragment.create("x"));
    assertThat(relative.getName()).isEqualTo("y");
  }

  @Test
  public void testGetRepositoryRelative() throws Exception {
    Label defaultBase = Label.parseAbsolute("//foo/bar:baz", ImmutableMap.of());
    Label repoBase = Label.parseAbsolute("@repo//foo/bar:baz", ImmutableMap.of());
    Label mainBase = Label.parseAbsolute("@//foo/bar:baz", ImmutableMap.of());
    Label externalTarget = Label.parseAbsolute("//external:target", ImmutableMap.of());
    Label l = defaultBase.resolveRepositoryRelative(externalTarget);
    assertThat(l.getRepository().isMain()).isTrue();
    assertThat(l.getPackageName()).isEqualTo("external");
    assertThat(l.getName()).isEqualTo("target");
    assertThat(repoBase.resolveRepositoryRelative(externalTarget)).isEqualTo(l);
    assertThat(mainBase.resolveRepositoryRelative(externalTarget)).isEqualTo(l);
  }

  @Test
  public void testFactory() throws Exception {
    Label l = Label.create("foo/bar", "quux");
    assertThat(l.getPackageName()).isEqualTo("foo/bar");
    assertThat(l.getName()).isEqualTo("quux");
  }

  @Test
  public void testIdentities() throws Exception {

    Label l1 = Label.parseAbsolute("//foo/bar:baz", ImmutableMap.of());
    Label l2 = Label.parseAbsolute("//foo/bar:baz", ImmutableMap.of());
    Label l3 = Label.parseAbsolute("//foo/bar:quux", ImmutableMap.of());

    new EqualsTester()
        .addEqualityGroup(l1, l2)
        .addEqualityGroup(l3)
        .testEquals();
  }

  @Test
  public void testToString() throws Exception {
    {
      String s = "@//foo/bar:baz";
      Label l = Label.parseAbsolute(s, ImmutableMap.of());
      assertThat(l.toString()).isEqualTo("//foo/bar:baz");
    }
    {
      Label l = Label.parseAbsolute("//foo/bar", ImmutableMap.of());
      assertThat(l.toString()).isEqualTo("//foo/bar:bar");
    }
    {
      Label l = Label.parseAbsolute("@foo", ImmutableMap.of());
      assertThat(l.toString()).isEqualTo("@foo//:foo");
    }
  }

  @Test
  public void testToShorthandString() throws Exception {
    {
      Label l = Label.parseAbsolute("//bar/baz:baz", ImmutableMap.of());
      assertThat(l.toShorthandString()).isEqualTo("//bar/baz");
    }
    {
      Label l = Label.parseAbsolute("//bar/baz:bat", ImmutableMap.of());
      assertThat(l.toShorthandString()).isEqualTo("//bar/baz:bat");
    }
    {
      Label l = Label.parseAbsolute("@foo//bar/baz:baz", ImmutableMap.of());
      assertThat(l.toShorthandString()).isEqualTo("@foo//bar/baz");
    }
    {
      Label l = Label.parseAbsolute("@foo//bar/baz:bat", ImmutableMap.of());
      assertThat(l.toShorthandString()).isEqualTo("@foo//bar/baz:bat");
    }
  }

  @Test
  public void testDotDot() throws Exception {
    Label.parseAbsolute("//foo/bar:baz..gif", ImmutableMap.of());
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
            () -> Label.parseAbsolute(label, ImmutableMap.of()));
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
    assertThat(Label.parseAbsolute("//foo:dir", ImmutableMap.of()))
        .isEqualTo(Label.parseAbsolute("//foo:dir/.", ImmutableMap.of()));
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
    Label.parseAbsolute("//foo:..bar", ImmutableMap.of());
    Label.parseAbsolute("//Foo:..bar", ImmutableMap.of());
    Label.parseAbsolute("//-Foo:..bar", ImmutableMap.of());
    Label.parseAbsolute("//00:..bar", ImmutableMap.of());
    Label.parseAbsolute("//package:foo+bar", ImmutableMap.of());
    Label.parseAbsolute("//package:foo_bar", ImmutableMap.of());
    Label.parseAbsolute("//package:foo=bar", ImmutableMap.of());
    Label.parseAbsolute("//package:foo-bar", ImmutableMap.of());
    Label.parseAbsolute("//package:foo.bar", ImmutableMap.of());
    Label.parseAbsolute("//package:foo@bar", ImmutableMap.of());
    Label.parseAbsolute("//package:foo~bar", ImmutableMap.of());
    Label.parseAbsolute("//$( ):$( )", ImmutableMap.of());
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
  public void testSerializationSimple() throws Exception {
    checkSerialization("//a", 93);
  }

  @Test
  public void testSerializationNested() throws Exception {
    checkSerialization("//foo/bar:baz", 101);
  }

  @Test
  public void testSerializationWithoutTargetName() throws Exception {
    checkSerialization("//foo/bar", 101);
  }

  private void checkSerialization(String labelString, int expectedSize) throws Exception {
    Label a = Label.parseAbsolute(labelString, ImmutableMap.of());
    byte[] sa = TestUtils.serializeObject(a);
    assertThat(sa).hasLength(expectedSize);

    Label a2 = (Label) TestUtils.deserializeObject(sa);
    assertThat(a2).isEqualTo(a);
  }

  @Test
  public void testRepoLabel() throws Exception {
    Label label = Label.parseAbsolute("@foo//bar/baz:bat/boo", ImmutableMap.of());
    assertThat(label.toString()).isEqualTo("@foo//bar/baz:bat/boo");
  }

  @Test
  public void testNoRepo() throws Exception {
    Label label = Label.parseAbsolute("//bar/baz:bat/boo", ImmutableMap.of());
    assertThat(label.toString()).isEqualTo("//bar/baz:bat/boo");
  }

  @Test
  public void testInvalidRepo() throws Exception {
    LabelSyntaxException e =
        assertThrows(
            LabelSyntaxException.class,
            () -> Label.parseAbsolute("foo//bar/baz:bat/boo", ImmutableMap.of()));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("invalid repository name 'foo': workspace names must start with '@'");
  }

  @Test
  public void testInvalidRepoWithColon() throws Exception {
    LabelSyntaxException e =
        assertThrows(
            LabelSyntaxException.class, () -> Label.parseAbsolute("@foo:xyz", ImmutableMap.of()));
    assertThat(e)
        .hasMessageThat()
        .containsMatch("invalid repository name '@foo:xyz': workspace names may contain only");
  }

  @Test
  public void testGetWorkspaceRoot() throws Exception {
    Label label = Label.parseAbsolute("//bar/baz", ImmutableMap.of());
    assertThat(label.getWorkspaceRootForStarlarkOnly(StarlarkSemantics.DEFAULT)).isEmpty();
    label = Label.parseAbsolute("@repo//bar/baz", ImmutableMap.of());
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
    assertThat(Label.parseAbsolute("@foo//bar:baz", ImmutableMap.of()).getWorkspaceName())
        .isEqualTo("foo");
    assertThat(Label.parseAbsolute("//bar:baz", ImmutableMap.of()).getWorkspaceName()).isEmpty();
    assertThat(Label.parseAbsolute("@//bar:baz", ImmutableMap.of()).getWorkspaceName()).isEmpty();
  }

  @Test
  public void testStarlarkStrAndRepr() throws Exception {
    Label label = Label.parseAbsolute("//x", ImmutableMap.of());
    assertThat(Starlark.str(label)).isEqualTo("//x:x");
    assertThat(Starlark.repr(label)).isEqualTo("Label(\"//x:x\")");
  }
}
