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

package com.google.devtools.build.lib.cmdline;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.cmdline.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 * Unit tests for {@link PackageIdentifier}.
 */
@RunWith(JUnit4.class)
public class PackageIdentifierTest {
  @Test
  public void testParsing() throws Exception {
    PackageIdentifier fooA = PackageIdentifier.parse("@foo//a");
    assertThat(fooA.getRepository().strippedName()).isEqualTo("foo");
    assertThat(fooA.getPackageFragment().getPathString()).isEqualTo("a");

    PackageIdentifier absoluteA = PackageIdentifier.parse("//a");
    assertThat(absoluteA.getRepository().strippedName()).isEqualTo("");
    assertThat(fooA.getPackageFragment().getPathString()).isEqualTo("a");

    PackageIdentifier plainA = PackageIdentifier.parse("a");
    assertThat(plainA.getRepository().strippedName()).isEqualTo("");
    assertThat(fooA.getPackageFragment().getPathString()).isEqualTo("a");

    PackageIdentifier mainA = PackageIdentifier.parse("@//a");
    assertThat(mainA.getRepository()).isEqualTo(PackageIdentifier.MAIN_REPOSITORY_NAME);
  }

  public void assertNotValid(String name, String expectedMessage) {
    try {
      RepositoryName.create(name);
      fail();
    } catch (LabelSyntaxException expected) {
      assertThat(expected.getMessage()).contains(expectedMessage);
    }
  }

  @Test
  public void testValidateRepositoryName() throws Exception {
    assertEquals("@foo", RepositoryName.create("@foo").toString());
    assertThat(RepositoryName.create("").toString()).isEmpty();
    assertEquals("@foo/bar", RepositoryName.create("@foo/bar").toString());
    assertEquals("@foo.bar", RepositoryName.create("@foo.bar").toString());
    assertEquals("@..foo", RepositoryName.create("@..foo").toString());
    assertEquals("@foo..", RepositoryName.create("@foo..").toString());
    assertEquals("@.foo", RepositoryName.create("@.foo").toString());

    assertNotValid("@/", "workspace names are not allowed to start with '@/'");
    assertNotValid("@.", "workspace names are not allowed to be '@.'");
    assertNotValid("@./", "workspace names are not allowed to start with '@./'");
    assertNotValid("@../", "workspace names are not allowed to start with '@..'");
    assertNotValid("@x/./x", "workspace names are not allowed to contain '/./'");
    assertNotValid("@x/../x", "workspace names are not allowed to contain '/../'");
    assertNotValid("@abc/", "workspace names are not allowed to end with '/'");
    assertNotValid("@/abc", "workspace names are not allowed to start with '@/'");
    assertNotValid("@a//////b", "workspace names are not allowed to contain '//'");
    assertNotValid("@foo@",
        "workspace names may contain only A-Z, a-z, 0-9, '-', '_', '.', and '/'");
    assertNotValid("@foo\0",
        "workspace names may contain only A-Z, a-z, 0-9, '-', '_', '.', and '/'");
    assertNotValid("x", "workspace names must start with '@'");
  }

  @Test
  public void testToString() throws Exception {
    PackageIdentifier local = PackageIdentifier.create("", new PathFragment("bar/baz"));
    assertEquals("bar/baz", local.toString());
    PackageIdentifier external = PackageIdentifier.create("@foo", new PathFragment("bar/baz"));
    assertEquals("@foo//bar/baz", external.toString());
  }

  @Test
  public void testCompareTo() throws Exception {
    PackageIdentifier foo1 = PackageIdentifier.create("@foo", new PathFragment("bar/baz"));
    PackageIdentifier foo2 = PackageIdentifier.create("@foo", new PathFragment("bar/baz"));
    PackageIdentifier foo3 = PackageIdentifier.create("@foo", new PathFragment("bar/bz"));
    PackageIdentifier bar = PackageIdentifier.create("@bar", new PathFragment("bar/baz"));
    assertEquals(0, foo1.compareTo(foo2));
    assertThat(foo1.compareTo(foo3)).isLessThan(0);
    assertThat(foo1.compareTo(bar)).isGreaterThan(0);
  }

  @Test
  public void testInvalidPackageName() throws Exception {
    // This shouldn't throw an exception, package names aren't validated.
    PackageIdentifier.create("@foo", new PathFragment("bar.baz"));
  }

  @Test
  public void testSerialization() throws Exception {
    PackageIdentifier inId = PackageIdentifier.create("@foo", new PathFragment("bar/baz"));
    ByteArrayOutputStream data = new ByteArrayOutputStream();
    ObjectOutputStream out = new ObjectOutputStream(data);
    out.writeObject(inId);
    ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(data.toByteArray()));
    PackageIdentifier outId = (PackageIdentifier) in.readObject();
    assertEquals(inId, outId);
  }

  @Test
  public void testPackageFragmentEquality() throws Exception {
    // Make sure package fragments are canonicalized.
    PackageIdentifier p1 = PackageIdentifier.create("@whatever", new PathFragment("foo/bar"));
    PackageIdentifier p2 = PackageIdentifier.create("@whatever", new PathFragment("foo/bar"));
    assertSame(p2.getPackageFragment(), p1.getPackageFragment());
  }
}
