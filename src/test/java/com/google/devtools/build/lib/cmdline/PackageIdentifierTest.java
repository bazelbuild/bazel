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
