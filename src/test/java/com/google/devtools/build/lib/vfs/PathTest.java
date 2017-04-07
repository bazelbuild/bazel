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
package com.google.devtools.build.lib.vfs;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.Lists;
import com.google.common.testing.EqualsTester;
import com.google.common.testing.GcFinalization;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.ref.WeakReference;
import java.util.Collections;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for {@link Path}.
 */
@RunWith(JUnit4.class)
public class PathTest {
  private FileSystem filesystem;
  private Path root;

  @Before
  public final void initializeFileSystem() throws Exception  {
    filesystem = new InMemoryFileSystem(BlazeClock.instance());
    root = filesystem.getRootDirectory();
    Path first = root.getChild("first");
    first.createDirectory();
  }

  @Test
  public void testStartsWithWorksForSelf() {
    assertStartsWithReturns(true, "/first/child", "/first/child");
  }

  @Test
  public void testStartsWithWorksForChild() {
    assertStartsWithReturns(true,
        "/first/child", "/first/child/grandchild");
  }

  @Test
  public void testStartsWithWorksForDeepDescendant() {
    assertStartsWithReturns(true,
        "/first/child", "/first/child/grandchild/x/y/z");
  }

  @Test
  public void testStartsWithFailsForParent() {
    assertStartsWithReturns(false, "/first/child", "/first");
  }

  @Test
  public void testStartsWithFailsForSibling() {
    assertStartsWithReturns(false, "/first/child", "/first/child2");
  }

  @Test
  public void testStartsWithFailsForLinkToDescendant()
      throws Exception {
    Path linkTarget = filesystem.getPath("/first/linked_to");
    FileSystemUtils.createEmptyFile(linkTarget);
    Path second = filesystem.getPath("/second/");
    second.createDirectory();
    second.getChild("child_link").createSymbolicLink(linkTarget);
    assertStartsWithReturns(false, "/first", "/second/child_link");
  }

  @Test
  public void testStartsWithFailsForNullPrefix() {
    try {
      filesystem.getPath("/first").startsWith(null);
      fail();
    } catch (Exception e) {
    }
  }

  private void assertStartsWithReturns(boolean expected,
                                       String ancestor,
                                       String descendant) {
    Path parent = filesystem.getPath(ancestor);
    Path child = filesystem.getPath(descendant);
    assertEquals(expected, child.startsWith(parent));
  }

  @Test
  public void testGetChildWorks() {
    assertGetChildWorks("second");
    assertGetChildWorks("...");
    assertGetChildWorks("....");
  }

  private void assertGetChildWorks(String childName) {
    assertEquals(filesystem.getPath("/first/" + childName),
        filesystem.getPath("/first").getChild(childName));
  }

  @Test
  public void testGetChildFailsForChildWithSlashes() {
    assertGetChildFails("second/third");
    assertGetChildFails("./third");
    assertGetChildFails("../third");
    assertGetChildFails("second/..");
    assertGetChildFails("second/.");
    assertGetChildFails("/third");
    assertGetChildFails("third/");
  }

  private void assertGetChildFails(String childName) {
    try {
      filesystem.getPath("/first").getChild(childName);
      fail();
    } catch (IllegalArgumentException e) {
      // Expected.
    }
  }

  @Test
  public void testGetChildFailsForDotAndDotDot() {
    assertGetChildFails(".");
    assertGetChildFails("..");
  }

  @Test
  public void testGetChildFailsForEmptyString() {
    assertGetChildFails("");
  }

  @Test
  public void testRelativeToWorks() {
    assertRelativeToWorks("apple", "/fruit/apple", "/fruit");
    assertRelativeToWorks("apple/jonagold", "/fruit/apple/jonagold", "/fruit");
  }

  @Test
  public void testGetRelativeWithStringWorks() {
    assertGetRelativeWorks("/first/x/y", "y");
    assertGetRelativeWorks("/y", "/y");
    assertGetRelativeWorks("/first/x/x", "./x");
    assertGetRelativeWorks("/first/y", "../y");
    assertGetRelativeWorks("/", "../../../../..");
  }

  @Test
  public void testAsFragmentWorks() {
    assertAsFragmentWorks("/");
    assertAsFragmentWorks("//");
    assertAsFragmentWorks("/first");
    assertAsFragmentWorks("/first/x/y");
    assertAsFragmentWorks("/first/x/y.foo");
  }

  @Test
  public void testGetRelativeWithFragmentWorks() {
    Path dir = filesystem.getPath("/first/x");
    assertEquals("/first/x/y",
                 dir.getRelative(PathFragment.create("y")).toString());
    assertEquals("/first/x/x",
                 dir.getRelative(PathFragment.create("./x")).toString());
    assertEquals("/first/y",
                 dir.getRelative(PathFragment.create("../y")).toString());

  }

  @Test
  public void testGetRelativeWithAbsoluteFragmentWorks() {
    Path root = filesystem.getPath("/first/x");
    assertEquals("/x/y",
                 root.getRelative(PathFragment.create("/x/y")).toString());
  }

  @Test
  public void testGetRelativeWithAbsoluteStringWorks() {
    Path root = filesystem.getPath("/first/x");
    assertEquals("/x/y", root.getRelative("/x/y").toString());
  }

  @Test
  public void testComparableSortOrder() {
    Path zzz = filesystem.getPath("/zzz");
    Path ZZZ = filesystem.getPath("/ZZZ");
    Path abc = filesystem.getPath("/abc");
    Path aBc = filesystem.getPath("/aBc");
    Path AbC = filesystem.getPath("/AbC");
    Path ABC = filesystem.getPath("/ABC");
    List<Path> list = Lists.newArrayList(zzz, ZZZ, ABC, aBc, AbC, abc);
    Collections.sort(list);
    assertThat(list).containsExactly(ABC, AbC, ZZZ, aBc, abc, zzz).inOrder();
  }

  @Test
  public void testParentOfRootIsRoot() {
    assertSame(root, root.getRelative(".."));

    assertSame(root.getRelative("dots"),
               root.getRelative("broken/../../dots"));
  }

  @Test
  public void testSingleSegmentEquivalence() {
    assertSame(
        root.getRelative("aSingleSegment"),
        root.getRelative("aSingleSegment"));
  }

  @Test
  public void testSiblingNonEquivalenceString() {
    assertNotSame(
        root.getRelative("aSingleSegment"),
        root.getRelative("aDifferentSegment"));
  }

  @Test
  public void testSiblingNonEquivalenceFragment() {
    assertNotSame(
        root.getRelative(PathFragment.create("aSingleSegment")),
        root.getRelative(PathFragment.create("aDifferentSegment")));
  }

  @Test
  public void testHashCodeStableAcrossGarbageCollections() {
    Path parent = filesystem.getPath("/a");
    PathFragment childFragment = PathFragment.create("b");
    Path child = parent.getRelative(childFragment);
    WeakReference<Path> childRef = new WeakReference<>(child);
    int childHashCode1 = childRef.get().hashCode();
    assertEquals(childHashCode1, parent.getRelative(childFragment).hashCode());
    child = null;
    GcFinalization.awaitClear(childRef);
    int childHashCode2 = parent.getRelative(childFragment).hashCode();
    assertEquals(childHashCode1, childHashCode2);
  }

  @Test
  public void testSerialization() throws Exception {
    FileSystem oldFileSystem = Path.getFileSystemForSerialization();
    try {
      Path.setFileSystemForSerialization(filesystem);
      Path root = filesystem.getPath("/");
      Path p1 = filesystem.getPath("/foo");
      Path p2 = filesystem.getPath("/foo/bar");

      ByteArrayOutputStream bos = new ByteArrayOutputStream();
      ObjectOutputStream oos = new ObjectOutputStream(bos);

      oos.writeObject(root);
      oos.writeObject(p1);
      oos.writeObject(p2);

      ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
      ObjectInputStream ois = new ObjectInputStream(bis);

      Path dsRoot = (Path) ois.readObject();
      Path dsP1 = (Path) ois.readObject();
      Path dsP2 = (Path) ois.readObject();

      new EqualsTester()
          .addEqualityGroup(root, dsRoot)
          .addEqualityGroup(p1, dsP1)
          .addEqualityGroup(p2, dsP2)
          .testEquals();

      assertTrue(p2.startsWith(p1));
      assertTrue(p2.startsWith(dsP1));
      assertTrue(dsP2.startsWith(p1));
      assertTrue(dsP2.startsWith(dsP1));

      // Regression test for a very specific bug in compareTo involving our incorrect usage of
      // reference equality rather than logical equality.
      String relativePathStringA = "child/grandchildA";
      String relativePathStringB = "child/grandchildB";
      assertEquals(
          p1.getRelative(relativePathStringA).compareTo(p1.getRelative(relativePathStringB)),
          p1.getRelative(relativePathStringA).compareTo(dsP1.getRelative(relativePathStringB)));
    } finally {
      Path.setFileSystemForSerialization(oldFileSystem);
    }
  }

  @Test
  public void testAbsolutePathRoot() {
    assertEquals("/", new Path(null).toString());
  }

  @Test
  public void testAbsolutePath() {
    Path segment = new Path(null, "bar.txt",
      new Path(null, "foo", new Path(null)));
    assertEquals("/foo/bar.txt", segment.toString());
  }

  private void assertAsFragmentWorks(String expected) {
    assertEquals(PathFragment.create(expected), filesystem.getPath(expected).asFragment());
  }

  private void assertGetRelativeWorks(String expected, String relative) {
    assertEquals(filesystem.getPath(expected),
        filesystem.getPath("/first/x").getRelative(relative));
  }

  private void assertRelativeToWorks(String expected, String relative, String original) {
    assertEquals(PathFragment.create(expected),
                 filesystem.getPath(relative).relativeTo(filesystem.getPath(original)));
  }
}
