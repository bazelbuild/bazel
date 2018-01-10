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
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.testing.EqualsTester;
import com.google.common.testing.GcFinalization;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.skyframe.serialization.InjectingObjectCodecAdapter;
import com.google.devtools.build.lib.skyframe.serialization.testutils.FsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.ObjectCodecTester;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.ref.WeakReference;
import java.net.URI;
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
    assertThat(child.startsWith(parent)).isEqualTo(expected);
  }

  @Test
  public void testGetChildWorks() {
    assertGetChildWorks("second");
    assertGetChildWorks("...");
    assertGetChildWorks("....");
  }

  private void assertGetChildWorks(String childName) {
    assertThat(filesystem.getPath("/first").getChild(childName))
        .isEqualTo(filesystem.getPath("/first/" + childName));
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
    assertThat(dir.getRelative(PathFragment.create("y")).toString()).isEqualTo("/first/x/y");
    assertThat(dir.getRelative(PathFragment.create("./x")).toString()).isEqualTo("/first/x/x");
    assertThat(dir.getRelative(PathFragment.create("../y")).toString()).isEqualTo("/first/y");
  }

  @Test
  public void testGetRelativeWithAbsoluteFragmentWorks() {
    Path root = filesystem.getPath("/first/x");
    assertThat(root.getRelative(PathFragment.create("/x/y")).toString()).isEqualTo("/x/y");
  }

  @Test
  public void testGetRelativeWithAbsoluteStringWorks() {
    Path root = filesystem.getPath("/first/x");
    assertThat(root.getRelative("/x/y").toString()).isEqualTo("/x/y");
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
    assertThat(root.getRelative("..")).isSameAs(root);

    assertThat(root.getRelative("broken/../../dots")).isSameAs(root.getRelative("dots"));
  }

  @Test
  public void testSingleSegmentEquivalence() {
    assertThat(root.getRelative("aSingleSegment")).isSameAs(root.getRelative("aSingleSegment"));
  }

  @Test
  public void testSiblingNonEquivalenceString() {
    assertThat(root.getRelative("aDifferentSegment"))
        .isNotSameAs(root.getRelative("aSingleSegment"));
  }

  @Test
  public void testSiblingNonEquivalenceFragment() {
    assertThat(root.getRelative(PathFragment.create("aDifferentSegment")))
        .isNotSameAs(root.getRelative(PathFragment.create("aSingleSegment")));
  }

  @Test
  public void testHashCodeStableAcrossGarbageCollections() {
    Path parent = filesystem.getPath("/a");
    PathFragment childFragment = PathFragment.create("b");
    Path child = parent.getRelative(childFragment);
    WeakReference<Path> childRef = new WeakReference<>(child);
    int childHashCode1 = childRef.get().hashCode();
    assertThat(parent.getRelative(childFragment).hashCode()).isEqualTo(childHashCode1);
    child = null;
    GcFinalization.awaitClear(childRef);
    int childHashCode2 = parent.getRelative(childFragment).hashCode();
    assertThat(childHashCode2).isEqualTo(childHashCode1);
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

      assertThat(p2.startsWith(p1)).isTrue();
      assertThat(p2.startsWith(dsP1)).isTrue();
      assertThat(dsP2.startsWith(p1)).isTrue();
      assertThat(dsP2.startsWith(dsP1)).isTrue();

      // Regression test for a very specific bug in compareTo involving our incorrect usage of
      // reference equality rather than logical equality.
      String relativePathStringA = "child/grandchildA";
      String relativePathStringB = "child/grandchildB";
      assertThat(
              p1.getRelative(relativePathStringA).compareTo(dsP1.getRelative(relativePathStringB)))
          .isEqualTo(
              p1.getRelative(relativePathStringA).compareTo(p1.getRelative(relativePathStringB)));
    } finally {
      Path.setFileSystemForSerialization(oldFileSystem);
    }
  }

  @Test
  public void testAbsolutePathRoot() {
    assertThat(new Path(null).toString()).isEqualTo("/");
  }

  @Test
  public void testAbsolutePath() {
    Path segment = new Path(null, "bar.txt",
      new Path(null, "foo", new Path(null)));
    assertThat(segment.toString()).isEqualTo("/foo/bar.txt");
  }

  @Test
  public void testToURI() throws Exception {
    Path p = root.getRelative("/tmp/foo bar.txt");
    URI uri = p.toURI();
    assertThat(uri.toString()).isEqualTo("file:///tmp/foo%20bar.txt");
  }

  @Test
  public void testCodec() throws Exception {
    ObjectCodecTester.newBuilder(
            new InjectingObjectCodecAdapter<>(Path.CODEC, FsUtils.TEST_FILESYSTEM_PROVIDER))
        .addSubjects(
            ImmutableList.of(
                FsUtils.TEST_FILESYSTEM.getPath("/"),
                FsUtils.TEST_FILESYSTEM.getPath("/some/path"),
                FsUtils.TEST_FILESYSTEM.getPath("/some/other/path/with/empty/last/fragment/")))
        .buildAndRunTests();
  }

  private void assertAsFragmentWorks(String expected) {
    assertThat(filesystem.getPath(expected).asFragment()).isEqualTo(PathFragment.create(expected));
  }

  private void assertGetRelativeWorks(String expected, String relative) {
    assertThat(filesystem.getPath("/first/x").getRelative(relative))
        .isEqualTo(filesystem.getPath(expected));
  }

  private void assertRelativeToWorks(String expected, String relative, String original) {
    assertThat(filesystem.getPath(relative).relativeTo(filesystem.getPath(original)))
        .isEqualTo(PathFragment.create(expected));
  }
}
