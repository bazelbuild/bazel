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
import static org.junit.Assert.assertThrows;

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link RootedPath}.
 */
@RunWith(JUnit4.class)
public class RootedPathTest {
  private FileSystem filesystem;
  private Path root;

  @Before
  public final void initializeFileSystem() throws Exception  {
    filesystem = new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256);
    root = filesystem.getPath("/");
  }

  @Test
  public void testEqualsAndHashCodeContract() {
    Path pkgRoot1 = root.getRelative("pkgroot1");
    Path pkgRoot2 = root.getRelative("pkgroot2");
    RootedPath rootedPathA1 =
        RootedPath.toRootedPath(Root.fromPath(pkgRoot1), PathFragment.create("foo/bar"));
    RootedPath rootedPathA2 =
        RootedPath.toRootedPath(Root.fromPath(pkgRoot1), PathFragment.create("foo/bar"));
    RootedPath absolutePath1 =
        RootedPath.toRootedPath(Root.fromPath(root), PathFragment.create("pkgroot1/foo/bar"));
    RootedPath rootedPathB1 =
        RootedPath.toRootedPath(Root.fromPath(pkgRoot2), PathFragment.create("foo/bar"));
    RootedPath rootedPathB2 =
        RootedPath.toRootedPath(Root.fromPath(pkgRoot2), PathFragment.create("foo/bar"));
    RootedPath absolutePath2 =
        RootedPath.toRootedPath(Root.fromPath(root), PathFragment.create("pkgroot2/foo/bar"));
    new EqualsTester()
        .addEqualityGroup(rootedPathA1, rootedPathA2)
        .addEqualityGroup(rootedPathB1, rootedPathB2)
        .addEqualityGroup(absolutePath1)
        .addEqualityGroup(absolutePath2)
        .testEquals();
  }

  @Test
  public void testGetParentDirectory() {
    RootedPath path = createRootedPath("root/folder", "folder1/folder2");

    RootedPath parent = path.getParentDirectory();
    assertThat(parent).isNotNull();
    assertThat(parent.asPath().getPathString()).isEqualTo("/root/folder/folder1");
    assertThat(parent.getRootRelativePath().getPathString()).isEqualTo("folder1");

    parent = parent.getParentDirectory();
    assertThat(parent).isNotNull();
    assertThat(parent.asPath().getPathString()).isEqualTo("/root/folder");
    assertThat(parent.getRootRelativePath().getPathString()).isEmpty();

    assertThat(parent.getParentDirectory()).isNull();
  }

  @Test
  public void testGetParentDirectoryOfRoot() {
    RootedPath path = createRootedPath("root", "");
    assertThat(path.getParentDirectory()).isNull();
  }

  @Test
  public void testGetChild() {
    RootedPath rootPath = createRootedPath("root", "");
    RootedPath child = rootPath.getChild("foo");
    RootedPath grandchild = child.getChild("BUILD");

    assertThat(child).isEqualTo(createRootedPath("root", "foo"));
    assertThat(grandchild).isEqualTo(createRootedPath("root", "foo/BUILD"));
    assertThat(grandchild.getRoot()).isEqualTo(rootPath.getRoot());
    assertThat(grandchild.getParentDirectory()).isEqualTo(child);
    assertThat(grandchild.asPath().getPathString()).isEqualTo("/root/foo/BUILD");
  }

  @Test
  public void testGetChildOfAbsoluteRoot() {
    RootedPath rootPath = RootedPath.toRootedPath(Root.absoluteRoot(filesystem), root);
    RootedPath child = rootPath.getChild("foo");

    assertThat(child.getRootRelativePath().getPathString()).isEqualTo("/foo");
    assertThat(child.asPath()).isEqualTo(root.getChild("foo"));
    assertThat(child.getParentDirectory()).isEqualTo(rootPath);
  }

  @Test
  public void testGetChildRejectsInvalidBaseName() {
    RootedPath path = createRootedPath("root", "foo");
    for (String invalid : new String[] {"", ".", "..", "a/b", "/a"}) {
      assertThrows(IllegalArgumentException.class, () -> path.getChild(invalid));
    }
  }

  @Test
  public void testGetRootRelativeBaseName() {
    assertThat(createRootedPath("root", "foo/bar").getRootRelativeBaseName()).isEqualTo("bar");
    assertThat(createRootedPath("root", "foo").getRootRelativeBaseName()).isEqualTo("foo");
    assertThat(createRootedPath("root", "").getRootRelativeBaseName()).isEmpty();
    assertThat(
            RootedPath.toRootedPath(Root.absoluteRoot(filesystem), root).getRootRelativeBaseName())
        .isEmpty();
  }

  private RootedPath createRootedPath(String relativeRootPath, String relativePath) {
    return RootedPath.toRootedPath(
        Root.fromPath(root.getRelative(relativeRootPath)), PathFragment.create(relativePath));
  }
}
