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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.util.Preconditions;

import org.junit.Before;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collection;

/**
 * Generic tests for any file system that implements {@link ScopeEscapableFileSystem},
 * i.e. any file system that supports symlinks that escape its scope.
 *
 * Each suitable file system test should inherit from this class, thereby obtaining
 * all the tests.
 */
public abstract class ScopeEscapableFileSystemTest extends SymlinkAwareFileSystemTest {

  /**
   * Trivial FileSystem implementation that can record the last path passed to each method
   * and read/write to a unified "state" variable (which can then be checked by tests) for
   * each data type this class manipulates.
   *
   * The default implementation of each method throws an exception. Each test case should
   * selectively override the methods it expects to be invoked.
   */
  private static class TestDelegator extends FileSystem {
    protected Path lastPath;
    protected boolean booleanState;
    protected long longState;
    protected Object objectState;

    public void setState(boolean state) { booleanState = state; }
    public void setState(long state) { longState = state; }
    public void setState(Object state) { objectState = state; }

    public boolean booleanState() { return booleanState; }
    public long longState() { return longState; }
    public Object objectState() { return objectState; }

    public PathFragment lastPath() {
      Path ans = lastPath;
      // Clear this out to protect against accidental matches when testing the same path multiple
      // consecutive times.
      lastPath = null;
      return ans != null ? ans.asFragment() : null;
    }

    @Override public boolean supportsModifications() {
      return true;
    }

    @Override public boolean supportsSymbolicLinksNatively() {
      return true;
    }

    private static RuntimeException re() {
      return new RuntimeException("This method should not be called in this context");
    }

    @Override protected boolean isReadable(Path path) { throw re(); }
    @Override protected boolean isWritable(Path path) { throw re(); }
    @Override protected boolean isDirectory(Path path, boolean followSymlinks) { throw re(); }
    @Override protected boolean isFile(Path path, boolean followSymlinks) { throw re(); }
    @Override protected boolean isSpecialFile(Path path, boolean followSymlinks) { throw re(); }
    @Override protected boolean isExecutable(Path path) { throw re(); }
    @Override protected boolean exists(Path path, boolean followSymlinks) {throw re(); }
    @Override protected boolean isSymbolicLink(Path path) { throw re(); }
    @Override protected boolean createDirectory(Path path) { throw re(); }
    @Override protected boolean delete(Path path) { throw re(); }

    @Override protected long getFileSize(Path path, boolean followSymlinks) { throw re(); }
    @Override protected long getLastModifiedTime(Path path, boolean followSymlinks) { throw re(); }

    @Override protected void setWritable(Path path, boolean writable) { throw re(); }
    @Override protected void setExecutable(Path path, boolean executable) { throw re(); }
    @Override protected void setReadable(Path path, boolean readable) { throw re(); }
    @Override protected void setLastModifiedTime(Path path, long newTime) { throw re(); }
    @Override protected void renameTo(Path sourcePath, Path targetPath) { throw re(); }
    @Override protected void createSymbolicLink(Path linkPath, PathFragment targetFragment) {
      throw re();
    }

    @Override protected PathFragment readSymbolicLink(Path path) { throw re(); }
    @Override protected InputStream getInputStream(Path path) { throw re(); }
    @Override protected Collection<Path> getDirectoryEntries(Path path) { throw re(); }
    @Override protected OutputStream getOutputStream(Path path, boolean append)  { throw re(); }
    @Override
    protected FileStatus statIfFound(Path path, boolean followSymlinks) throws IOException {
      throw re();
    }
  }

  protected static final PathFragment SCOPE_ROOT = new PathFragment("/fs/root");

  private Path fileLink;
  private PathFragment fileLinkTarget;
  private Path dirLink;
  private PathFragment dirLinkTarget;

  @Before
  public final void createLinks() throws Exception  {
    Preconditions.checkState(
        testFS instanceof ScopeEscapableFileSystem, "Not ScopeEscapable: %s", testFS);
    ((ScopeEscapableFileSystem) testFS).enableScopeChecking(false);
    for (int i = 1; i <= SCOPE_ROOT.segmentCount(); i++) {
      testFS.getPath(SCOPE_ROOT.subFragment(0, i)).createDirectory();
    }

    fileLink = testFS.getPath(SCOPE_ROOT.getRelative("link"));
    fileLinkTarget = new PathFragment("/should/be/delegated/fileLinkTarget");
    testFS.createSymbolicLink(fileLink, fileLinkTarget);

    dirLink = testFS.getPath(SCOPE_ROOT.getRelative("dirlink"));
    dirLinkTarget = new PathFragment("/should/be/delegated/dirLinkTarget");
    testFS.createSymbolicLink(dirLink, dirLinkTarget);
  }

  /**
   * Returns the file system supplied by {@link #getFreshFileSystem}, cast to
   * a {@link ScopeEscapableFileSystem}. Also enables scope checking within
   * the file system (which we keep disabled for inherited tests that aren't
   * intended to test scope boundaries).
   */
  private ScopeEscapableFileSystem scopedFS() {
    ScopeEscapableFileSystem fs = (ScopeEscapableFileSystem) testFS;
    fs.enableScopeChecking(true);
    return fs;
  }

  // Checks that the semi-resolved path passed to the delegator matches the expected value.
  private void checkPath(TestDelegator delegator, PathFragment expectedDelegatedPath) {
    assertEquals(delegator.lastPath(), expectedDelegatedPath);
  }

  // Asserts that the condition is false and checks that the expected path was delegated.
  private void assertFalseWithPathCheck(boolean result, TestDelegator delegator,
      PathFragment expectedDelegatedPath) {
    assertFalse(result);
    checkPath(delegator, expectedDelegatedPath);
  }

  // Asserts that the condition is true and checks that the expected path was delegated.
  private void assertTrueWithPathCheck(boolean result, TestDelegator delegator,
      PathFragment expectedDelegatedPath) {
    assertTrue(result);
    checkPath(delegator, expectedDelegatedPath);
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests:
  /////////////////////////////////////////////////////////////////////////////

  @Test
  public void testIsReadableCallOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected boolean isReadable(Path path) {
        lastPath = path;
        return booleanState();
      }
    };
    scopedFS().setDelegator(delegator);

    delegator.setState(false);
    assertFalseWithPathCheck(fileLink.isReadable(), delegator, fileLinkTarget);
    assertFalseWithPathCheck(dirLink.getRelative("a").isReadable(), delegator,
        dirLinkTarget.getRelative("a"));

    delegator.setState(true);
    assertTrueWithPathCheck(fileLink.isReadable(), delegator, fileLinkTarget);
    assertTrueWithPathCheck(dirLink.getRelative("a").isReadable(), delegator,
        dirLinkTarget.getRelative("a"));
  }

  @Test
  public void testIsWritableCallOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected boolean isWritable(Path path) {
        lastPath = path;
        return booleanState();
      }
    };
    scopedFS().setDelegator(delegator);

    delegator.setState(false);
    assertFalseWithPathCheck(fileLink.isWritable(), delegator, fileLinkTarget);
    assertFalseWithPathCheck(dirLink.getRelative("a").isWritable(), delegator,
        dirLinkTarget.getRelative("a"));

    delegator.setState(true);
    assertTrueWithPathCheck(fileLink.isWritable(), delegator, fileLinkTarget);
    assertTrueWithPathCheck(dirLink.getRelative("a").isWritable(), delegator,
        dirLinkTarget.getRelative("a"));
  }

  @Test
  public void testisExecutableCallOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected boolean isExecutable(Path path) {
        lastPath = path;
        return booleanState();
      }
    };
    scopedFS().setDelegator(delegator);

    delegator.setState(false);
    assertFalseWithPathCheck(fileLink.isExecutable(), delegator, fileLinkTarget);
    assertFalseWithPathCheck(dirLink.getRelative("a").isExecutable(), delegator,
        dirLinkTarget.getRelative("a"));

    delegator.setState(true);
    assertTrueWithPathCheck(fileLink.isExecutable(), delegator, fileLinkTarget);
    assertTrueWithPathCheck(dirLink.getRelative("a").isExecutable(), delegator,
        dirLinkTarget.getRelative("a"));
  }

  @Test
  public void testIsDirectoryCallOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected boolean isDirectory(Path path, boolean followSymlinks) {
        lastPath = path;
        return booleanState();
      }
      @Override protected boolean exists(Path path, boolean followSymlinks) { return true; }
      @Override protected long getLastModifiedTime(Path path, boolean followSymlinks) { return 0; }
    };
    scopedFS().setDelegator(delegator);

    delegator.setState(false);
    assertFalseWithPathCheck(fileLink.isDirectory(), delegator, fileLinkTarget);
    assertFalseWithPathCheck(dirLink.getRelative("a").isDirectory(), delegator,
        dirLinkTarget.getRelative("a"));

    delegator.setState(true);
    assertTrueWithPathCheck(fileLink.isDirectory(), delegator, fileLinkTarget);
    assertTrueWithPathCheck(dirLink.getRelative("a").isDirectory(), delegator,
        dirLinkTarget.getRelative("a"));
  }

  @Test
  public void testIsFileCallOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected boolean isFile(Path path, boolean followSymlinks) {
        lastPath = path;
        return booleanState();
      }
      @Override protected boolean exists(Path path, boolean followSymlinks) { return true; }
      @Override protected long getLastModifiedTime(Path path, boolean followSymlinks) { return 0; }
    };
    scopedFS().setDelegator(delegator);

    delegator.setState(false);
    assertFalseWithPathCheck(fileLink.isFile(), delegator, fileLinkTarget);
    assertFalseWithPathCheck(dirLink.getRelative("a").isFile(), delegator,
        dirLinkTarget.getRelative("a"));

    delegator.setState(true);
    assertTrueWithPathCheck(fileLink.isFile(), delegator, fileLinkTarget);
    assertTrueWithPathCheck(dirLink.getRelative("a").isFile(), delegator,
        dirLinkTarget.getRelative("a"));
  }

  @Test
  public void testIsSymbolicLinkCallOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected boolean isSymbolicLink(Path path) {
        lastPath = path;
        return booleanState();
      }
      @Override protected boolean exists(Path path, boolean followSymlinks) { return true; }
      @Override protected long getLastModifiedTime(Path path, boolean followSymlinks) { return 0; }
      @Override protected boolean isDirectory(Path path, boolean followSymlinks) { return true; }
    };
    scopedFS().setDelegator(delegator);

    // We shouldn't follow final-segment links, so they should never invoke the delegator.
    delegator.setState(false);
    assertTrue(fileLink.isSymbolicLink());
    assertNull(delegator.lastPath());

    assertFalseWithPathCheck(dirLink.getRelative("a").isSymbolicLink(), delegator,
        dirLinkTarget.getRelative("a"));

    delegator.setState(true);
    assertTrueWithPathCheck(dirLink.getRelative("a").isSymbolicLink(), delegator,
        dirLinkTarget.getRelative("a"));
  }

  /**
   * Returns a test delegator that reflects info passed to Path.exists() calls.
   */
  private TestDelegator newExistsDelegator() {
    return new TestDelegator() {
      @Override protected boolean exists(Path path, boolean followSymlinks) {
        lastPath = path;
        return booleanState();
      }
      @Override protected FileStatus stat(Path path, boolean followSymlinks) throws IOException {
        if (!exists(path, followSymlinks)) {
          throw new IOException("Expected exception on stat of non-existent file");
        }
        return super.stat(path, followSymlinks);
      }
      @Override protected long getLastModifiedTime(Path path, boolean followSymlinks) { return 0; }
    };
  }

  @Test
  public void testExistsCallOnEscapingSymlink() throws Exception {
    TestDelegator delegator = newExistsDelegator();
    scopedFS().setDelegator(delegator);

    delegator.setState(false);
    assertFalseWithPathCheck(fileLink.exists(), delegator, fileLinkTarget);
    assertFalseWithPathCheck(dirLink.getRelative("a").exists(), delegator,
        dirLinkTarget.getRelative("a"));

    delegator.setState(true);
    assertTrueWithPathCheck(fileLink.exists(), delegator, fileLinkTarget);
    assertTrueWithPathCheck(dirLink.getRelative("a").exists(), delegator,
        dirLinkTarget.getRelative("a"));
  }

  @Test
  public void testCreateDirectoryCallOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected boolean createDirectory(Path path) {
        lastPath = path;
        return booleanState();
      }
      @Override protected boolean isDirectory(Path path, boolean followSymlinks) { return true; }
    };
    scopedFS().setDelegator(delegator);

    delegator.setState(false);
    assertFalseWithPathCheck(dirLink.getRelative("a").createDirectory(), delegator,
        dirLinkTarget.getRelative("a"));

    delegator.setState(true);
    assertTrueWithPathCheck(dirLink.getRelative("a").createDirectory(), delegator,
        dirLinkTarget.getRelative("a"));
  }

  @Test
  public void testDeleteCallOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected boolean delete(Path path) {
        lastPath = path;
        return booleanState();
      }
      @Override protected boolean isDirectory(Path path, boolean followSymlinks) { return true; }
      @Override protected long getLastModifiedTime(Path path, boolean followSymlinks) { return 0; }
    };
    scopedFS().setDelegator(delegator);

    delegator.setState(false);
    assertTrue(fileLink.delete());
    assertNull(delegator.lastPath()); // Deleting a link shouldn't require delegation.
    assertFalseWithPathCheck(dirLink.getRelative("a").delete(), delegator,
        dirLinkTarget.getRelative("a"));

    delegator.setState(true);
    assertTrueWithPathCheck(dirLink.getRelative("a").delete(), delegator,
        dirLinkTarget.getRelative("a"));
  }

  @Test
  public void testCallGetFileSizeOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected long getFileSize(Path path, boolean followSymlinks) {
        lastPath = path;
        return longState();
      }
      @Override protected long getLastModifiedTime(Path path, boolean followSymlinks) { return 0; }
    };
    scopedFS().setDelegator(delegator);

    final int state1 = 10;
    delegator.setState(state1);
    assertEquals(state1, fileLink.getFileSize());
    checkPath(delegator, fileLinkTarget);
    assertEquals(state1, dirLink.getRelative("a").getFileSize());
    checkPath(delegator, dirLinkTarget.getRelative("a"));

    final int state2 = 10;
    delegator.setState(state2);
    assertEquals(state2, fileLink.getFileSize());
    checkPath(delegator, fileLinkTarget);
    assertEquals(state2, dirLink.getRelative("a").getFileSize());
    checkPath(delegator, dirLinkTarget.getRelative("a"));
   }

  @Test
  public void testCallGetLastModifiedTimeOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected long getLastModifiedTime(Path path, boolean followSymlinks) {
        lastPath = path;
        return longState();
      }
    };
    scopedFS().setDelegator(delegator);

    final int state1 = 10;
    delegator.setState(state1);
    assertEquals(state1, fileLink.getLastModifiedTime());
    checkPath(delegator, fileLinkTarget);
    assertEquals(state1, dirLink.getRelative("a").getLastModifiedTime());
    checkPath(delegator, dirLinkTarget.getRelative("a"));

    final int state2 = 10;
    delegator.setState(state2);
    assertEquals(state2, fileLink.getLastModifiedTime());
    checkPath(delegator, fileLinkTarget);
    assertEquals(state2, dirLink.getRelative("a").getLastModifiedTime());
    checkPath(delegator, dirLinkTarget.getRelative("a"));
  }

  @Test
  public void testCallSetReadableOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected void setReadable(Path path, boolean readable) {
        lastPath = path;
        setState(readable);
      }
    };
    scopedFS().setDelegator(delegator);

    delegator.setState(false);
    fileLink.setReadable(true);
    assertTrue(delegator.booleanState());
    checkPath(delegator, fileLinkTarget);
    fileLink.setReadable(false);
    assertFalse(delegator.booleanState());
    checkPath(delegator, fileLinkTarget);

    delegator.setState(false);
    dirLink.getRelative("a").setReadable(true);
    assertTrue(delegator.booleanState());
    checkPath(delegator, dirLinkTarget.getRelative("a"));
    dirLink.getRelative("a").setReadable(false);
    assertFalse(delegator.booleanState());
    checkPath(delegator, dirLinkTarget.getRelative("a"));
  }

  @Test
  public void testCallSetWritableOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected void setWritable(Path path, boolean writable) {
        lastPath = path;
        setState(writable);
      }
    };
    scopedFS().setDelegator(delegator);

    delegator.setState(false);
    fileLink.setWritable(true);
    assertTrue(delegator.booleanState());
    checkPath(delegator, fileLinkTarget);
    fileLink.setWritable(false);
    assertFalse(delegator.booleanState());
    checkPath(delegator, fileLinkTarget);

    delegator.setState(false);
    dirLink.getRelative("a").setWritable(true);
    assertTrue(delegator.booleanState());
    checkPath(delegator, dirLinkTarget.getRelative("a"));
    dirLink.getRelative("a").setWritable(false);
    assertFalse(delegator.booleanState());
    checkPath(delegator, dirLinkTarget.getRelative("a"));
  }

  @Test
  public void testCallSetExecutableOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected void setReadable(Path path, boolean readable) {
        lastPath = path;
        setState(readable);
      }
    };
    scopedFS().setDelegator(delegator);

    delegator.setState(false);
    fileLink.setReadable(true);
    assertTrue(delegator.booleanState());
    checkPath(delegator, fileLinkTarget);
    fileLink.setReadable(false);
    assertFalse(delegator.booleanState());
    checkPath(delegator, fileLinkTarget);

    delegator.setState(false);
    dirLink.getRelative("a").setReadable(true);
    assertTrue(delegator.booleanState());
    checkPath(delegator, dirLinkTarget.getRelative("a"));
    dirLink.getRelative("a").setReadable(false);
    assertFalse(delegator.booleanState());
    checkPath(delegator, dirLinkTarget.getRelative("a"));
  }

  @Test
  public void testCallSetLastModifiedTimeOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected void setLastModifiedTime(Path path, long newTime) {
        lastPath = path;
        setState(newTime);
      }
    };
    scopedFS().setDelegator(delegator);

    delegator.setState(0);
    fileLink.setLastModifiedTime(10);
    assertEquals(10, delegator.longState());
    checkPath(delegator, fileLinkTarget);
    fileLink.setLastModifiedTime(15);
    assertEquals(15, delegator.longState());
    checkPath(delegator, fileLinkTarget);

    dirLink.getRelative("a").setLastModifiedTime(20);
    assertEquals(20, delegator.longState());
    checkPath(delegator, dirLinkTarget.getRelative("a"));
    dirLink.getRelative("a").setLastModifiedTime(25);
    assertEquals(25, delegator.longState());
    checkPath(delegator, dirLinkTarget.getRelative("a"));
  }

  @Test
  public void testCallRenameToOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected void renameTo(Path sourcePath, Path targetPath) {
        lastPath = sourcePath;
        setState(targetPath);
      }
      @Override protected boolean isDirectory(Path path, boolean followSymlinks) { return true; }
    };
    scopedFS().setDelegator(delegator);

    // Renaming a link should work fine.
    delegator.setState(null);
    fileLink.renameTo(testFS.getPath(SCOPE_ROOT).getRelative("newname"));
    assertNull(delegator.lastPath()); // Renaming a link shouldn't require delegation.
    assertNull(delegator.objectState());

    // Renaming an out-of-scope path to an in-scope path should fail due to filesystem mismatch
    // errors.
    Path newPath = testFS.getPath(SCOPE_ROOT.getRelative("blah"));
    try {
      dirLink.getRelative("a").renameTo(newPath);
      fail("This is an attempt at a cross-filesystem renaming, which should fail");
    } catch (IOException e) {
      // Expected.
    }

    // Renaming an out-of-scope path to another out-of-scope path can be valid.
    newPath = dirLink.getRelative("b");
    dirLink.getRelative("a").renameTo(newPath);
    assertEquals(dirLinkTarget.getRelative("a"), delegator.lastPath());
    assertEquals(dirLinkTarget.getRelative("b"), ((Path) delegator.objectState()).asFragment());
  }

  @Test
  public void testCallCreateSymbolicLinkOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected void createSymbolicLink(Path linkPath, PathFragment targetFragment) {
        lastPath = linkPath;
        setState(targetFragment);
      }
      @Override protected boolean isDirectory(Path path, boolean followSymlinks) { return true; }
    };
    scopedFS().setDelegator(delegator);

    PathFragment newLinkTarget = new PathFragment("/something/else");
    dirLink.getRelative("a").createSymbolicLink(newLinkTarget);
    assertEquals(dirLinkTarget.getRelative("a"), delegator.lastPath());
    assertSame(newLinkTarget, delegator.objectState());
  }

  @Test
  public void testCallReadSymbolicLinkOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected PathFragment readSymbolicLink(Path path) {
        lastPath = path;
        return (PathFragment) objectState;
      }
      @Override protected boolean isDirectory(Path path, boolean followSymlinks) { return true; }
    };
    scopedFS().setDelegator(delegator);

    // Since we're not following the link, this shouldn't invoke delegation.
    delegator.setState(new PathFragment("whatever"));
    PathFragment p = fileLink.readSymbolicLink();
    assertNull(delegator.lastPath());
    assertNotSame(delegator.objectState(), p);

    // This should.
    p = dirLink.getRelative("a").readSymbolicLink();
    assertEquals(dirLinkTarget.getRelative("a"), delegator.lastPath());
    assertSame(delegator.objectState(), p);
  }

  @Test
  public void testCallGetInputStreamOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected InputStream getInputStream(Path path) {
        lastPath = path;
        return (InputStream) objectState;
      }
    };
    scopedFS().setDelegator(delegator);

    delegator.setState(new ByteArrayInputStream("blah".getBytes()));
    InputStream is = fileLink.getInputStream();
    assertEquals(fileLinkTarget, delegator.lastPath());
    assertSame(delegator.objectState(), is);

    delegator.setState(new ByteArrayInputStream("blah2".getBytes()));
    is = dirLink.getInputStream();
    assertEquals(dirLinkTarget, delegator.lastPath());
    assertSame(delegator.objectState(), is);
  }

  @Test
  public void testCallGetOutputStreamOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected OutputStream getOutputStream(Path path, boolean append)  {
        lastPath = path;
        return (OutputStream) objectState;
      }
      @Override protected boolean isDirectory(Path path, boolean followSymlinks) { return true; }
    };
    scopedFS().setDelegator(delegator);

    delegator.setState(new ByteArrayOutputStream());
    OutputStream os = fileLink.getOutputStream();
    assertEquals(fileLinkTarget, delegator.lastPath());
    assertSame(delegator.objectState(), os);

    delegator.setState(new ByteArrayOutputStream());
    os = dirLink.getOutputStream();
    assertEquals(dirLinkTarget, delegator.lastPath());
    assertSame(delegator.objectState(), os);
  }

  @Test
  public void testCallGetDirectoryEntriesOnEscapingSymlink() throws Exception {
    TestDelegator delegator = new TestDelegator() {
      @Override protected Collection<Path> getDirectoryEntries(Path path) {
        lastPath = path;
        return ImmutableList.of((Path) objectState);
      }
      @Override protected boolean isDirectory(Path path, boolean followSymlinks) { return true; }
    };
    scopedFS().setDelegator(delegator);

    delegator.setState(testFS.getPath("/anything"));
    Collection<Path> entries = dirLink.getDirectoryEntries();
    assertEquals(dirLinkTarget, delegator.lastPath());
    assertThat(entries).hasSize(1);
    assertSame(delegator.objectState(), entries.iterator().next());
  }

  /**
   * Asserts that "link" is an in-scope link that doesn't result in an out-of-FS
   * delegation. If link is relative, its path is relative to SCOPE_ROOT.
   *
   * Note that we don't actually check that the canonicalized target path matches
   * the link's target value. Such testing should be covered by
   * SymlinkAwareFileSystemTest.
   */
  private void assertInScopeLink(String link, String target, TestDelegator d) throws IOException {
    Path l = testFS.getPath(SCOPE_ROOT.getRelative(link));
    testFS.createSymbolicLink(l, new PathFragment(target));
    l.exists();
    assertNull(d.lastPath());
  }

  /**
   * Asserts that "link" is an out-of-scope link and that the re-delegated path
   * matches expectedPath. If link is relative, its path is relative to SCOPE_ROOT.
   */
  private void assertOutOfScopeLink(String link, String target, String expectedPath,
      TestDelegator d) throws IOException {
    Path l = testFS.getPath(SCOPE_ROOT.getRelative(link));
    testFS.createSymbolicLink(l, new PathFragment(target));
    l.exists();
    assertEquals(expectedPath, d.lastPath().getPathString());
  }

  /**
   * Returns the scope root with the final n segments chopped off (or a 0-segment path
   * if n > SCOPE_ROOT.segmentCount()).
   */
  private String chopScopeRoot(int n) {
    return SCOPE_ROOT
        .subFragment(0, n > SCOPE_ROOT.segmentCount() ? 0 : SCOPE_ROOT.segmentCount() - n)
        .getPathString();
  }

  /**
   * Tests that absolute symlinks with ".." and "." segments are delegated to
   * the expected paths.
   */
  @Test
  public void testAbsoluteSymlinksWithParentReferences() throws Exception {
    TestDelegator d = newExistsDelegator();
    scopedFS().setDelegator(d);
    testFS.createDirectory(testFS.getPath(SCOPE_ROOT.getRelative("dir")));
    String scopeRoot = SCOPE_ROOT.getPathString();
    String scopeBase = SCOPE_ROOT.getBaseName();

    // Symlinks that should never escape our scope.
    assertInScopeLink("ilink1", scopeRoot, d);
    assertInScopeLink("ilink2", scopeRoot + "/target", d);
    assertInScopeLink("ilink3", scopeRoot + "/dir/../target", d);
    assertInScopeLink("ilink4", scopeRoot + "/dir/../dir/dir2/../target", d);
    assertInScopeLink("ilink5", scopeRoot + "/./dir/.././target", d);
    assertInScopeLink("ilink6", scopeRoot + "/../" + scopeBase + "/target", d);
    assertInScopeLink("ilink7", "/some/path/../.." + scopeRoot + "/target", d);

    // Symlinks that should escape our scope.
    assertOutOfScopeLink("olink1", scopeRoot + "/../target", chopScopeRoot(1) + "/target", d);
    assertOutOfScopeLink("olink2", "/some/other/path", "/some/other/path", d);
    assertOutOfScopeLink("olink3", scopeRoot + "/../target", chopScopeRoot(1) + "/target", d);
    assertOutOfScopeLink("olink4", chopScopeRoot(1) + "/target", chopScopeRoot(1) + "/target", d);
    assertOutOfScopeLink("olink5", scopeRoot + "/../../../../target", "/target", d);

    // In-scope symlink that's not the final segment in a query.
    Path iDirLink = testFS.getPath(SCOPE_ROOT.getRelative("ilinkdir"));
    testFS.createSymbolicLink(iDirLink, SCOPE_ROOT.getRelative("dir"));
    iDirLink.getRelative("file").exists();
    assertNull(d.lastPath());

    // Out-of-scope symlink that's not the final segment in a query.
    Path oDirLink = testFS.getPath(SCOPE_ROOT.getRelative("olinkdir"));
    testFS.createSymbolicLink(oDirLink, new PathFragment("/some/other/dir"));
    oDirLink.getRelative("file").exists();
    assertEquals("/some/other/dir/file", d.lastPath().getPathString());
  }

  /**
   * Tests that relative symlinks with ".." and "." segments are delegated to
   * the expected paths.
   */
  @Test
  public void testRelativeSymlinksWithParentReferences() throws Exception {
    TestDelegator d = newExistsDelegator();
    scopedFS().setDelegator(d);
    testFS.createDirectory(testFS.getPath(SCOPE_ROOT.getRelative("dir")));
    testFS.createDirectory(testFS.getPath(SCOPE_ROOT.getRelative("dir/dir2")));
    testFS.createDirectory(testFS.getPath(SCOPE_ROOT.getRelative("dir/dir2/dir3")));
    String scopeRoot = SCOPE_ROOT.getPathString();
    String scopeBase = SCOPE_ROOT.getBaseName();

    // Symlinks that should never escape our scope.
    assertInScopeLink("ilink1", "target", d);
    assertInScopeLink("ilink2", "dir/../otherdir/target", d);
    assertInScopeLink("dir/ilink3", "../target", d);
    assertInScopeLink("dir/dir2/ilink4", "../../target", d);
    assertInScopeLink("dir/dir2/ilink5", ".././../dir/./target", d);
    assertInScopeLink("dir/dir2/ilink6", "../dir2/../../dir/dir2/dir3/../../../target", d);

    // Symlinks that should escape our scope.
    assertOutOfScopeLink("olink1", "../target", chopScopeRoot(1) + "/target", d);
    assertOutOfScopeLink("dir/olink2", "../../target", chopScopeRoot(1) + "/target", d);
    assertOutOfScopeLink("olink3", "../" + scopeBase + "/target", scopeRoot + "/target", d);
    assertOutOfScopeLink("dir/dir2/olink5", "../../../target", chopScopeRoot(1) + "/target", d);
    assertOutOfScopeLink("dir/dir2/olink6", "../dir2/../../dir/dir2/../../../target",
        chopScopeRoot(1) + "/target", d);
    assertOutOfScopeLink("dir/olink7", "../../../target", chopScopeRoot(2) + "target", d);
    assertOutOfScopeLink("olink8", "../../../../../target", "/target", d);

    // In-scope symlink that's not the final segment in a query.
    Path iDirLink = testFS.getPath(SCOPE_ROOT.getRelative("dir/dir2/ilinkdir"));
    testFS.createSymbolicLink(iDirLink, new PathFragment("../../dir"));
    iDirLink.getRelative("file").exists();
    assertNull(d.lastPath());

    // Out-of-scope symlink that's not the final segment in a query.
    Path oDirLink = testFS.getPath(SCOPE_ROOT.getRelative("dir/dir2/olinkdir"));
    testFS.createSymbolicLink(oDirLink, new PathFragment("../../../other/dir"));
    oDirLink.getRelative("file").exists();
    assertEquals(chopScopeRoot(1) + "/other/dir/file", d.lastPath().getPathString());
  }
}
