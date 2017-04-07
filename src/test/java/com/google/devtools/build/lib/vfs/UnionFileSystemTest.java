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
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.OutputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for the UnionFileSystem, both of generic FileSystem functionality
 * (inherited) and tests of UnionFileSystem-specific behavior.
 */
@RunWith(JUnit4.class)
public class UnionFileSystemTest extends SymlinkAwareFileSystemTest {
  private XAttrInMemoryFs inDelegate;
  private XAttrInMemoryFs outDelegate;
  private XAttrInMemoryFs defaultDelegate;
  private UnionFileSystem unionfs;

  private static final String XATTR_VAL = "SOME_XATTR_VAL";
  private static final String XATTR_KEY = "SOME_XATTR_KEY";

  private void setupDelegateFileSystems() {
    inDelegate = new XAttrInMemoryFs(BlazeClock.instance());
    outDelegate = new XAttrInMemoryFs(BlazeClock.instance());
    defaultDelegate = new XAttrInMemoryFs(BlazeClock.instance());

    unionfs = createDefaultUnionFileSystem();
  }

  private UnionFileSystem createDefaultUnionFileSystem() {
    return createDefaultUnionFileSystem(false);
  }

  private UnionFileSystem createDefaultUnionFileSystem(boolean readOnly) {
    return new UnionFileSystem(ImmutableMap.<PathFragment, FileSystem>of(
        PathFragment.create("/in"), inDelegate,
        PathFragment.create("/out"), outDelegate),
        defaultDelegate, readOnly);
  }

  @Override
  protected FileSystem getFreshFileSystem() {
    // Executed with each new test because it is called by super.setUp().
    setupDelegateFileSystems();
    return unionfs;
  }

  @Override
  public void destroyFileSystem(FileSystem fileSystem) {
    // Nothing.
  }

  // Tests of UnionFileSystem-specific behavior below.

  @Test
  public void testBasicDelegation() throws Exception {
    unionfs = createDefaultUnionFileSystem();
    Path fooPath = unionfs.getPath("/foo");
    Path inPath = unionfs.getPath("/in");
    Path outPath = unionfs.getPath("/out/in.txt");
    assertSame(inDelegate, unionfs.getDelegate(inPath));
    assertSame(outDelegate, unionfs.getDelegate(outPath));
    assertSame(defaultDelegate, unionfs.getDelegate(fooPath));
  }

  @Test
  public void testBasicXattr() throws Exception {
    Path fooPath = unionfs.getPath("/foo");
    Path inPath = unionfs.getPath("/in");
    Path outPath = unionfs.getPath("/out/in.txt");

    assertArrayEquals(XATTR_VAL.getBytes(UTF_8), inPath.getxattr(XATTR_KEY));
    assertArrayEquals(XATTR_VAL.getBytes(UTF_8), outPath.getxattr(XATTR_KEY));
    assertArrayEquals(XATTR_VAL.getBytes(UTF_8), fooPath.getxattr(XATTR_KEY));
    assertNull(inPath.getxattr("not_key"));
    assertNull(outPath.getxattr("not_key"));
    assertNull(fooPath.getxattr("not_key"));
  }

  @Test
  public void testDefaultFileSystemRequired() throws Exception {
    try {
      new UnionFileSystem(ImmutableMap.<PathFragment, FileSystem>of(), null);
      fail("Able to create a UnionFileSystem with no default!");
    } catch (NullPointerException expected) {
      // OK - should fail in this case.
    }
  }

  // Check for appropriate registration and lookup of delegate filesystems based
  // on path prefixes, including non-canonical paths.
  @Test
  public void testPrefixDelegation() throws Exception {
    unionfs = new UnionFileSystem(ImmutableMap.<PathFragment, FileSystem>of(
              PathFragment.create("/foo"), inDelegate,
              PathFragment.create("/foo/bar"), outDelegate), defaultDelegate);

    assertSame(inDelegate, unionfs.getDelegate(unionfs.getPath("/foo/foo.txt")));
    assertSame(outDelegate, unionfs.getDelegate(unionfs.getPath("/foo/bar/foo.txt")));
    assertSame(inDelegate, unionfs.getDelegate(unionfs.getPath("/foo/bar/../foo.txt")));
    assertSame(defaultDelegate, unionfs.getDelegate(unionfs.getPath("/bar/foo.txt")));
    assertSame(defaultDelegate, unionfs.getDelegate(unionfs.getPath("/foo/bar/../..")));
  }

  // Checks that files cannot be modified when the filesystem is created
  // read-only, even if the delegate filesystems are read/write.
  @Test
  public void testModificationFlag() throws Exception {
    assertTrue(unionfs.supportsModifications());
    Path outPath = unionfs.getPath("/out/foo.txt");
    assertTrue(unionfs.createDirectory(outPath.getParentDirectory()));
    OutputStream outFile = unionfs.getOutputStream(outPath);
    outFile.write('b');
    outFile.close();

    unionfs.setExecutable(outPath, true);

    // Note that this does not destroy the underlying filesystems;
    // UnionFileSystem is just a view.
    unionfs = createDefaultUnionFileSystem(true);
    assertFalse(unionfs.supportsModifications());

    InputStream outFileInput = unionfs.getInputStream(outPath);
    int outFileByte = outFileInput.read();
    outFileInput.close();
    assertEquals('b', outFileByte);

    assertTrue(unionfs.isExecutable(outPath));

    // Modifying files through the unionfs isn't permitted, even if the
    // delegates are read/write.
    try {
      unionfs.setExecutable(outPath, false);
      fail("Modification to a read-only UnionFileSystem succeeded.");
    } catch (UnsupportedOperationException expected) {
      // OK - should fail.
    }
  }

  // Checks that roots of delegate filesystems are created outside of the
  // delegate filesystems; i.e. they can be seen from the filesystem of the parent.
  @Test
  public void testDelegateRootDirectoryCreation() throws Exception {
    Path foo = unionfs.getPath("/foo");
    Path bar = unionfs.getPath("/bar");
    Path out = unionfs.getPath("/out");
    assertTrue(unionfs.createDirectory(foo));
    assertTrue(unionfs.createDirectory(bar));
    assertTrue(unionfs.createDirectory(out));
    Path outFile = unionfs.getPath("/out/in");
    FileSystemUtils.writeContentAsLatin1(outFile, "Out");

    // FileSystemTest.setUp() silently creates the test root on the filesystem...
    Path testDirUnderRoot = unionfs.getPath(workingDir.asFragment().subFragment(0, 1));
    assertThat(unionfs.getDirectoryEntries(unionfs.getRootDirectory())).containsExactly(foo, bar,
        out, testDirUnderRoot);
    assertThat(unionfs.getDirectoryEntries(out)).containsExactly(outFile);

    assertSame(unionfs.getDelegate(foo), defaultDelegate);
    assertEquals(foo.asFragment(), unionfs.adjustPath(foo, defaultDelegate).asFragment());
    assertSame(unionfs.getDelegate(bar), defaultDelegate);
    assertSame(unionfs.getDelegate(outFile), outDelegate);
    assertSame(unionfs.getDelegate(out), outDelegate);

    // As a fragment (i.e. without filesystem or root info), the path name should be preserved.
    assertEquals(outFile.asFragment(), unionfs.adjustPath(outFile, outDelegate).asFragment());
  }

  // Ensure that the right filesystem is still chosen when paths contain "..".
  @Test
  public void testDelegationOfUpLevelReferences() throws Exception {
    assertSame(defaultDelegate, unionfs.getDelegate(unionfs.getPath("/in/../foo.txt")));
    assertSame(inDelegate, unionfs.getDelegate(unionfs.getPath("/out/../in")));
    assertSame(outDelegate, unionfs.getDelegate(unionfs.getPath("/out/../in/../out/foo.txt")));
    assertSame(inDelegate, unionfs.getDelegate(unionfs.getPath("/in/./foo.txt")));
  }

  // Basic *explicit* cross-filesystem symlink check.
  // Note: This does not work implicitly yet, as the next test illustrates.
  @Test
  public void testCrossDeviceSymlinks() throws Exception {
    assertTrue(unionfs.createDirectory(unionfs.getPath("/out")));

    // Create an "/in" directory directly on the output delegate to bypass the
    // UnionFileSystem's mapping.
    assertTrue(inDelegate.getPath("/in").createDirectory());
    OutputStream outStream = inDelegate.getPath("/in/bar.txt").getOutputStream();
    outStream.write('i');
    outStream.close();

    Path outFoo = unionfs.getPath("/out/foo");
    unionfs.createSymbolicLink(outFoo, PathFragment.create("../in/bar.txt"));
    assertTrue(unionfs.stat(outFoo, false).isSymbolicLink());

    try {
      unionfs.stat(outFoo, true).isFile();
      fail("Stat on cross-device symlink succeeded!");
    } catch (FileNotFoundException expected) {
      // OK
    }

    Path resolved = unionfs.resolveSymbolicLinks(outFoo);
    assertSame(unionfs, resolved.getFileSystem());
    InputStream barInput = resolved.getInputStream();
    int barChar = barInput.read();
    barInput.close();
    assertEquals('i', barChar);
  }

  @Test
  public void testNoDelegateLeakage() throws Exception {
    assertSame(unionfs, unionfs.getPath("/in/foo.txt").getFileSystem());
    assertSame(unionfs, unionfs.getPath("/in/foo/bar").getParentDirectory().getFileSystem());
    unionfs.createDirectory(unionfs.getPath("/out"));
    unionfs.createDirectory(unionfs.getPath("/out/foo"));
    unionfs.createDirectory(unionfs.getPath("/out/foo/bar"));
    assertSame(unionfs, Iterables.getOnlyElement(unionfs.getDirectoryEntries(
        unionfs.getPath("/out/foo"))).getParentDirectory().getFileSystem());
  }

  // Prefix mappings can apply to files starting with a prefix within a directory.
  @Test
  public void testWithinDirectoryMapping() throws Exception {
    unionfs = new UnionFileSystem(ImmutableMap.<PathFragment, FileSystem>of(
        PathFragment.create("/fruit/a"), inDelegate,
        PathFragment.create("/fruit/b"), outDelegate), defaultDelegate);
    assertTrue(unionfs.createDirectory(unionfs.getPath("/fruit")));
    assertTrue(defaultDelegate.getPath("/fruit").isDirectory());
    assertTrue(inDelegate.getPath("/fruit").createDirectory());
    assertTrue(outDelegate.getPath("/fruit").createDirectory());

    Path apple = unionfs.getPath("/fruit/apple");
    Path banana = unionfs.getPath("/fruit/banana");
    Path cherry = unionfs.getPath("/fruit/cherry");
    unionfs.createDirectory(apple);
    unionfs.createDirectory(banana);
    assertSame(inDelegate, unionfs.getDelegate(apple));
    assertSame(outDelegate, unionfs.getDelegate(banana));
    assertSame(defaultDelegate, unionfs.getDelegate(cherry));

    FileSystemUtils.writeContentAsLatin1(apple.getRelative("table"), "penny");
    FileSystemUtils.writeContentAsLatin1(banana.getRelative("nana"), "nanana");
    FileSystemUtils.writeContentAsLatin1(cherry, "garcia");

    assertEquals("penny", new String(
        FileSystemUtils.readContentAsLatin1(inDelegate.getPath("/fruit/apple/table"))));
    assertEquals("nanana", new String(
        FileSystemUtils.readContentAsLatin1(outDelegate.getPath("/fruit/banana/nana"))));
    assertEquals("garcia", new String(
        FileSystemUtils.readContentAsLatin1(defaultDelegate.getPath("/fruit/cherry"))));
  }

  // Write using the VFS through a UnionFileSystem and check that the file can
  // be read back in the same location using standard Java IO.
  // There is a similar test in UnixFileSystem, but this is essential to ensure
  // that paths aren't being remapped in some nasty way on the underlying FS.
  @Test
  public void testDelegateOperationsReflectOnLocalFilesystem() throws Exception {
    unionfs = new UnionFileSystem(ImmutableMap.<PathFragment, FileSystem>of(
        workingDir.getParentDirectory().asFragment(), new UnixFileSystem()),
        defaultDelegate, false);
    // This is a child of the current tmpdir, and doesn't exist on its own.
    // It would be created in setup(), but of course, that didn't use a UnixFileSystem.
    unionfs.createDirectory(workingDir);
    Path testFile = unionfs.getPath(workingDir.getRelative("test_file").asFragment());
    assertTrue(testFile.asFragment().startsWith(workingDir.asFragment()));
    String testString = "This is a test file";
    FileSystemUtils.writeContentAsLatin1(testFile, testString);
    try {
      assertEquals(testString, new String(FileSystemUtils.readContentAsLatin1(testFile)));
    } finally {
      testFile.delete();
      assertTrue(unionfs.delete(workingDir));
    }
  }

  // Regression test for [UnionFS: Directory creation across mapping fails.]
  @Test
  public void testCreateParentsAcrossMapping() throws Exception {
    unionfs = new UnionFileSystem(ImmutableMap.<PathFragment, FileSystem>of(
        PathFragment.create("/out/dir"), outDelegate), defaultDelegate, false);
    Path outDir = unionfs.getPath("/out/dir/biz/bang");
    FileSystemUtils.createDirectoryAndParents(outDir);
    assertTrue(outDir.isDirectory());
  }

  private static class XAttrInMemoryFs extends InMemoryFileSystem {
    public XAttrInMemoryFs(Clock clock) {
      super(clock);
    }

    @Override
    protected byte[] getxattr(Path path, String name) {
      assertSame(this, path.getFileSystem());
      return (name.equals(XATTR_KEY)) ? XATTR_VAL.getBytes(UTF_8) : null;
    }
  }
}
