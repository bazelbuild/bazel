// Copyright 2026 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.channels.SeekableByteChannel;
import java.util.Collection;
import javax.annotation.Nullable;
import org.junit.Test;

/**
 * Tests for {@link OverlayFileSystem}.
 *
 * <p>The tests inherited from {@link SymlinkAwareFileSystemTest} and ancestors exercise the
 * canonicalize-first machinery over a single underlying source; the tests defined here exercise
 * symlink chains that cross sources.
 */
public final class OverlayFileSystemTest extends SymlinkAwareFileSystemTest {

  private static final PathFragment UPPER_PREFIX = PathFragment.create("/upper");

  /**
   * A minimal overlay that routes paths under {@link #UPPER_PREFIX} to one in-memory file system
   * and all other paths to another.
   */
  private static class RoutedOverlayFileSystem extends OverlayFileSystem {
    final InMemoryFileSystem upperFs;
    final InMemoryFileSystem lowerFs;

    RoutedOverlayFileSystem(DigestHashFunction digestHashFunction) {
      super(digestHashFunction);
      this.upperFs = new InMemoryFileSystem(BlazeClock.instance(), digestHashFunction);
      this.lowerFs = new InMemoryFileSystem(BlazeClock.instance(), digestHashFunction);
    }

    private FileSystem sourceFor(PathFragment path) {
      return path.startsWith(UPPER_PREFIX) ? upperFs : lowerFs;
    }

    @Override
    @Nullable
    protected FileStatus statNofollow(PathFragment path) throws IOException {
      return sourceFor(path).statIfFound(path, /* followSymlinks= */ false);
    }

    @Override
    protected PathFragment readSymlinkNofollow(PathFragment path) throws IOException {
      return sourceFor(path).readSymbolicLink(path);
    }

    @Override
    protected Collection<Dirent> readdirNofollow(PathFragment path) throws IOException {
      return sourceFor(path).readdir(path, /* followSymlinks= */ false);
    }

    @Override
    protected byte[] getDigestNofollow(PathFragment path) throws IOException {
      return sourceFor(path).getDigest(path);
    }

    @Override
    @Nullable
    protected byte[] getFastDigestNofollow(PathFragment path) throws IOException {
      return sourceFor(path).getFastDigest(path);
    }

    @Override
    protected boolean deleteNofollow(PathFragment path) throws IOException {
      return sourceFor(path).delete(path);
    }

    @Override
    protected void renameToNofollow(PathFragment sourcePath, PathFragment targetPath)
        throws IOException {
      sourceFor(sourcePath).renameTo(sourcePath, targetPath);
    }

    @Override
    protected void createSymbolicLinkNofollow(
        PathFragment linkPath, PathFragment targetFragment, SymlinkTargetType type)
        throws IOException {
      sourceFor(linkPath).createSymbolicLink(linkPath, targetFragment, type);
    }

    @Override
    protected void setLastModifiedTimeNofollow(PathFragment path, long newTime)
        throws IOException {
      sourceFor(path).setLastModifiedTime(path, newTime);
    }

    @Override
    protected boolean isReadableNofollow(PathFragment path) throws IOException {
      return sourceFor(path).isReadable(path);
    }

    @Override
    protected boolean isWritableNofollow(PathFragment path) throws IOException {
      return sourceFor(path).isWritable(path);
    }

    @Override
    protected boolean isExecutableNofollow(PathFragment path) throws IOException {
      return sourceFor(path).isExecutable(path);
    }

    @Override
    protected void setReadableNofollow(PathFragment path, boolean readable) throws IOException {
      sourceFor(path).setReadable(path, readable);
    }

    @Override
    protected void setWritableNofollow(PathFragment path, boolean writable) throws IOException {
      sourceFor(path).setWritable(path, writable);
    }

    @Override
    protected void setExecutableNofollow(PathFragment path, boolean executable)
        throws IOException {
      sourceFor(path).setExecutable(path, executable);
    }

    @Override
    protected void chmodNofollow(PathFragment path, int mode) throws IOException {
      sourceFor(path).chmod(path, mode);
    }

    @Override
    public InputStream getInputStream(PathFragment path) throws IOException {
      PathFragment resolvedPath = canonicalize(path);
      return sourceFor(resolvedPath).getInputStream(resolvedPath);
    }

    @Override
    public OutputStream getOutputStream(PathFragment path, boolean append, boolean internal)
        throws IOException {
      PathFragment resolvedPath = canonicalizeParent(path);
      return sourceFor(resolvedPath).getOutputStream(resolvedPath, append, internal);
    }

    @Override
    public SeekableByteChannel createReadWriteByteChannel(PathFragment path) throws IOException {
      PathFragment resolvedPath = canonicalizeParent(path);
      return sourceFor(resolvedPath).createReadWriteByteChannel(resolvedPath);
    }

    @Override
    public boolean createDirectory(PathFragment path) throws IOException {
      return sourceFor(path).createDirectory(path);
    }

    @Override
    public void createDirectoryAndParents(PathFragment path) throws IOException {
      sourceFor(path).createDirectoryAndParents(path);
    }

    @Override
    public void createFSDependentHardLink(PathFragment linkPath, PathFragment originalPath)
        throws IOException {
      sourceFor(linkPath).createFSDependentHardLink(linkPath, originalPath);
    }

    @Override
    public boolean supportsModifications(PathFragment path) {
      return sourceFor(path).supportsModifications(path);
    }

    @Override
    public boolean supportsSymbolicLinksNatively(PathFragment path) {
      return sourceFor(path).supportsSymbolicLinksNatively(path);
    }

    @Override
    public boolean supportsHardLinksNatively(PathFragment path) {
      return sourceFor(path).supportsHardLinksNatively(path);
    }

    @Override
    public boolean mayBeCaseOrNormalizationInsensitive() {
      return false;
    }
  }

  /**
   * A variant of {@link RoutedOverlayFileSystem} that only overlays paths under {@link
   * #UPPER_PREFIX} and delegates all other paths wholesale to the lower file system.
   */
  private static final class ScopedOverlayFileSystem extends RoutedOverlayFileSystem {
    ScopedOverlayFileSystem(DigestHashFunction digestHashFunction) {
      super(digestHashFunction);
    }

    @Override
    @Nullable
    protected FileSystem wholesaleDelegate(PathFragment path) {
      return path.startsWith(UPPER_PREFIX) ? null : lowerFs;
    }
  }

  @Override
  public FileSystem getFreshFileSystem(DigestHashFunction digestHashFunction) {
    return new RoutedOverlayFileSystem(digestHashFunction);
  }

  @Override
  public void destroyFileSystem(FileSystem fileSystem) {}

  private Path upperPath(String relativePath) throws IOException {
    Path path = testFS.getPath(UPPER_PREFIX.getRelative(relativePath));
    path.getParentDirectory().createDirectoryAndParents();
    return path;
  }

  private Path lowerFile(String relativePath, String content) throws IOException {
    Path path = absolutize(relativePath);
    FileSystemUtils.writeContent(path, UTF_8, content);
    return path;
  }

  @Test
  public void crossSourceSymlink_upperToLower_followed() throws Exception {
    Path target = lowerFile("target", "hello");
    Path link = upperPath("link");
    link.createSymbolicLink(target.asFragment());

    assertThat(link.isFile(Symlinks.FOLLOW)).isTrue();
    assertThat(link.stat(Symlinks.FOLLOW).getSize()).isEqualTo(5);
    assertThat(link.getFileSize()).isEqualTo(5);
    assertThat(FileSystemUtils.readContent(link, UTF_8)).isEqualTo("hello");
    assertThat(link.getDigest()).isEqualTo(target.getDigest());
    assertThat(link.resolveSymbolicLinks()).isEqualTo(target);
  }

  @Test
  public void crossSourceSymlink_lowerToUpper_followed() throws Exception {
    Path target = upperPath("target");
    FileSystemUtils.writeContent(target, UTF_8, "hello");
    Path link = absolutize("link");
    link.createSymbolicLink(target.asFragment());

    assertThat(link.isFile(Symlinks.FOLLOW)).isTrue();
    assertThat(FileSystemUtils.readContent(link, UTF_8)).isEqualTo("hello");
    assertThat(link.resolveSymbolicLinks()).isEqualTo(target);
  }

  @Test
  public void crossSourceSymlink_chainAcrossSources_followed() throws Exception {
    Path target = lowerFile("target", "hello");
    Path upperLink = upperPath("upperLink");
    upperLink.createSymbolicLink(target.asFragment());
    Path lowerLink = absolutize("lowerLink");
    lowerLink.createSymbolicLink(upperLink.asFragment());

    assertThat(lowerLink.resolveSymbolicLinks()).isEqualTo(target);
    assertThat(FileSystemUtils.readContent(lowerLink, UTF_8)).isEqualTo("hello");
  }

  @Test
  public void crossSourceSymlink_relativeTarget_followed() throws Exception {
    Path target = lowerFile("target", "hello");
    Path link = upperPath("dir/link");
    // Walks out of the upper prefix through uplevel references.
    link.createSymbolicLink(
        PathFragment.create("../..").getRelative(target.asFragment().toRelative()));

    assertThat(link.resolveSymbolicLinks()).isEqualTo(target);
    assertThat(FileSystemUtils.readContent(link, UTF_8)).isEqualTo("hello");
  }

  @Test
  public void crossSourceSymlink_intermediateDirectorySymlink_followed() throws Exception {
    Path targetDir = absolutize("targetDir");
    targetDir.createDirectoryAndParents();
    FileSystemUtils.writeContent(targetDir.getChild("file"), UTF_8, "hello");
    Path dirLink = upperPath("dirLink");
    dirLink.createSymbolicLink(targetDir.asFragment());

    Path fileThroughLink = dirLink.getChild("file");
    assertThat(fileThroughLink.isFile(Symlinks.FOLLOW)).isTrue();
    assertThat(fileThroughLink.resolveSymbolicLinks()).isEqualTo(targetDir.getChild("file"));
    assertThat(FileSystemUtils.readContent(fileThroughLink, UTF_8)).isEqualTo("hello");
  }

  @Test
  public void crossSourceSymlink_dangling_behavesLikeDanglingLink() throws Exception {
    Path link = upperPath("link");
    Path missing = absolutize("missing");
    link.createSymbolicLink(missing.asFragment());

    assertThat(link.isSymbolicLink()).isTrue();
    assertThat(link.exists(Symlinks.FOLLOW)).isFalse();
    assertThat(link.statIfFound(Symlinks.FOLLOW)).isNull();
    assertThat(link.readSymbolicLink()).isEqualTo(missing.asFragment());
    assertThrows(FileNotFoundException.class, link::resolveSymbolicLinks);
  }

  @Test
  public void crossSourceSymlink_loopAcrossSources_throws() throws Exception {
    Path upperLink = upperPath("link");
    Path lowerLink = absolutize("link");
    upperLink.createSymbolicLink(lowerLink.asFragment());
    lowerLink.createSymbolicLink(upperLink.asFragment());

    assertThrows(FileSymlinkLoopException.class, upperLink::resolveSymbolicLinks);
    assertThrows(FileSymlinkLoopException.class, () -> upperLink.stat(Symlinks.FOLLOW));
    assertThat(upperLink.statNullable(Symlinks.FOLLOW)).isNull();
  }

  @Test
  public void crossSourceSymlink_readdirFollowClassifiesTargetType() throws Exception {
    Path target = lowerFile("target", "hello");
    Path dir = upperPath("dir");
    dir.createDirectoryAndParents();
    dir.getChild("link").createSymbolicLink(target.asFragment());

    assertThat(dir.readdir(Symlinks.NOFOLLOW))
        .containsExactly(new Dirent("link", Dirent.Type.SYMLINK));
    assertThat(dir.readdir(Symlinks.FOLLOW)).containsExactly(new Dirent("link", Dirent.Type.FILE));
  }

  @Test
  public void crossSourceSymlink_resolutionReflectsDeletionAndRecreation() throws Exception {
    Path target1 = lowerFile("target1", "first");
    Path target2 = lowerFile("target2", "second");
    Path link = upperPath("link");
    link.createSymbolicLink(target1.asFragment());
    assertThat(link.resolveSymbolicLinks()).isEqualTo(target1);

    link.delete();
    link.createSymbolicLink(target2.asFragment());
    assertThat(link.resolveSymbolicLinks()).isEqualTo(target2);
    assertThat(FileSystemUtils.readContent(link, UTF_8)).isEqualTo("second");
  }

  @Test
  public void deleteThroughParentSymlink_invalidatesResolvedLocation() throws Exception {
    Path dir = absolutize("dir");
    dir.createDirectoryAndParents();
    Path dirAlias = absolutize("dirAlias");
    dirAlias.createSymbolicLink(dir.asFragment());
    Path target1 = lowerFile("target1", "first");
    Path target2 = lowerFile("target2", "second");
    Path link = dir.getChild("link");
    link.createSymbolicLink(target1.asFragment());
    // Populate the canonicalization cache for the resolved location of the link.
    assertThat(link.resolveSymbolicLinks()).isEqualTo(target1);

    // Delete the link through the alias and recreate it with a different target.
    assertThat(dirAlias.getChild("link").delete()).isTrue();
    assertThat(link.exists(Symlinks.NOFOLLOW)).isFalse();
    link.createSymbolicLink(target2.asFragment());

    assertThat(link.resolveSymbolicLinks()).isEqualTo(target2);
    assertThat(FileSystemUtils.readContent(link, UTF_8)).isEqualTo("second");
  }

  @Test
  public void crossSourceSymlink_resolutionReflectsRename() throws Exception {
    Path target = lowerFile("target", "hello");
    Path link1 = upperPath("link1");
    link1.createSymbolicLink(target.asFragment());
    Path link2 = upperPath("link2");
    assertThat(link1.resolveSymbolicLinks()).isEqualTo(target);

    link1.renameTo(link2);
    assertThat(link1.exists(Symlinks.NOFOLLOW)).isFalse();
    assertThat(link2.resolveSymbolicLinks()).isEqualTo(target);
  }

  @Test
  public void wholesaleDelegate_scopedSymlinkIntoDelegate_followed() throws Exception {
    var scopedFs = new ScopedOverlayFileSystem(digestHashFunction);
    Path target = scopedFs.getPath("/work/target");
    target.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContent(target, UTF_8, "hello");
    Path link = scopedFs.getPath("/upper/link");
    link.getParentDirectory().createDirectoryAndParents();
    link.createSymbolicLink(target.asFragment());

    assertThat(link.isFile(Symlinks.FOLLOW)).isTrue();
    assertThat(FileSystemUtils.readContent(link, UTF_8)).isEqualTo("hello");
    Path resolved = link.resolveSymbolicLinks();
    assertThat(resolved).isEqualTo(target);
    assertThat(resolved.getFileSystem()).isSameInstanceAs(scopedFs);
  }

  @Test
  public void wholesaleDelegate_resolveSymbolicLinksDoesNotLeaveOverlay() throws Exception {
    var scopedFs = new ScopedOverlayFileSystem(digestHashFunction);
    Path target = scopedFs.getPath("/work/target");
    target.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.createEmptyFile(target);
    Path link = scopedFs.getPath("/work/link");
    link.createSymbolicLink(target.asFragment());

    Path resolved = link.resolveSymbolicLinks();
    assertThat(resolved).isEqualTo(target);
    assertThat(resolved.getFileSystem()).isSameInstanceAs(scopedFs);
  }

  @Test
  public void wholesaleDelegate_mutationInDelegatedTerritory_invalidatesEscapedChainCache()
      throws Exception {
    var scopedFs = new ScopedOverlayFileSystem(digestHashFunction);
    Path target1 = scopedFs.getPath("/work/sub/target");
    target1.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContent(target1, UTF_8, "first");
    Path link = scopedFs.getPath("/upper/link");
    link.getParentDirectory().createDirectoryAndParents();
    link.createSymbolicLink(target1.asFragment());
    // Populate the canonicalization cache with the escaped part of the chain.
    assertThat(link.resolveSymbolicLinks()).isEqualTo(target1);

    // Replace the target's parent directory with a symlink through wholesale-delegated
    // operations.
    Path newDir = scopedFs.getPath("/work/newSub");
    newDir.createDirectoryAndParents();
    FileSystemUtils.writeContent(newDir.getChild("target"), UTF_8, "second");
    target1.delete();
    target1.getParentDirectory().delete();
    scopedFs.getPath("/work/sub").createSymbolicLink(newDir.asFragment());

    assertThat(link.resolveSymbolicLinks()).isEqualTo(newDir.getChild("target"));
    assertThat(FileSystemUtils.readContent(link, UTF_8)).isEqualTo("second");
  }

  @Test
  public void wholesaleDelegate_delegateSymlinkIntoScope_resolvedByDelegateOnly()
      throws Exception {
    var scopedFs = new ScopedOverlayFileSystem(digestHashFunction);
    Path target = scopedFs.getPath("/upper/target");
    target.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.createEmptyFile(target);
    Path link = scopedFs.getPath("/work/link");
    link.getParentDirectory().createDirectoryAndParents();
    link.createSymbolicLink(target.asFragment());

    // Symlink chains starting outside the overlay scope are resolved entirely by the delegate,
    // which cannot see into the overlay.
    assertThat(link.exists(Symlinks.FOLLOW)).isFalse();
  }
}
