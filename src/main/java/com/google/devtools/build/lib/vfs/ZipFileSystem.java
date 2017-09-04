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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path.PathFactory;
import java.io.Closeable;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.logging.Logger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * A FileSystem that provides a read-only filesystem view on a zip file.
 * Inherits the constraints imposed by ReadonlyFileSystem.
 */
@ThreadCompatible  // Can only be accessed from one thread at a time (including its Path objects)
public class ZipFileSystem extends ReadonlyFileSystem implements Closeable {
  private static final Logger logger = Logger.getLogger(ZipFileSystem.class.getName());

  private final File tempFile;  // In case this needs to be written to the file system
  private final ZipFile zipFile;
  private boolean open;

  /**
   * The sole purpose of this field is to hold a strong reference to all leaf
   * {@link Path}s which have a non-null "entry" field, preventing them from
   * being garbage-collected.  (The leaf paths hold string references to their
   * parents, so we don't need to include them here.)
   *
   * <p>This is necessary because {@link Path}s may be recycled when they
   * become unreachable, but the ZipFileSystem uses them to hold the {@link
   * ZipEntry} for that path, if any.  Without this additional strong
   * reference, ZipEntries would seem to "disappear" during garbage collection.
   */
  @SuppressWarnings("unused")
  private final Object paths;

  /**
   * Constructs a ZipFileSystem from a zip file identified with a given path.
   */
  public ZipFileSystem(Path zipPath) throws IOException {
    if (!zipPath.exists()) {
      throw new FileNotFoundException(String.format("File '%s' does not exist", zipPath));
    }

    File file = zipPath.getPathFile();
    if (!file.exists()) {
      // If the File says that it does not exist but the Path says that it does, we are probably
      // dealing with a FileSystem that does not represent the actual file system we are running
      // under. Then we copy the Path into a temporary File.
      tempFile = File.createTempFile("bazel.test.", ".tmp");
      file = tempFile;
      byte[] contents = FileSystemUtils.readContent(zipPath);
      try (OutputStream os = new FileOutputStream(tempFile)) {
        os.write(contents);
      }
    } else {
      tempFile = null;
    }

    if (!file.isFile()) {
      throw new IOException(String.format("'%s' is not a file", zipPath));
    }
    if (!file.canRead()) {
      throw new IOException(String.format("File '%s' is not readable", zipPath));
    }

    this.zipFile = new ZipFile(file);
    this.paths = populatePathTree();
    this.open = true;
  }

  // ZipPath extends Path with a set-once ZipEntry field.
  // TODO(bazel-team): (2009) Delete class ZipPath, and perform the
  // Path-to-ZipEntry lookup in {@link #zipEntry} and {@link
  // #getDirectoryEntries}.  Then this field becomes redundant.
  @ThreadSafe
  private static class ZipPath extends Path {

    private enum Factory implements PathFactory {
      INSTANCE {
        @Override
        public Path createRootPath(FileSystem filesystem) {
          Preconditions.checkArgument(filesystem instanceof ZipFileSystem);
          return new ZipPath((ZipFileSystem) filesystem);
        }

        @Override
        public Path createChildPath(Path parent, String childName) {
          Preconditions.checkState(parent instanceof ZipPath);
          return new ZipPath((ZipFileSystem) parent.getFileSystem(), childName, (ZipPath) parent);
        }

        @Override
        public Path getCachedChildPathInternal(Path path, String childName) {
          return Path.getCachedChildPathInternal(path, childName, /*cacheable=*/ true);
        }
      };
    }

    /**
     * Non-null iff this file/directory exists.  Set by setZipEntry for files
     * explicitly mentioned in the zipfile's table of contents, or implicitly
     * an ancestor of them.
     */
    ZipEntry entry = null;

    // Root path.
    private ZipPath(ZipFileSystem fileSystem) {
      super(fileSystem);
    }

    // Non-root paths.
    private ZipPath(ZipFileSystem fileSystem, String name, ZipPath parent) {
      super(fileSystem, name, parent);
    }

    void setZipEntry(ZipEntry entry) {
      if (this.entry != null) {
        throw new IllegalStateException("setZipEntry(" + entry
                                        + ") called twice!");
      }
      this.entry = entry;

      // Ensure all parents of this path have a directory ZipEntry:
      for (ZipPath path = (ZipPath) getParentDirectory();
           path != null && path.entry == null;
           path = (ZipPath) path.getParentDirectory()) {
        // Note, the ZipEntry for the root path is called "//", but that's ok.
        path.setZipEntry(new ZipEntry(path + "/")); // trailing "/" => isDir
      }
    }
  }

  /**
   * Scans the Zip file and associates a ZipEntry with each filename
   * (ZipPath) that is mentioned in the table of contents.  Returns a
   * collection of all corresponding Paths.
   */
  private Collection<Path> populatePathTree() {
    Collection<Path> paths = new ArrayList<>();
    for (ZipEntry entry : Collections.list(zipFile.entries())) {
      PathFragment frag = PathFragment.create(entry.getName());
      Path path = rootPath.getRelative(frag);
      paths.add(path);
      ((ZipPath) path).setZipEntry(entry);
    }
    return paths;
  }

  @Override
  public String getFileSystemType(Path path) {
    return "zipfs";
  }

  @Override
  protected PathFactory getPathFactory() {
    return ZipPath.Factory.INSTANCE;
  }

  /** Returns the ZipEntry associated with a given path name, if any. */
  private static ZipEntry zipEntry(Path path) {
    return ((ZipPath) path).entry;
  }

  /** Like zipEntry, but throws FileNotFoundException unless path exists. */
  private static ZipEntry zipEntryNonNull(Path path)
      throws FileNotFoundException {
    ZipEntry zipEntry = zipEntry(path);
    if (zipEntry == null) {
      throw new FileNotFoundException(path + " (No such file or directory)");
    }
    return zipEntry;
  }

  @Override
  protected InputStream getInputStream(Path path) throws IOException {
    Preconditions.checkState(open);
    return zipFile.getInputStream(zipEntryNonNull(path));
  }

  @Override
  protected Collection<Path> getDirectoryEntries(Path path)
      throws IOException {
    Preconditions.checkState(open);
    zipEntryNonNull(path);
    final Collection<Path> result = new ArrayList<>();
    ((ZipPath) path)
        .applyToChildren(
            child -> {
              if (zipEntry(child) != null) {
                result.add(child);
              }
              return true;
            });
    return result;
  }

  @Override
  protected boolean exists(Path path, boolean followSymlinks) {
    Preconditions.checkState(open);
    return zipEntry(path) != null;
  }

  @Override
  protected boolean isDirectory(Path path, boolean followSymlinks) {
    Preconditions.checkState(open);
    ZipEntry entry = zipEntry(path);
    return entry != null && entry.isDirectory();
  }

  @Override
  protected boolean isFile(Path path, boolean followSymlinks) {
    Preconditions.checkState(open);
    ZipEntry entry = zipEntry(path);
    return entry != null && !entry.isDirectory();
  }

  @Override
  protected boolean isSpecialFile(Path path, boolean followSymlinks) {
    Preconditions.checkState(open);
    return false;
  }

  @Override
  protected boolean isReadable(Path path) throws IOException {
    Preconditions.checkState(open);
    zipEntryNonNull(path);
    return true;
  }

  @Override
  protected boolean isWritable(Path path) throws IOException {
    Preconditions.checkState(open);
    zipEntryNonNull(path);
    return false;
  }

  @Override
  protected boolean isExecutable(Path path) throws IOException {
    Preconditions.checkState(open);
    zipEntryNonNull(path);
    return false;
  }

  @Override
  protected PathFragment readSymbolicLink(Path path) throws IOException {
    Preconditions.checkState(open);
    zipEntryNonNull(path);
    throw new NotASymlinkException(path);
  }

  @Override
  protected long getFileSize(Path path, boolean followSymlinks)
      throws IOException {
    Preconditions.checkState(open);
    return zipEntryNonNull(path).getSize();
  }

  @Override
  protected long getLastModifiedTime(Path path, boolean followSymlinks)
      throws FileNotFoundException {
    Preconditions.checkState(open);
    return zipEntryNonNull(path).getTime();
  }

  @Override
  protected boolean isSymbolicLink(Path path) {
    Preconditions.checkState(open);
    return false;
  }

  @Override
  protected FileStatus statIfFound(Path path, boolean followSymlinks) {
    Preconditions.checkState(open);
    try {
      return stat(path, followSymlinks);
    } catch (FileNotFoundException e) {
      return null;
    } catch (IOException e) {
      // getLastModifiedTime can only throw FileNotFoundException, which is what stat uses.
      throw new IllegalStateException (e);
    }
  }

  @Override
  public void close() {
    if (open) {
      try {
        zipFile.close();
      } catch (IOException e) {
        // Not a lot can be done about this. Log an error and move on.
        logger.warning(
            String.format(
                "Error while closing zip file '%s': %s", zipFile.getName(), e.getMessage()));
      }
      if (tempFile != null) {
        tempFile.delete();
      }
      open = false;
    }
  }
}
