// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.windows;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Path.PathFactory;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.windows.jni.WindowsFileOperations;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.attribute.DosFileAttributes;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/** File system implementation for Windows. */
@ThreadSafe
public class WindowsFileSystem extends JavaIoFileSystem {

  // Properties of 8dot3 (DOS-style) short file names:
  // - they are at most 11 characters long
  // - they have a prefix (before "~") that is {1..6} characters long, may contain numbers, letters,
  //   "_", even "~", and maybe even more
  // - they have a "~" after the prefix
  // - have {1..6} numbers after "~" (according to [1] this is only one digit, but MSDN doesn't
  //   clarify this), the combined length up till this point is at most 8
  // - they have an optional "." afterwards, and another {0..3} more characters
  // - just because a path looks like a short name it isn't necessarily one; the user may create
  //   such names and they'd resolve to themselves
  // [1] https://en.wikipedia.org/wiki/8.3_filename#VFAT_and_Computer-generated_8.3_filenames
  //     bullet point (3) (on 2016-12-05)
  @VisibleForTesting
  static final Predicate<String> SHORT_NAME_MATCHER =
      new Predicate<String>() {
        private final Pattern pattern = Pattern.compile("^(.{1,6})~([0-9]{1,6})(\\..{0,3}){0,1}");

        @Override
        public boolean apply(@Nullable String input) {
          Matcher m = pattern.matcher(input);
          return input.length() <= 12
              && m.matches()
              && m.groupCount() >= 2
              && (m.group(1).length() + m.group(2).length()) < 8; // the "~" makes it at most 8
        }
      };

  /** Resolves DOS-style, shortened path names, returning the last segment's long form. */
  private static final Function<String, String> WINDOWS_SHORT_PATH_RESOLVER =
      path -> {
        try {
          // Since Path objects are created hierarchically, we know for sure that every segment of
          // the path, except the last one, is already canonicalized, so we can return just that.
          // Plus the returned value is passed to Path.getChild so we must not return a full
          // path here.
          return PathFragment.create(WindowsFileOperations.getLongPath(path)).getBaseName();
        } catch (IOException e) {
          return null;
        }
      };

  @VisibleForTesting
  private enum WindowsPathFactory implements PathFactory {
    INSTANCE {
      @Override
      public Path createRootPath(FileSystem filesystem) {
        return new WindowsPath(filesystem, PathFragment.ROOT_DIR, null);
      }

      @Override
      public Path createChildPath(Path parent, String childName) {
        Preconditions.checkState(parent instanceof WindowsPath);
        return new WindowsPath(parent.getFileSystem(), childName, (WindowsPath) parent);
      }

      @Override
      public Path getCachedChildPathInternal(Path path, String childName) {
        return WindowsPathFactory.getCachedChildPathInternalImpl(
            path, childName, WINDOWS_SHORT_PATH_RESOLVER);
      }
    };

    private static Path getCachedChildPathInternalImpl(
        Path parent, String child, Function<String, String> resolver) {
      if (parent != null && parent.isRootDirectory()) {
        // This is a top-level directory. It must be a drive name ("C:" or "c").
        if (WindowsPath.isWindowsVolumeName(child)) {
          child = WindowsPath.getDriveLetter((WindowsPath) parent, child) + ":";
        } else {
          throw new IllegalArgumentException("Cannot create Unix-style paths on Windows.");
        }
      }

      String resolvedChild = child;
      if (parent != null && !parent.isRootDirectory() && SHORT_NAME_MATCHER.apply(child)) {
        String pathString = parent.getPathString();
        if (!pathString.endsWith("/")) {
          pathString += "/";
        }
        pathString += child;
        resolvedChild = resolver.apply(pathString);
      }
      return Path.getCachedChildPathInternal(
          parent,
          // If resolution succeeded, or we didn't attempt to resolve, then `resolvedChild` has the
          // child name. If it's null, then resolution failed; use the unresolved child name in that
          // case.
          resolvedChild != null ? resolvedChild : child,
          // If resolution failed, likely because the path doesn't exist, then do not cache the
          // child. If we did, then in case the path later came into existence, we'd have a stale
          // cache entry.
          /* cacheable */ resolvedChild != null);
    }

    /**
     * Creates a {@link PathFactory} with a mock shortname resolver.
     *
     * <p>The factory works exactly like the actual one ({@link WindowsPathFactory#INSTANCE}) except
     * it's using the mock resolver.
     */
    public static PathFactory createForTesting(final Function<String, String> mockResolver) {
      return new PathFactory() {
        @Override
        public Path createRootPath(FileSystem filesystem) {
          return INSTANCE.createRootPath(filesystem);
        }

        @Override
        public Path createChildPath(Path parent, String childName) {
          return INSTANCE.createChildPath(parent, childName);
        }

        @Override
        public Path getCachedChildPathInternal(Path path, String childName) {
          return WindowsPathFactory.getCachedChildPathInternalImpl(path, childName, mockResolver);
        }
      };
    }
  }

  /** A windows-specific subclass of Path. */
  @VisibleForTesting
  protected static final class WindowsPath extends Path {

    // The drive letter is '\0' if and only if this Path is the filesystem root "/".
    private char driveLetter;

    private WindowsPath(FileSystem fileSystem) {
      super(fileSystem);
      this.driveLetter = '\0';
    }

    private WindowsPath(FileSystem fileSystem, String name, WindowsPath parent) {
      super(fileSystem, name, parent);
      this.driveLetter = getDriveLetter(parent, name);
    }

    @Override
    protected void buildPathString(StringBuilder result) {
      if (isRootDirectory()) {
        result.append(PathFragment.ROOT_DIR);
      } else {
        if (isTopLevelDirectory()) {
          result.append(driveLetter).append(':').append(PathFragment.SEPARATOR_CHAR);
        } else {
          WindowsPath parent = (WindowsPath) getParentDirectory();
          parent.buildPathString(result);
          if (!parent.isTopLevelDirectory()) {
            result.append(PathFragment.SEPARATOR_CHAR);
          }
          result.append(getBaseName());
        }
      }
    }

    @Override
    public void reinitializeAfterDeserialization() {
      Preconditions.checkState(
          getParentDirectory().isRootDirectory() || getParentDirectory() instanceof WindowsPath);
      this.driveLetter =
          (getParentDirectory() != null) ? ((WindowsPath) getParentDirectory()).driveLetter : '\0';
    }

    @Override
    public boolean isMaybeRelativeTo(Path ancestorCandidate) {
      Preconditions.checkState(ancestorCandidate instanceof WindowsPath);
      return ancestorCandidate.isRootDirectory()
          || driveLetter == ((WindowsPath) ancestorCandidate).driveLetter;
    }

    @Override
    public boolean isTopLevelDirectory() {
      return isRootDirectory() || getParentDirectory().isRootDirectory();
    }

    @Override
    public PathFragment asFragment() {
      String[] segments = getSegments();
      if (segments.length > 0) {
        // Strip off the first segment that contains the volume name.
        segments = Arrays.copyOfRange(segments, 1, segments.length);
      }

      return PathFragment.create(driveLetter, true, segments);
    }

    @Override
    protected Path getRootForRelativePathComputation(PathFragment relative) {
      Path result = this;
      if (relative.isAbsolute()) {
        result = getFileSystem().getRootDirectory();
        if (!relative.windowsVolume().isEmpty()) {
          result = result.getRelative(relative.windowsVolume());
        }
      }
      return result;
    }

    private static boolean isWindowsVolumeName(String name) {
      return (name.length() == 1 || (name.length() == 2 && name.charAt(1) == ':'))
          && Character.isLetter(name.charAt(0));
    }

    private static char getDriveLetter(WindowsPath parent, String name) {
      if (parent == null) {
        return '\0';
      } else {
        if (parent.isRootDirectory()) {
          Preconditions.checkState(
              isWindowsVolumeName(name),
              "top-level directory on Windows must be a drive (name = '%s')",
              name);
          return Character.toUpperCase(name.charAt(0));
        } else {
          return parent.driveLetter;
        }
      }
    }

    @VisibleForTesting
    @Override
    protected synchronized void applyToChildren(Predicate<Path> function) {
      super.applyToChildren(function);
    }
  }

  @VisibleForTesting
  static PathFactory getPathFactoryForTesting(Function<String, String> mockResolver) {
    return WindowsPathFactory.createForTesting(mockResolver);
  }

  public static final LinkOption[] NO_OPTIONS = new LinkOption[0];
  public static final LinkOption[] NO_FOLLOW = new LinkOption[] {LinkOption.NOFOLLOW_LINKS};

  public WindowsFileSystem() {}

  public WindowsFileSystem(HashFunction hashFunction) {
    super(hashFunction);
  }

  @Override
  protected PathFactory getPathFactory() {
    return WindowsPathFactory.INSTANCE;
  }

  @Override
  public String getFileSystemType(Path path) {
    // TODO(laszlocsomor): implement this properly, i.e. actually query this information from
    // somewhere (java.nio.Filesystem? System.getProperty? implement JNI method and use WinAPI?).
    return "ntfs";
  }

  @Override
  protected void createSymbolicLink(Path linkPath, PathFragment targetFragment) throws IOException {
    Path targetPath =
        targetFragment.isAbsolute()
            ? getPath(targetFragment)
            : linkPath.getParentDirectory().getRelative(targetFragment);
    try {
      java.nio.file.Path link = getIoFile(linkPath).toPath();
      java.nio.file.Path target = getIoFile(targetPath).toPath();
      // Still Create a dangling junction if the target doesn't exist.
      if (!target.toFile().exists() || target.toFile().isDirectory()) {
        WindowsFileOperations.createJunction(link.toString(), target.toString());
      } else {
        Files.copy(target, link);
      }
    } catch (java.nio.file.FileAlreadyExistsException e) {
      throw new IOException(linkPath + ERR_FILE_EXISTS);
    } catch (java.nio.file.AccessDeniedException e) {
      throw new IOException(linkPath + ERR_PERMISSION_DENIED);
    } catch (java.nio.file.NoSuchFileException e) {
      throw new FileNotFoundException(linkPath + ERR_NO_SUCH_FILE_OR_DIR);
    }
  }

  @Override
  public boolean supportsSymbolicLinksNatively(Path path) {
    return false;
  }

  @Override
  public boolean isFilePathCaseSensitive() {
    return false;
  }

  @Override
  protected boolean fileIsSymbolicLink(File file) {
    try {
      if (isJunction(file)) {
        return true;
      }
    } catch (IOException e) {
      // Did not work, try in another way
    }
    return super.fileIsSymbolicLink(file);
  }

  public static LinkOption[] symlinkOpts(boolean followSymlinks) {
    return followSymlinks ? NO_OPTIONS : NO_FOLLOW;
  }

  @Override
  protected FileStatus stat(Path path, boolean followSymlinks) throws IOException {
    File file = getIoFile(path);
    final DosFileAttributes attributes;
    try {
      attributes = getAttribs(file, followSymlinks);
    } catch (IOException e) {
      throw new FileNotFoundException(path + ERR_NO_SUCH_FILE_OR_DIR);
    }

    final boolean isSymbolicLink = !followSymlinks && fileIsSymbolicLink(file);
    FileStatus status =
        new FileStatus() {
          @Override
          public boolean isFile() {
            return attributes.isRegularFile() || (isSpecialFile() && !isDirectory());
          }

          @Override
          public boolean isSpecialFile() {
            return attributes.isOther();
          }

          @Override
          public boolean isDirectory() {
            return attributes.isDirectory();
          }

          @Override
          public boolean isSymbolicLink() {
            return isSymbolicLink;
          }

          @Override
          public long getSize() throws IOException {
            return attributes.size();
          }

          @Override
          public long getLastModifiedTime() throws IOException {
            return attributes.lastModifiedTime().toMillis();
          }

          @Override
          public long getLastChangeTime() {
            // This is the best we can do with Java NIO...
            return attributes.lastModifiedTime().toMillis();
          }

          @Override
          public long getNodeId() {
            // TODO(bazel-team): Consider making use of attributes.fileKey().
            return -1;
          }
        };

    return status;
  }

  @Override
  protected boolean isDirectory(Path path, boolean followSymlinks) {
    if (!followSymlinks) {
      try {
        if (isJunction(getIoFile(path))) {
          return false;
        }
      } catch (IOException e) {
        return false;
      }
    }
    return super.isDirectory(path, followSymlinks);
  }

  /**
   * Returns true if the path refers to a directory junction, directory symlink, or regular symlink.
   *
   * <p>Directory junctions are symbolic links created with "mklink /J" where the target is a
   * directory or another directory junction. Directory junctions can be created without any user
   * privileges.
   *
   * <p>Directory symlinks are symbolic links created with "mklink /D" where the target is a
   * directory or another directory symlink. Note that directory symlinks can only be created by
   * Administrators.
   *
   * <p>Normal symlinks are symbolic links created with "mklink". Normal symlinks should not point
   * at directories, because even though "mklink" can create the link, it will not be a functional
   * one (the linked directory's contents cannot be listed). Only Administrators may create regular
   * symlinks.
   *
   * <p>This method returns true for all three types as long as their target is a directory (even if
   * they are dangling), though only directory junctions and directory symlinks are useful.
   */
  @VisibleForTesting
  static boolean isJunction(File file) throws IOException {
    return WindowsFileOperations.isJunction(file.getPath());
  }

  private static DosFileAttributes getAttribs(File file, boolean followSymlinks)
      throws IOException {
    return Files.readAttributes(
        file.toPath(), DosFileAttributes.class, symlinkOpts(followSymlinks));
  }
}
