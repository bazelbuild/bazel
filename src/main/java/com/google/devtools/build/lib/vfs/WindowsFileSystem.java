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
package com.google.devtools.build.lib.vfs;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path.PathFactory;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.DosFileAttributes;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

/** Jury-rigged file system for Windows. */
@ThreadSafe
public class WindowsFileSystem extends JavaIoFileSystem {

  @VisibleForTesting
  enum WindowsPathFactory implements PathFactory {
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
    };
  }

  private static final class WindowsPath extends Path {

    private static final String WINDOWS_UNIX_ROOT_JVM_ARG = "bazel.windows_unix_root";
    private static final String BAZEL_SH_ENV_VAR = "BAZEL_SH";

    // Absolute Windows path specifying the root of absolute Unix paths.
    // This is typically the MSYS installation root, e.g. C:\\tools\\msys64
    private static final PathFragment UNIX_ROOT =
        determineUnixRoot(WINDOWS_UNIX_ROOT_JVM_ARG, BAZEL_SH_ENV_VAR);

    // The drive letter is '\0' if and only if this Path is the filesystem root "/".
    private char driveLetter;

    private WindowsPath(FileSystem fileSystem) {
      super(fileSystem);
      this.driveLetter = '\0';
    }

    private WindowsPath(FileSystem fileSystem, String name, WindowsPath parent) {
      super(fileSystem, translateName(parent, name), translateParent(parent, name));
      this.driveLetter = getDriveLetter((WindowsPath) getParentDirectory(), getBaseName());
    }

    @Override
    protected void buildPathString(StringBuilder result) {
      if (isRootDirectory()) {
        result.append(PathFragment.ROOT_DIR);
      } else {
        if (isTopLevelDirectory()) {
          result.append(driveLetter).append(':').append(PathFragment.SEPARATOR_CHAR);
        } else {
          getParentDirectory().buildPathString(result);
          if (!getParentDirectory().isTopLevelDirectory()) {
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

      return new PathFragment(driveLetter, true, segments);
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

    private static String translateName(WindowsPath parent, String name) {
      if (parent != null && parent.isRootDirectory() && name.length() == 1) {
        // Path is /c (or similar), which is equivalent to C:/
        return name.toUpperCase() + ":";
      } else {
        return name;
      }
    }

    /**
     * Heuristic to translate absolute Unix paths to Windows paths.
     *
     * <p>Unix paths under MSYS should be resolved using /etc/mtab, but reading that every time we
     * create a Path object would be too expensive.
     *
     * <p>As a heuristic, we check if the paths looks like an absolute Unix path (e.g. "/usr/lib")
     * and make it relative to the MSYS root (yielding "c:/tools/msys64/usr/lib"), but only if the
     * path doesn't also look like an absolute path on a drive (e.g. /c/windows which means
     * C:/windows).
     *
     * <p>This is an imperfect workaround because the user may have other paths mounted as well, and
     * this heuristic won't handle those properly, but it's good enough.
     *
     * <p>The correct long-term solution is to update all tools on Windows to output Windows paths
     * and lock down WindowsPath creation to disallow absolute Unix paths.
     */
    private static WindowsPath translateParent(WindowsPath parent, String name) {
      if (parent != null && parent.isRootDirectory() && !isWindowsVolumeName(name)) {
        // This is a top-level directory which is not a drive name, e.g. "/usr".
        // Make it relative to UNIX_ROOT.
        Preconditions.checkNotNull(
            UNIX_ROOT,
            "Could not determine Unix path root or it is not an absolute Windows path. Set the "
                + "\"%s\" JVM argument, or export the \"%s\" environment variable for the MSYS bash"
                + " and have /usr/bin/cygpath installed",
            WINDOWS_UNIX_ROOT_JVM_ARG,
            BAZEL_SH_ENV_VAR);

        return (WindowsPath) parent.getRelative(UNIX_ROOT);
      } else {
        // This is not a top-level directory.
        return parent;
      }
    }

    private char getDriveLetter(WindowsPath parent, String name) {
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
  }

  public static final LinkOption[] NO_OPTIONS = new LinkOption[0];
  public static final LinkOption[] NO_FOLLOW = new LinkOption[] {LinkOption.NOFOLLOW_LINKS};

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
    // TODO(lberki): Add some JNI to create hard links/junctions instead of calling out to
    // cmd.exe
    File file = getIoFile(linkPath);
    try {
      File targetFile = new File(targetFragment.getPathString());
      if (targetFile.isDirectory()) {
        createDirectoryJunction(targetFile, file);
      } else {
        Files.copy(targetFile.toPath(), file.toPath());
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
  public boolean supportsSymbolicLinksNatively() {
    return false;
  }

  @Override
  public boolean isFilePathCaseSensitive() {
    return false;
  }

  private void createDirectoryJunction(File sourceDirectory, File targetPath) throws IOException {
    StringBuilder cl = new StringBuilder("cmd.exe /c ");
    cl.append("mklink /J ");
    cl.append('"');
    cl.append(targetPath.getAbsolutePath());
    cl.append('"');
    cl.append(' ');
    cl.append('"');
    cl.append(sourceDirectory.getAbsolutePath());
    cl.append('"');
    Process process = Runtime.getRuntime().exec(cl.toString());
    try {
      process.waitFor();
      if (process.exitValue() != 0) {
        throw new IOException("Command failed " + cl);
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException("Command failed ", e);
    }
  }

  @Override
  protected boolean fileIsSymbolicLink(File file) {
    try {
      if (file.isDirectory() && isJunction(file.toPath())) {
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
    final BasicFileAttributes attributes;
    try {
      attributes =
          Files.readAttributes(
              file.toPath(), BasicFileAttributes.class, symlinkOpts(followSymlinks));
    } catch (java.nio.file.FileSystemException e) {
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
        if (isJunction(getIoFile(path).toPath())) {
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
  // TODO(laszlocsomor): fix https://github.com/bazelbuild/bazel/issues/1735 and use the JNI method
  // in WindowsFileOperations.
  @VisibleForTesting
  static boolean isJunction(java.nio.file.Path p) throws IOException {
    if (Files.exists(p, symlinkOpts(/* followSymlinks */ false))) {
      DosFileAttributes attributes =
          Files.readAttributes(p, DosFileAttributes.class, symlinkOpts(/* followSymlinks */ false));

      if (attributes.isRegularFile()) {
        return false;
      }

      if (attributes.isDirectory()) {
        return attributes.isOther();
      } else {
        return attributes.isSymbolicLink();
      }
    }
    return false;
  }

  private static PathFragment determineUnixRoot(String jvmArgName, String bazelShEnvVar) {
    // Get the path from a JVM argument, if specified.
    String path = System.getProperty(jvmArgName);

    if (path == null || path.isEmpty()) {
      path = "";

      // Fall back to executing cygpath.
      String bash = System.getenv(bazelShEnvVar);
      Process process = null;
      try {
        process = Runtime.getRuntime().exec(bash + "-c \"/usr/bin/cygpath -m /\"");

        // Wait 3 seconds max, that should be enough to run this command.
        process.waitFor(3, TimeUnit.SECONDS);

        if (process.exitValue() == 0) {
          char[] buf = new char[256];
          try (InputStreamReader r = new InputStreamReader(process.getInputStream())) {
            int len = 0;
            while ((len = r.read(buf)) > 0) {
              path = path + new String(buf, 0, len);
            }
          }
        }
      } catch (InterruptedException | IOException e) {
        // Silently ignore failure. Either MSYS is installed at a different location, or not
        // installed at all, or some error occurred. We can't do anything anymore but throw an
        // exception if someone tries to create a Path from an absolute Unix path.
      }
    }

    path = path.trim();
    PathFragment result = new PathFragment(path);
    if (path.isEmpty() || result.getDriveLetter() == '\0' || !result.isAbsolute()) {
      return null;
    } else {
      return result;
    }
  }
}
