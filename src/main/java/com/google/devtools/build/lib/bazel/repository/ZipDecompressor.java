// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.bazel.repository.DecompressorValue.Decompressor;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.zip.ZipFileEntry;
import com.google.devtools.build.zip.ZipReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.*;
import java.util.zip.ZipFile;

import javax.annotation.Nullable;

/**
 * Creates a repository by decompressing a zip file.
 */
public class ZipDecompressor implements Decompressor {

  public static final Decompressor INSTANCE = new ZipDecompressor();
  private static final long MAX_PATH_LENGTH = 256;

  private ZipDecompressor() {
  }

  private static final int S_IFDIR = 040000;
  private static final int S_IFREG = 0100000;
  private static final int S_IFLNK = 0120000;
  private static final int EXECUTABLE_MASK = 0755;
  @VisibleForTesting
  static final int WINDOWS_DIRECTORY = 0x10;
  @VisibleForTesting
  static final int WINDOWS_FILE = 0x20;

  /**
   * This unzips the zip file to a sibling directory of {@link DecompressorDescriptor#archivePath}.
   * The zip file is expected to have the WORKSPACE file at the top level, e.g.:
   *
   * <pre>
   * $ unzip -lf some-repo.zip
   * Archive:  ../repo.zip
   *  Length      Date    Time    Name
   * ---------  ---------- -----   ----
   *        0  2014-11-20 15:50   WORKSPACE
   *        0  2014-11-20 16:10   foo/
   *      236  2014-11-20 15:52   foo/BUILD
   *      ...
   * </pre>
   */
  @Override
  @Nullable
  public Path decompress(DecompressorDescriptor descriptor)
      throws IOException {
    Path destinationDirectory = descriptor.archivePath().getParentDirectory();
    Optional<String> prefix = descriptor.prefix();
    boolean foundPrefix = false;
    ZipReader reader = new ZipReader(descriptor.archivePath().getPathFile());
    Collection<ZipFileEntry> entries = reader.entries();
    // Store link, target info of symlinks, we create them after regular files are extracted.
    Map<Path, PathFragment> symlinks = new HashMap<>();
    for (ZipFileEntry entry : entries) {
      StripPrefixedPath entryPath = StripPrefixedPath.maybeDeprefix(entry.getName(), prefix);
      foundPrefix = foundPrefix || entryPath.foundPrefix();
      if (entryPath.skip()) {
        continue;
      }
      extractZipEntry(reader, entry, destinationDirectory, entryPath.getPathFragment(), symlinks);
    }
    for (Map.Entry<Path, PathFragment> symlink : symlinks.entrySet()) {
      symlink.getKey().createSymbolicLink(symlink.getValue());
    }

    if (prefix.isPresent() && !foundPrefix) {
      Set<String> prefixes = new HashSet<>();
      for (ZipFileEntry entry : entries) {
        StripPrefixedPath entryPath = StripPrefixedPath
            .maybeDeprefix(entry.getName(), Optional.absent());
        Optional<String> suggestion = CouldNotFindPrefixException.getPrefixSuggestion(entryPath.getPathFragment());
        // Is that a directory?
        if (suggestion.isPresent()) {
          prefixes.add(suggestion.get());
        }
      }
      throw new CouldNotFindPrefixException(prefix.get(), prefixes);
    }

    return destinationDirectory;
  }

  private static void extractZipEntry(
      ZipReader reader,
      ZipFileEntry entry,
      Path destinationDirectory,
      PathFragment strippedRelativePath,
      Map<Path, PathFragment> symlinks)
      throws IOException {
    if (strippedRelativePath.isAbsolute()) {
      throw new IOException(
          String.format(
              "Failed to extract %s, zipped paths cannot be absolute", strippedRelativePath));
    }
    Path outputPath = destinationDirectory.getRelative(strippedRelativePath);
    int permissions = getPermissions(entry.getExternalAttributes(), entry.getName());
    FileSystemUtils.createDirectoryAndParents(outputPath.getParentDirectory());
    boolean isDirectory = (permissions & S_IFDIR) == S_IFDIR;
    boolean isSymlink = (permissions & S_IFLNK) == S_IFLNK;
    if (isDirectory) {
      FileSystemUtils.createDirectoryAndParents(outputPath);
    } else if (isSymlink) {
      Preconditions.checkState(entry.getSize() < MAX_PATH_LENGTH);
      byte[] buffer = new byte[(int) entry.getSize()];
      // For symlinks, the "compressed data" is actually the target name.
      int read = reader.getInputStream(entry).read(buffer);
      Preconditions.checkState(read == buffer.length);
      PathFragment target = PathFragment.create(new String(buffer, Charset.defaultCharset()));
      if (target.containsUplevelReferences()) {
        PathFragment pointsTo = strippedRelativePath.getParentDirectory().getRelative(target);
        if (pointsTo.containsUplevelReferences()) {
          throw new IOException("Zip entries cannot refer to files outside of their directory: "
              + reader.getFilename() + " has a symlink " + strippedRelativePath + " pointing to "
              + target);
        }
      }
      if (target.isAbsolute()) {
        target = target.relativeTo("/");
        target = destinationDirectory.getRelative(target).asFragment();
      }
      symlinks.put(outputPath, target);
    } else {
      // TODO(kchodorow): should be able to be removed when issue #236 is resolved, but for now
      // this delete+rewrite is required or the build will error out if outputPath exists here.
      // The zip file is not re-unzipped when the WORKSPACE file is changed (because it is assumed
      // to be immutable) but is on server restart (which is a bug).
      File outputFile = outputPath.getPathFile();
      try (InputStream input = reader.getInputStream(entry)) {
        Files.copy(input, outputFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
      }
      outputPath.chmod(permissions);
      outputPath.setLastModifiedTime(entry.getTime());
    }
  }

  @VisibleForTesting
  static int getPermissions(int permissions, String path) throws IOException {
    // Sometimes zip files list directories as being "regular" executable files (i.e., 0100755).
    // I'm looking at you, Go AppEngine SDK 1.9.37 (see #1263 for details).
    if (path.endsWith("/")) {
      return S_IFDIR | EXECUTABLE_MASK;
    }

    // Posix permissions are in the high-order 2 bytes of the external attributes. After this
    // operation, permissions holds 0100755 (or 040755 for directories).
    int shiftedPermissions = permissions >>> 16;
    if (shiftedPermissions != 0) {
      return shiftedPermissions;
    }

    // If this was zipped up on FAT, it won't have posix permissions set. Instead, this
    // checks if extra attributes is set to 0 for files. From
    // https://github.com/miloyip/rapidjson/archive/v1.0.2.zip, it looks like executables end up
    // with "normal" (posix) permissions (oddly), so they'll be handled above.
    int windowsPermission = permissions & 0xff;
    if ((windowsPermission & WINDOWS_DIRECTORY) == WINDOWS_DIRECTORY) {
      // Directory.
      return S_IFDIR | EXECUTABLE_MASK;
    } else if (permissions == 0 || (windowsPermission & WINDOWS_FILE) == WINDOWS_FILE) {
      // File.
      return S_IFREG | EXECUTABLE_MASK;
    }

    // No idea.
    throw new IOException("Unrecognized file mode for " + path + ": 0x"
        + Integer.toHexString(permissions));
  }

}
