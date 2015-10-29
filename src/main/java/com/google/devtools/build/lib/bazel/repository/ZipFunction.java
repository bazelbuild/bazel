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

import com.google.common.base.Optional;
import com.google.devtools.build.lib.bazel.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.zip.ZipFileEntry;
import com.google.devtools.build.zip.ZipReader;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Collection;

import javax.annotation.Nullable;

/**
 * Creates a repository by decompressing a zip file.
 */
public class ZipFunction implements SkyFunction {

  public static final SkyFunctionName NAME = SkyFunctionName.create("ZIP_FUNCTION");

  private static final int WINDOWS_DIRECTORY = 0x10;
  private static final int WINDOWS_FILE = 0x20;

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
  public SkyValue compute(SkyKey skyKey, Environment env) throws RepositoryFunctionException {
    DecompressorDescriptor descriptor = (DecompressorDescriptor) skyKey.argument();
    Path destinationDirectory = descriptor.archivePath().getParentDirectory();
    Optional<String> prefix = descriptor.prefix();
    boolean foundPrefix = false;
    try (ZipReader reader = new ZipReader(descriptor.archivePath().getPathFile())) {
      Collection<ZipFileEntry> entries = reader.entries();
      for (ZipFileEntry entry : entries) {
        StripPrefixedPath entryPath = StripPrefixedPath.maybeDeprefix(entry.getName(), prefix);
        foundPrefix = foundPrefix || entryPath.foundPrefix();
        if (entryPath.skip()) {
          continue;
        }
        extractZipEntry(reader, entry, destinationDirectory, entryPath.getPathFragment());
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(new IOException(
          String.format("Error extracting %s to %s: %s",
              descriptor.archivePath(), destinationDirectory, e.getMessage())),
          Transience.TRANSIENT);
    }

    if (prefix.isPresent() && !foundPrefix) {
      throw new RepositoryFunctionException(
          new IOException("Prefix " + prefix.get() + " was given, but not found in the zip"),
          Transience.PERSISTENT);
    }

    return new DecompressorValue(destinationDirectory);
  }

  private void extractZipEntry(
      ZipReader reader,
      ZipFileEntry entry,
      Path destinationDirectory,
      PathFragment strippedRelativePath)
      throws IOException {
    if (strippedRelativePath.isAbsolute()) {
      throw new IOException(
          String.format(
              "Failed to extract %s, zipped paths cannot be absolute", strippedRelativePath));
    }
    Path outputPath = destinationDirectory.getRelative(strippedRelativePath);
    int permissions = getPermissions(entry.getExternalAttributes(), entry.getName());
    FileSystemUtils.createDirectoryAndParents(outputPath.getParentDirectory());
    boolean isDirectory = (permissions & 040000) == 040000;
    if (isDirectory) {
      FileSystemUtils.createDirectoryAndParents(outputPath);
    } else {
      // TODO(kchodorow): should be able to be removed when issue #236 is resolved, but for now
      // this delete+rewrite is required or the build will error out if outputPath exists here.
      // The zip file is not re-unzipped when the WORKSPACE file is changed (because it is assumed
      // to be immutable) but is on server restart (which is a bug).
      File outputFile = outputPath.getPathFile();
      Files.copy(reader.getInputStream(entry), outputFile.toPath(),
          StandardCopyOption.REPLACE_EXISTING);
      outputPath.chmod(permissions);
    }
  }

  @Override
  @Nullable
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private int getPermissions(int permissions, String path) throws IOException {
    // Posix permissions are in the high-order 2 bytes of the external attributes. After this
    // operation, permissions holds 0100755 (or 040755 for directories).
    int shiftedPermissions = permissions >>> 16;
    if (shiftedPermissions != 0) {
      return shiftedPermissions;
    }

    // If this was zipped up on FAT, it won't have posix permissions set. Instead, this
    // checks if the filename ends with / (for directories) and extra attributes set to 0 for
    // files. From  https://github.com/miloyip/rapidjson/archive/v1.0.2.zip, it looks like
    // executables end up with "normal" (posix) permissions (oddly), so they'll be handled above.
    int windowsPermission = permissions & 0xff;
    if (path.endsWith("/") || (windowsPermission & WINDOWS_DIRECTORY) == WINDOWS_DIRECTORY) {
      // Directory.
      return 040755;
    } else if (permissions == 0 || (windowsPermission & WINDOWS_FILE) == WINDOWS_FILE) {
      // File.
      return 010644;
    }

    // No idea.
    throw new IOException("Unrecognized file mode for " + path + ": 0x"
        + Integer.toHexString(permissions));
  }

}
