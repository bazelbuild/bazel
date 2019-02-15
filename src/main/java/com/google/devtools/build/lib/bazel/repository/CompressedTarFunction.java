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
import com.google.devtools.build.lib.bazel.repository.DecompressorValue.Decompressor;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Date;
import java.util.HashSet;
import java.util.Set;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;

/**
 * Common code for unarchiving a compressed TAR file.
 */
public abstract class CompressedTarFunction implements Decompressor {
  protected abstract InputStream getDecompressorStream(DecompressorDescriptor descriptor)
      throws IOException;

  @Override
  public Path decompress(DecompressorDescriptor descriptor) throws IOException {
    Optional<String> prefix = descriptor.prefix();
    boolean foundPrefix = false;
    Set<String> availablePrefixes = new HashSet<>();

    try (InputStream decompressorStream = getDecompressorStream(descriptor)) {
      TarArchiveInputStream tarStream = new TarArchiveInputStream(decompressorStream);
      TarArchiveEntry entry;
      while ((entry = tarStream.getNextTarEntry()) != null) {
        StripPrefixedPath entryPath = StripPrefixedPath.maybeDeprefix(entry.getName(), prefix);
        foundPrefix = foundPrefix || entryPath.foundPrefix();

        if (prefix.isPresent() && !foundPrefix) {
          Optional<String> suggestion =
              CouldNotFindPrefixException.maybeMakePrefixSuggestion(entryPath.getPathFragment());
          if (suggestion.isPresent()) {
            availablePrefixes.add(suggestion.get());
          }
        }

        if (entryPath.skip()) {
          continue;
        }

        Path filename = descriptor.repositoryPath().getRelative(entryPath.getPathFragment());
        FileSystemUtils.createDirectoryAndParents(filename.getParentDirectory());
        if (entry.isDirectory()) {
          FileSystemUtils.createDirectoryAndParents(filename);
        } else {
          if (entry.isSymbolicLink() || entry.isLink()) {
            PathFragment linkName = PathFragment.create(entry.getLinkName());
            boolean wasAbsolute = linkName.isAbsolute();
            // Strip the prefix from the link path if set.
            linkName =
                StripPrefixedPath.maybeDeprefix(linkName.getPathString(), prefix).getPathFragment();
            if (wasAbsolute) {
              // Recover the path to an absolute path as maybeDeprefix() relativize the path
              // even if the prefix is not set
              linkName = descriptor.repositoryPath().getRelative(linkName).asFragment();
            }
            if (filename.exists()) {
              filename.delete();
            }
            if (entry.isSymbolicLink()) {
              FileSystemUtils.ensureSymbolicLink(filename, linkName);
            } else {
              FileSystemUtils.createHardLink(
                  filename, descriptor.repositoryPath().getRelative(linkName));
            }
          } else {
            Files.copy(
                tarStream, filename.getPathFile().toPath(), StandardCopyOption.REPLACE_EXISTING);
            filename.chmod(entry.getMode());

            // This can only be done on real files, not links, or it will skip the reader to
            // the next "real" file to try to find the mod time info.
            Date lastModified = entry.getLastModifiedDate();
            filename.setLastModifiedTime(lastModified.getTime());
          }
        }
      }

      if (prefix.isPresent() && !foundPrefix) {
        throw new CouldNotFindPrefixException(prefix.get(), availablePrefixes);
      }
    }

    return descriptor.repositoryPath();
  }
}
