// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.decompressor;

import static com.google.common.base.Strings.isNullOrEmpty;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableMap;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.bazel.repository.RepositoryFunctionException;
import com.google.devtools.build.lib.bazel.repository.decompressor.DecompressorValue.Decompressor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashSet;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;
import org.apache.commons.compress.archivers.sevenz.SevenZArchiveEntry;
import org.apache.commons.compress.archivers.sevenz.SevenZFile;

/**
 * Creates a repository by decompressing a 7-zip file. This implementation generally follows the
 * logic from {@link ZipDecompressor} with the exception that the 7z format does not support file
 * permissions or symbolic links.
 */
public class SevenZDecompressor implements Decompressor {
  public static final Decompressor INSTANCE = new SevenZDecompressor();

  /** Decompresses the file to directory {@link DecompressorDescriptor#destinationPath()} */
  @Override
  @Nullable
  public Path decompress(DecompressorDescriptor descriptor)
      throws IOException, RepositoryFunctionException, InterruptedException {
    Path destinationDirectory = descriptor.destinationPath();
    Optional<String> prefix = descriptor.prefix();
    ImmutableMap<String, String> renameFiles = descriptor.renameFiles();
    boolean foundPrefix = false;

    try (SevenZFile sevenZFile =
        SevenZFile.builder().setFile(descriptor.archivePath().getPathFile()).get()) {
      Iterable<SevenZArchiveEntry> entries = sevenZFile.getEntries();
      for (SevenZArchiveEntry entry : entries) {
        String entryName = entry.getName();
        /*
         * From https://commons.apache.org/proper/commons-compress/examples.html
         *
         * <blockquote>
         *
         * Some 7z archives don't contain any names for the archive entries. The native 7zip tools
         * derive a default name from the name of the archive itself for such entries. Starting with
         * Compress 1.19 SevenZFile has an option to mimic this behavior, but by default unnamed
         * archive entries will return null from {@link SevenZArchiveEntry#getName}.
         *
         * </blockquote>
         *
         * The 7-zip command line will try to rename ALL nameless entries with the same default file
         * name. The user will be prompted if they want to overwrite a previously extracted nameless
         * file with the next nameless file. Since we don't have interactive prompting when doing
         * extractions, and don't know the correct behavior desired (overwrite the file with the
         * later entries or not), we will simply throw an error for ALL nameless entries. Maybe
         * there should be a flag/option to dictate the behavior, but it's probably too small of an
         * edge case.
         */
        if (isNullOrEmpty(entryName)) {
          throw new IOException("7z archive contains unnamed entry");
        }
        entryName = renameFiles.getOrDefault(entryName, entryName);
        StripPrefixedPath entryPath =
            StripPrefixedPath.maybeDeprefix(entryName.getBytes(UTF_8), prefix);
        foundPrefix = foundPrefix || entryPath.foundPrefix();
        if (entryPath.skip()) {
          continue;
        }
        extract7zEntry(sevenZFile, entry, destinationDirectory, entryPath.getPathFragment());
      }

      if (prefix.isPresent() && !foundPrefix) {
        Set<String> prefixes = new HashSet<>();
        for (SevenZArchiveEntry entry : entries) {
          StripPrefixedPath entryPath =
              StripPrefixedPath.maybeDeprefix(entry.getName().getBytes(UTF_8), Optional.empty());
          CouldNotFindPrefixException.maybeMakePrefixSuggestion(entryPath.getPathFragment())
              .ifPresent(prefixes::add);
        }
        throw new CouldNotFindPrefixException(prefix.get(), prefixes);
      }
    }
    return destinationDirectory;
  }

  private static void extract7zEntry(
      SevenZFile sevenZFile,
      SevenZArchiveEntry entry,
      Path destinationDirectory,
      PathFragment strippedRelativePath)
      throws IOException, InterruptedException {
    if (strippedRelativePath.isAbsolute()) {
      throw new IOException(
          String.format(
              "Failed to extract %s, 7-zipped paths cannot be absolute", strippedRelativePath));
    }
    Path outputPath = destinationDirectory.getRelative(strippedRelativePath);
    if (!outputPath.startsWith(destinationDirectory)) {
      throw new IOException(
          String.format(
              "Failed to extract %s, path is escaping the destination directory",
              strippedRelativePath));
    }
    outputPath.getParentDirectory().createDirectoryAndParents();
    boolean isDirectory = entry.isDirectory();
    if (isDirectory) {
      outputPath.createDirectoryAndParents();
    } else {
      try (InputStream input = sevenZFile.getInputStream(entry);
          OutputStream output = outputPath.getOutputStream()) {
        ByteStreams.copy(input, output);
        if (Thread.interrupted()) {
          throw new InterruptedException();
        }
      }
      if (entry.getHasLastModifiedDate()) {
        outputPath.setLastModifiedTime(entry.getLastModifiedTime().toMillis());
      }
    }
  }
}
