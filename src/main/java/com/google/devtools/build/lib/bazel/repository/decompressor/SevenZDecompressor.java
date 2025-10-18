package com.google.devtools.build.lib.bazel.repository.decompressor;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.bazel.repository.RepositoryFunctionException;
import com.google.devtools.build.lib.bazel.repository.decompressor.DecompressorValue.Decompressor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.apache.commons.compress.archivers.sevenz.SevenZArchiveEntry;
import org.apache.commons.compress.archivers.sevenz.SevenZFile;

import javax.annotation.Nullable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

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
    Map<String, String> renameFiles = descriptor.renameFiles();
    boolean foundPrefix = false;

    try (SevenZFile sevenZFile =
        SevenZFile.builder().setFile(descriptor.archivePath().getPathFile()).get()) {
      Iterable<SevenZArchiveEntry> entries = sevenZFile.getEntries();
      for (SevenZArchiveEntry entry : entries) {
        String entryName = entry.getName();
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
    // Sanity/security check - at this point, uplevel references (..) should be resolved.
    // There shouldn't be any remaining uplevel references, otherwise, the extracted file could
    // "escape" the destination directory.
    if (strippedRelativePath.containsUplevelReferences()) {
      throw new IOException(
          String.format(
              "Failed to extract %s, 7-zipped entry contains uplevel references (..)",
              strippedRelativePath));
    }
    Path outputPath = destinationDirectory.getRelative(strippedRelativePath);
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
      outputPath.setLastModifiedTime(entry.getLastModifiedTime().toMillis());
    }
  }
}
