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
package com.google.devtools.build.android;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.collect.ImmutableSet;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.FileVisitOption;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.logging.Logger;

/**
 * Unpacks specially named tar files in a resource file tree.
 * 
 * <p>Scans a list of Resource directories looking for "raw/blaze_internal_packed_resources.tar".
 * When found, it is unpacked into a new resource directory.</p>
 */
// TODO(bazel-team): Remove when Android support library version is handled by configurable
// attribute.
class PackedResourceTarExpander implements DirectoryModifier {
  private static final Logger LOGGER = Logger.getLogger(PackedResourceTarExpander.class.getName());

  private static final class ConditionallyLinkingVisitor extends SimpleFileVisitor<Path> {

    private final Path fileToexclude;
    private Path out;
    private Path workingDirectory;

    private ConditionallyLinkingVisitor(Path fileToExclude, Path out, Path workingDirectory) {
      this.fileToexclude = fileToExclude;
      this.out = out;
      this.workingDirectory = workingDirectory;
    }

    @Override
    public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs)
        throws IOException {
      Files.createDirectories(out.resolve(workingDirectory.relativize(dir)));
      return super.preVisitDirectory(dir, attrs);
    }

    @Override
    public FileVisitResult visitFile(Path file, BasicFileAttributes attrs)
        throws IOException {
      if (!fileToexclude.equals(file)) {
        // TODO(bazel-team): Change to a symlink when the merge code supports symlinks.
        Files.copy(file, out.resolve(workingDirectory.relativize(file)));
        //Files.createSymbolicLink(out.resolve(workingDirectory.relativize(file)), file);
      }
      return super.visitFile(file, attrs);
    }
  }

  private final Path out;
  private Path workingDirectory;

  public PackedResourceTarExpander(Path out, Path workingDirectory) {
    this.out = out;
    this.workingDirectory = workingDirectory;
  }

  @Override
  public ImmutableList<Path> modify(ImmutableList<Path> resourceRoots) {
    final Builder<Path> outDirs = ImmutableList.builder();
    for (final Path unresolvedRoot : resourceRoots) {
      Path root = unresolvedRoot.toAbsolutePath();
      try {
        final Path packedResources =
            root.resolve("raw/blaze_internal_packed_resources.tar");
        if (Files.exists(packedResources)) {
          Preconditions.checkArgument(root.startsWith(workingDirectory),
              "%s is not under %s", root, workingDirectory);
          final Path resourcePrefix = workingDirectory.relativize(root);
          final Path targetDirectory = out.resolve(resourcePrefix);
          outDirs.add(targetDirectory);
          copyRemainingResources(root, packedResources);
          // Group the unpacked resource by the path they came from.
          final Path tarOut =
              out.resolve("blaze_internal_packed_resources").resolve(resourcePrefix);
          unTarPackedResources(tarOut, packedResources);
          outDirs.add(tarOut);
        } else {
          outDirs.add(root);
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
    return outDirs.build();
  }

  private void unTarPackedResources(final Path tarOut, final Path packedResources)
      throws IOException {
    LOGGER.fine(String.format("Found packed resources: %s", packedResources));
    try (InputStream inputStream = Files.newInputStream(packedResources);
        TarArchiveInputStream tarStream = new TarArchiveInputStream(inputStream)) {
      byte[] temp = new byte[4 * 1024];
      for (TarArchiveEntry entry = tarStream.getNextTarEntry(); entry != null;
          entry = tarStream.getNextTarEntry()) {
        if (!entry.isFile()) {
          continue;
        }
        int read = tarStream.read(temp);
        // packed tars can start with a ./. This can cause issues, so remove it.
        final Path entryPath = tarOut.resolve(entry.getName().replace("^\\./", ""));
        Files.createDirectories(entryPath.getParent());
        final OutputStream entryOutStream = Files.newOutputStream(entryPath);
        while (read > -1) {
          entryOutStream.write(temp, 0, read);
          read = tarStream.read(temp);
        }
        entryOutStream.flush();
        entryOutStream.close();
      }
    }
  }

  private void copyRemainingResources(final Path resourcePath, final Path packedResources)
      throws IOException {
    Files.walkFileTree(resourcePath, ImmutableSet.of(FileVisitOption.FOLLOW_LINKS),
        Integer.MAX_VALUE, new ConditionallyLinkingVisitor(packedResources, out, workingDirectory));
  }
}
