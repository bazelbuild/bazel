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
import com.google.common.base.Throwables;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.common.hash.HashCode;
import com.google.common.hash.HashFunction;
import com.google.common.hash.Hasher;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.FileVisitOption;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.logging.Logger;

/**
 * Deduplicates identical files in the provided directories. 
 * <p>
 * This is necessary for the andorid_resources deprecation -- the old style of inheritance
 * required all relevant resources to be copied from each dependency. This means each resource is
 * duplicated for each resource set. This modifier creates a sym link forest for each unique file
 * on a first come, first serve basis. Which makes aapt and the merging code loads happier.
 */
public class FileDeDuplicator implements DirectoryModifier {
  private static final Logger LOGGER = Logger.getLogger(FileDeDuplicator.class.getName());

  private static final class ConditionalCopyVisitor extends SimpleFileVisitor<Path> {
    private final Path newRoot;
    private final Path workingDir;
    private Multimap<Path, HashCode> seen;
    private HashFunction hashFunction;

    private ConditionalCopyVisitor(Path newRoot, Path workingDir,
        Multimap<Path, HashCode> seen, HashFunction hashFunction) {
      this.newRoot = newRoot;
      this.workingDir = workingDir;
      this.seen = seen;
      this.hashFunction = hashFunction;
    }

    @Override
    public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs)
        throws IOException {
      Files.createDirectories(newRoot.resolve(workingDir.relativize(dir)));
      return super.preVisitDirectory(dir, attrs);
    }
    @Override
    public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
      Path relativePath = workingDir.relativize(file);
      final HashCode fileHash = hashPath(file, hashFunction.newHasher());
      if (!seen.get(relativePath).contains(fileHash)) {
        seen.get(relativePath).add(fileHash);
        // TODO(bazel-team): Change to a symlink when the AOSP merge code supports symlinks.
        Files.copy(file, newRoot.resolve(relativePath));
        // Files.createSymbolicLink(newRoot.resolve(workingDir.relativize(file)), file);
      } else {
        LOGGER.warning(String.format("Duplicated file %s [%s]", relativePath, file));
      }
      return super.visitFile(file, attrs);
    }
  }

  private static HashCode hashPath(Path file, final Hasher hasher) throws IOException {
    byte[] tmpBuffer = new byte[512];
    final InputStream in = Files.newInputStream(file);
    for (int read = in.read(tmpBuffer); read > 0; read = in.read(tmpBuffer)) {
      hasher.putBytes(tmpBuffer, 0, read);
    }
    final HashCode fileHash = hasher.hash();
    in.close();
    return fileHash;
  }

  private final Multimap<Path, HashCode> seen;
  private final HashFunction hashFunction;
  private final Path out;
  private final Path workingDirectory;

  public FileDeDuplicator(HashFunction hashFunction, Path out, Path workingDirectory) {
    this.hashFunction = hashFunction;
    this.workingDirectory = workingDirectory;
    this.seen = HashMultimap.create();
    this.out = out;
  }

  private ImmutableList<Path> conditionallyCopy(ImmutableList<Path> roots)
      throws IOException {
    final Builder<Path> builder = ImmutableList.builder();
    for (Path root : roots) {
      Preconditions.checkArgument(root.startsWith(workingDirectory),
          root + " must start with root " + workingDirectory  + " from " + roots);
      Preconditions.checkArgument(!root.equals(workingDirectory),
          "Cannot deduplicate root directory: " + root + " from " + roots);
      if (!seen.containsKey(root)) {
        seen.put(root, null);
        final Path newRoot = out.resolve(workingDirectory.relativize(root));
        Files.walkFileTree(root, ImmutableSet.of(FileVisitOption.FOLLOW_LINKS), Integer.MAX_VALUE,
            new ConditionalCopyVisitor(newRoot, root, seen, hashFunction));
        builder.add(newRoot);
      } else {
        // Duplicated directories are ok -- multiple files from different libraries
        // can reside in the same directory, but duplicate files should not be seen mulitple times.
      }
    }
    return builder.build();
  }

  @Override
  public ImmutableList<Path> modify(ImmutableList<Path> directories) {
    try {
      return conditionallyCopy(directories);
    } catch (IOException e) {
      throw Throwables.propagate(e);
    }
  }
}
