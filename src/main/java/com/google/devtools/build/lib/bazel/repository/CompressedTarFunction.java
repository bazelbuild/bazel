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

import static com.google.devtools.build.lib.bazel.repository.StripPrefixedPath.maybeDeprefixSymlink;

import com.google.common.base.Optional;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.bazel.repository.DecompressorValue.Decompressor;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.*;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;

/**
 * Common code for unarchiving a compressed TAR file.
 */
public abstract class CompressedTarFunction implements Decompressor {
  protected abstract InputStream getDecompressorStream(DecompressorDescriptor descriptor)
          throws IOException;

  @Override
  public Path decompress(DecompressorDescriptor descriptor)
          throws InterruptedException, IOException {
    if (Thread.interrupted()) {
      throw new InterruptedException();
    }
    Optional<String> prefix = descriptor.prefix();
    boolean foundPrefix = false;
    Set<String> availablePrefixes = new HashSet<>();
    // Map to store for every nonexistent target a list of symlink Tar entries that point to it
    Map<String, List<TarArchiveEntry>> symlinkDepMap = new HashMap<>();

    try (InputStream decompressorStream = getDecompressorStream(descriptor)) {
      TarArchiveInputStream tarStream = new TarArchiveInputStream(decompressorStream);
      TarArchiveEntry entry;
      while ((entry = tarStream.getNextTarEntry()) != null) {
        foundPrefix = decompressEntry(entry, tarStream, descriptor, prefix, foundPrefix,
                availablePrefixes, symlinkDepMap, false) || foundPrefix;

        if (Thread.interrupted()) {
          throw new InterruptedException();
        }
      }

      if(!symlinkDepMap.isEmpty()) {
        // If some symlinks targets are not resolved, handle the symbolic links entries by creating dangling junctions
        for (Map.Entry<String, List<TarArchiveEntry>> symlinkDepEntry : symlinkDepMap.entrySet()) {
          for(TarArchiveEntry tarEntry : symlinkDepEntry.getValue()) {
            foundPrefix = decompressEntry(tarEntry, tarStream, descriptor, prefix, foundPrefix,
                    availablePrefixes, symlinkDepMap, true) || foundPrefix;
          }
        }
      }

      if (prefix.isPresent() && !foundPrefix) {
        throw new CouldNotFindPrefixException(prefix.get(), availablePrefixes);
      }
    }

    return descriptor.repositoryPath();
  }

  /* Decompresses the given tar archive entry if {@code forceDecompress} option is set or the entry is a symbolic link
   *  whose target exists or it is not a symbolic link at all.
   *
   *  If the symbolic link target does not exist, it adds an entry for it in {@code symlinkDepMap}. The symlink
   *  entries will be decompressed once their target entry is decompressed.
   *
   *  Returns if the {@code prefix} is found in the entry or not.
   *
   *  This is done to avoid creating dangling junctions for symlinks whose targets are part of
   *  the archive being decompressed and come after them in the archive stream.
   * */
  private boolean decompressEntry(TarArchiveEntry entry, TarArchiveInputStream tarStream,
                                  DecompressorDescriptor descriptor,
                                  Optional<String> prefix, boolean foundPrefix, Set<String> availablePrefixes,
                                  Map<String, List<TarArchiveEntry>> symlinkDepMap, boolean forceDecompress) throws IOException {

    StripPrefixedPath entryPath = StripPrefixedPath.maybeDeprefix(entry.getName(), prefix);
    Path filePath = descriptor.repositoryPath().getRelative(entryPath.getPathFragment());

    if(entry.isSymbolicLink() && !forceDecompress) {
      // If the entry is a symbolic link and the {@code forceDecompress} option is not set, check if the link target exists
      PathFragment targetFragment = PathFragment.create(entry.getLinkName());
      targetFragment = maybeDeprefixSymlink(targetFragment, prefix, descriptor.repositoryPath());
      String targetPathStr;
      if(targetFragment.isAbsolute()) {
        targetPathStr = targetFragment.getPathString();
      }
      else {
        targetPathStr = filePath.getParentDirectory().getRelative(targetFragment).getPathString();
      }

      if(!new File(targetPathStr).exists()) {
        // If the target does not exist, add the symlink to the list of symlink entries waiting on the target
        List<TarArchiveEntry> symlinks = symlinkDepMap.getOrDefault(targetPathStr, new LinkedList<>());
        symlinks.add(entry);
        symlinkDepMap.put(targetPathStr, symlinks);
        return false;
      }
    }

    foundPrefix = foundPrefix || entryPath.foundPrefix();
    if (prefix.isPresent() && !foundPrefix) {
      Optional<String> suggestion =
              CouldNotFindPrefixException.maybeMakePrefixSuggestion(entryPath.getPathFragment());
      if (suggestion.isPresent()) {
        availablePrefixes.add(suggestion.get());
      }
    }

    if (entryPath.skip()) {
      return foundPrefix;
    }

    FileSystemUtils.createDirectoryAndParents(filePath.getParentDirectory());
    if (entry.isDirectory()) {
      FileSystemUtils.createDirectoryAndParents(filePath);
    } else {
      if (entry.isSymbolicLink() || entry.isLink()) {
        PathFragment targetName = PathFragment.create(entry.getLinkName());
        targetName = maybeDeprefixSymlink(targetName, prefix, descriptor.repositoryPath());
        if (entry.isSymbolicLink()) {
          if (filePath.exists()) {
            filePath.delete();
          }
          FileSystemUtils.ensureSymbolicLink(filePath, targetName);
        } else {
          Path targetPath = descriptor.repositoryPath().getRelative(targetName);
          if (filePath.equals(targetPath)) {
            // The behavior here is semantically different, depending on whether the underlying
            // filesystem is case-sensitive or case-insensitive. However, it is effectively the
            // same: we drop the link entry.
            // * On a case-sensitive filesystem, this is a hardlink to itself, such as GNU tar
            //   creates when given repeated files. We do nothing since the link already exists.
            // * On a case-insensitive filesystem, we may be extracting a differently-cased
            //   hardlink to the same file (such as when extracting an archive created on a
            //   case-sensitive filesystem). GNU tar, for example, will drop the new link entry.
            //   BSD tar on MacOS X (by default case-insensitive) errors and aborts extraction.
          } else {
            if (filePath.exists()) {
              filePath.delete();
            }
            FileSystemUtils.createHardLink(filePath, targetPath);
          }
        }
      } else {
        try (OutputStream out = filePath.getOutputStream()) {
          ByteStreams.copy(tarStream, out);
        }
        filePath.chmod(entry.getMode());

        // This can only be done on real files, not links, or it will skip the reader to
        // the next "real" file to try to find the mod time info.
        Date lastModified = entry.getLastModifiedDate();
        filePath.setLastModifiedTime(lastModified.getTime());
      }
    }

    String entryPathStr = filePath.getPathString();
    if(symlinkDepMap.containsKey(entryPathStr)) {
      // If the current entry has a list of symbolic links waiting for it, then start decompressing them
      for(TarArchiveEntry symlinkEntry: symlinkDepMap.get(entryPathStr)) {
        foundPrefix = decompressEntry(symlinkEntry, tarStream, descriptor, prefix, foundPrefix,
                availablePrefixes, symlinkDepMap, forceDecompress) || foundPrefix;
      }
      symlinkDepMap.remove(entryPathStr);
    }

    return foundPrefix;
  }
}