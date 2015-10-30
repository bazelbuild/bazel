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
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.zip.GZIPInputStream;

import javax.annotation.Nullable;

/**
 * Creates a  repository by unarchiving a .tar.gz file.
 */
public class TarGzFunction implements SkyFunction {

  public static final SkyFunctionName NAME = SkyFunctionName.create("TAR_GZ_FUNCTION");

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws RepositoryFunctionException {
    DecompressorDescriptor descriptor = (DecompressorDescriptor) skyKey.argument();
    Optional<String> prefix = descriptor.prefix();
    boolean foundPrefix = false;

    try (GZIPInputStream gzipStream = new GZIPInputStream(
        new FileInputStream(descriptor.archivePath().getPathFile()))) {
      TarArchiveInputStream tarStream = new TarArchiveInputStream(gzipStream);
      TarArchiveEntry entry;
      while ((entry = tarStream.getNextTarEntry()) != null) {
        StripPrefixedPath entryPath = StripPrefixedPath.maybeDeprefix(entry.getName(), prefix);
        foundPrefix = foundPrefix || entryPath.foundPrefix();
        if (entryPath.skip()) {
          continue;
        }

        Path filename = descriptor.repositoryPath().getRelative(entryPath.getPathFragment());
        FileSystemUtils.createDirectoryAndParents(filename.getParentDirectory());
        if (entry.isDirectory()) {
          FileSystemUtils.createDirectoryAndParents(filename);
        } else {
          if (entry.isSymbolicLink()) {
            PathFragment linkName = new PathFragment(entry.getLinkName());
            if (linkName.isAbsolute()) {
              linkName = linkName.relativeTo(PathFragment.ROOT_DIR);
              linkName = descriptor.repositoryPath().getRelative(linkName).asFragment();
            }
            FileSystemUtils.ensureSymbolicLink(filename, linkName);
          } else {
            Files.copy(
                tarStream, filename.getPathFile().toPath(), StandardCopyOption.REPLACE_EXISTING);
            filename.chmod(entry.getMode());
          }
        }
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    if (prefix.isPresent() && !foundPrefix) {
      throw new RepositoryFunctionException(
          new IOException("Prefix " + prefix.get() + " was given, but not found in the archive"),
          Transience.PERSISTENT);
    }

    return new DecompressorValue(descriptor.repositoryPath());
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

}
