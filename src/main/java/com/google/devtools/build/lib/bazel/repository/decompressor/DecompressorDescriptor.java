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

package com.google.devtools.build.lib.bazel.repository.decompressor;

import static java.util.Objects.requireNonNull;

import com.google.auto.value.AutoBuilder;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.UnixGlob;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Pattern;

/**
 * Description of an archive to be decompressed.
 *
 * @param context The context in which this decompression is happening. Should only be used for
 *     reporting.
 */
public record DecompressorDescriptor(
    String context,
    Path archivePath,
    Path destinationPath,
    Optional<String> prefix,
    ImmutableMap<String, String> renameFiles,
    ImmutableList<String> includes,
    ImmutableList<String> excludes) {
  public DecompressorDescriptor {
    requireNonNull(context, "context");
    requireNonNull(archivePath, "archivePath");
    requireNonNull(destinationPath, "destinationPath");
    requireNonNull(prefix, "prefix");
    requireNonNull(renameFiles, "renameFiles");
    requireNonNull(includes, "includes");
    requireNonNull(excludes, "excludes");
  }

  public static Builder builder() {
    return new AutoBuilder_DecompressorDescriptor_Builder()
        .setContext("")
        .setRenameFiles(ImmutableMap.of())
        .setIncludes(ImmutableList.of())
        .setExcludes(ImmutableList.of());
  }

  /**
   * Returns if a given archive entry should be skipped for decompression.
   *
   * <p>This follows the BSD tar logic - {@link #includes} is the list of glob patterns that should
   * be decompressed. If no inclusions are specified, all entries are extracted. {@link #excludes}
   * takes precedence over inclusions.
   *
   * @param archiveEntry The filepath of the archive entry.
   * @param patternCache A cache for glob patterns.
   * @return {@code true} if the entry should be skipped.
   */
  public boolean skipArchiveEntry(String archiveEntry, Map<String, Pattern> patternCache) {
    if (!includes.isEmpty()) {
      boolean include = false;
      for (String includePattern : includes) {
        if (UnixGlob.matches(includePattern, archiveEntry, patternCache)) {
          include = true;
          break;
        }
      }
      if (!include) {
        return true;
      }
    }

    for (String excludePattern : excludes) {
      if (UnixGlob.matches(excludePattern, archiveEntry, patternCache)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns if a given archive entry should be skipped for decompression.
   *
   * @see #skipArchiveEntry(String, Map)
   */
  public boolean skipArchiveEntry(String archiveEntry) {
    return skipArchiveEntry(archiveEntry, null);
  }

  /** Builder for describing the file to be decompressed. */
  @AutoBuilder
  public abstract static class Builder {

    public abstract Builder setContext(String context);

    public abstract Builder setArchivePath(Path archivePath);

    public abstract Builder setDestinationPath(Path destinationPath);

    public abstract Builder setPrefix(String prefix);

    public abstract Builder setRenameFiles(Map<String, String> renameFiles);

    public abstract Builder setIncludes(List<String> includes);

    public abstract Builder setExcludes(List<String> excludes);

    public abstract DecompressorDescriptor build();
  }
}
