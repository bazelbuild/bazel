// Copyright 2022 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.runtime;

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Factory for creating {@link PathFragment}s from command-line options.
 *
 * <p>The difference between this and using {@link PathFragment#create(String)} directly is that
 * this factory replaces values starting with {@code %<name>%} with the corresponding (named) roots
 * (e.g., {@code %workspace%/foo} becomes {@code </path/to/workspace>/foo}).
 */
public final class CommandLinePathFactory {
  private static final Pattern REPLACEMENT_PATTERN = Pattern.compile("^(%([a-z_]+)%/)?([^%].*)$");

  private static final Splitter PATH_SPLITTER = Splitter.on(File.pathSeparator);

  private final FileSystem fileSystem;
  private final ImmutableMap<String, Path> roots;

  public CommandLinePathFactory(FileSystem fileSystem, ImmutableMap<String, Path> roots) {
    this.fileSystem = Preconditions.checkNotNull(fileSystem);
    this.roots = Preconditions.checkNotNull(roots);
  }

  /** Creates a {@link Path}. */
  public Path create(Map<String, String> env, String value) throws IOException {
    Preconditions.checkNotNull(env);
    Preconditions.checkNotNull(value);

    Matcher matcher = REPLACEMENT_PATTERN.matcher(value);
    Preconditions.checkArgument(matcher.matches());

    String rootName = matcher.group(2);
    PathFragment path = PathFragment.create(matcher.group(3));
    if (path.containsUplevelReferences()) {
      throw new IllegalArgumentException(
          String.format(
              Locale.US, "Path must not contain any uplevel references ('..'), got '%s'", value));
    }

    // Case 1: `path` is relative to a well-known root.
    if (!Strings.isNullOrEmpty(rootName)) {
      // The regex above cannot check that `value` is not of form `%foo%//abc` (group 2 will be
      // `foo` and group 3 will be `/abc`).
      Preconditions.checkArgument(!path.isAbsolute());

      Path root = roots.get(rootName);
      if (root == null) {
        throw new IllegalArgumentException(String.format(Locale.US, "Unknown root %s", rootName));
      }
      return root.getRelative(path);
    }

    // Case 2: `value` is an absolute path.
    if (path.isAbsolute()) {
      return fileSystem.getPath(path);
    }

    // Case 3: `value` is a relative path.
    //
    // Since relative paths from the command-line are ambiguous to where they are relative to (i.e.,
    // relative to the workspace?, the directory Bazel is running in? relative to the `.bazelrc` the
    // flag is from?), we only allow relative paths with a single segment (i.e., no `/`) and treat
    // it as relative to the user's `PATH`.
    if (path.segmentCount() > 1) {
      throw new IllegalArgumentException(
          "Path must either be absolute or not contain any path separators");
    }

    String pathVariable = env.getOrDefault("PATH", "");
    if (!Strings.isNullOrEmpty(pathVariable)) {
      for (String lookupPath : PATH_SPLITTER.split(pathVariable)) {
        Path maybePath = fileSystem.getPath(lookupPath).getRelative(path);
        if (maybePath.exists(Symlinks.FOLLOW)
            && maybePath.isFile(Symlinks.FOLLOW)
            && maybePath.isExecutable()) {
          return maybePath;
        }
      }
    }

    throw new FileNotFoundException(
        String.format(
            Locale.US, "Could not find file with name '%s' on PATH '%s'", path, pathVariable));
  }
}
