// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.exec;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.DigestOfDirectoryException;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.XattrProvider;
import java.io.IOException;
import javax.annotation.Nullable;
import javax.annotation.concurrent.ThreadSafe;

/**
 * An in-memory cache to ensure we do I/O for source files only once during a single build.
 *
 * <p>Simply maintains a cached mapping from filename to metadata that may be populated only once.
 */
@ThreadSafe
public class SingleBuildFileCache implements InputMetadataProvider {

  private final Path execRoot;

  // If we can't get the digest, we store the exception. This avoids extra file IO for files
  // that are allowed to be missing, as we first check a likely non-existent content file
  // first.  Further we won't need to unwrap the exception in getDigest().
  private final Cache<String, ActionInputMetadata> pathToMetadata =
      Caffeine.newBuilder()
          // Even small-ish builds, as of 11/21/2011 typically have over 10k artifacts, so it's
          // unlikely that this default will adversely affect memory in most cases.
          .initialCapacity(10000)
          .build();
  private final XattrProvider xattrProvider;

  public SingleBuildFileCache(String cwd, FileSystem fs, XattrProvider xattrProvider) {
    this.xattrProvider = xattrProvider;
    this.execRoot = fs.getPath(cwd);
  }

  @Override
  public FileArtifactValue getInputMetadata(ActionInput input) throws IOException {
    // TODO(lberki): It would be nice to assert that only source files are passed here.
    // Unfortunately, that's not quite true at the moment and an unknown amount of work would be
    // needed to make that true.
    return pathToMetadata
        .get(
            input.getExecPathString(),
            execPath -> {
              Path path = ActionInputHelper.toInputPath(input, execRoot);
              FileArtifactValue metadata;
              try {
                metadata =
                    FileArtifactValue.createFromStat(
                        path,
                        // TODO(b/199940216): should we use syscallCache here since caching anyway?
                        path.stat(Symlinks.FOLLOW),
                        xattrProvider);
              } catch (IOException e) {
                return new ActionInputMetadata(input, e);
              }
              if (metadata.getType().isDirectory()) {
                return new ActionInputMetadata(
                    input, new DigestOfDirectoryException("Input is a directory: " + execPath));
              }
              return new ActionInputMetadata(input, metadata);
            })
        .getMetadata();
  }

  @Override
  @Nullable
  public ActionInput getInput(String execPath) {
    ActionInputMetadata metadata = pathToMetadata.getIfPresent(execPath);
    if (metadata == null) {
      return null;
    }
    return metadata.getInput();
  }

  /** Container class for caching I/O around ActionInputs. */
  private static class ActionInputMetadata {
    private final ActionInput input;
    private final FileArtifactValue metadata;
    private final IOException exceptionOnAccess;

    /** Constructor for a successful lookup. */
    ActionInputMetadata(ActionInput input, FileArtifactValue metadata) {
      this.input = input;
      this.metadata = metadata;
      this.exceptionOnAccess = null;
    }

    /** Constructor for a failed lookup, size will be 0. */
    ActionInputMetadata(ActionInput input, IOException exceptionOnAccess) {
      this.input = input;
      this.exceptionOnAccess = exceptionOnAccess;
      this.metadata = null;
    }

    FileArtifactValue getMetadata() throws IOException {
      maybeRaiseException();
      return metadata;
    }

    ActionInput getInput() {
      return input;
    }

    private void maybeRaiseException() throws IOException {
      if (exceptionOnAccess != null) {
        throw exceptionOnAccess;
      }
    }
  }
}
