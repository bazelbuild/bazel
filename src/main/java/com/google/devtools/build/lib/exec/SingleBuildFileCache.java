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

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.DigestOfDirectoryException;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.SymlinkArtifactValue;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;
import javax.annotation.concurrent.ThreadSafe;

/**
 * An in-memory cache to ensure we do I/O for source files only once during a single build.
 *
 * <p>Simply maintains a cached mapping from filename to metadata that may be populated only once.
 */
@ThreadSafe
public class SingleBuildFileCache implements MetadataProvider {
  private final Path execRoot;

  public SingleBuildFileCache(String cwd, FileSystem fs) {
    this.execRoot = fs.getPath(cwd);
  }

  // If we can't get the digest, we store the exception. This avoids extra file IO for files
  // that are allowed to be missing, as we first check a likely non-existent content file
  // first.  Further we won't need to unwrap the exception in getDigest().
  private final Cache<String, ActionInputMetadata> pathToMetadata =
      CacheBuilder.newBuilder()
          // We default to 10 disk read threads, but we don't expect them all to edit the map
          // simultaneously.
          .concurrencyLevel(8)
          // Even small-ish builds, as of 11/21/2011 typically have over 10k artifacts, so it's
          // unlikely that this default will adversely affect memory in most cases.
          .initialCapacity(10000)
          .build();

  @Override
  public FileArtifactValue getMetadata(ActionInput input) throws IOException {
    try {
      return pathToMetadata
          .get(
              input.getExecPathString(),
              () -> {
                Path path = ActionInputHelper.toInputPath(input, execRoot);
                try {
                  FileArtifactValue metadata = FileArtifactValue.create(path);

                  boolean isDirectory = metadata.getType().isDirectory();
                  isDirectory = isDirectory || metadata.getType().isSymlink() &&
                      ((SymlinkArtifactValue) metadata).getResolvedValue().getType().isDirectory();
                  if (isDirectory) {
                    throw new DigestOfDirectoryException(
                        "Input is a directory: " + input.getExecPathString());
                  }
                  return new ActionInputMetadata(input, metadata);
                } catch (IOException e) {
                  return new ActionInputMetadata(input, e);
                }
              })
          .getMetadata();
    } catch (ExecutionException e) {
      throw new IllegalStateException("Unexpected cache loading error", e); // Should never happen.
    }
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
