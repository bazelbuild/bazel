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

import static java.nio.charset.StandardCharsets.US_ASCII;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.Maps;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.DigestOfDirectoryException;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import javax.annotation.Nullable;
import javax.annotation.concurrent.ThreadSafe;

/**
 * An in-memory cache to ensure we do I/O for source files only once during a single build.
 *
 * <p>Simply maintains a two-way cached mapping from digest <--> filename that may be populated
 * only once.
 */
@ThreadSafe
public class SingleBuildFileCache implements ActionInputFileCache {

  private final String cwd;
  private final FileSystem fs;

  public SingleBuildFileCache(String cwd, FileSystem fs) {
    this.fs = Preconditions.checkNotNull(fs);
    this.cwd = Preconditions.checkNotNull(cwd);
  }

  // If we can't get the digest, we store the exception. This avoids extra file IO for files
  // that are allowed to be missing, as we first check a likely non-existent content file
  // first.  Further we won't need to unwrap the exception in getDigest().
  private final LoadingCache<ActionInput, ActionInputMetadata> pathToMetadata =
      CacheBuilder.newBuilder()
      // We default to 10 disk read threads, but we don't expect them all to edit the map
      // simultaneously.
      .concurrencyLevel(8)
      // Even small-ish builds, as of 11/21/2011 typically have over 10k artifacts, so it's
      // unlikely that this default will adversely affect memory in most cases.
      .initialCapacity(10000)
      .build(new CacheLoader<ActionInput, ActionInputMetadata>() {
        @Override
        public ActionInputMetadata load(ActionInput input) {
          Path path = null;
          try {
            path = fs.getPath(fullPath(input));
            byte[] digest = path.getDigest();
            BaseEncoding hex = BaseEncoding.base16().lowerCase();
            ByteString hexDigest = ByteString.copyFrom(hex.encode(digest).getBytes(US_ASCII));
            // Inject reverse mapping. Doing this unconditionally in getDigest() showed up
            // as a hotspot in CPU profiling.
            digestToPath.put(hexDigest, input);
            return new ActionInputMetadata(digest, path.getFileSize());
          } catch (IOException e) {
            if (path != null && path.isDirectory()) {
              // TODO(bazel-team): This is rather presumptuous- it could have been another type of
              // IOException.
              return new ActionInputMetadata(new DigestOfDirectoryException(
                  "Input is a directory: " + input.getExecPathString()));
            } else {
              return new ActionInputMetadata(e);
            }
          }
        }
      });

  private final Map<ByteString, ActionInput> digestToPath = Maps.newConcurrentMap();

  @Nullable
  @Override
  public ActionInput getInputFromDigest(ByteString digest) {
    return digestToPath.get(digest);
  }

  @Override
  public Path getInputPath(ActionInput input) {
    return fs.getPath(fullPath(input));
  }

  @Override
  public long getSizeInBytes(ActionInput input) throws IOException {
    return pathToMetadata.getUnchecked(input).getSize();
  }

  @Override
  public byte[] getDigest(ActionInput input) throws IOException {
    return pathToMetadata.getUnchecked(input).getDigest();
  }

  @Override
  public boolean isFile(Artifact input) {
    // We shouldn't fall back on this functionality ever.
    throw new UnsupportedOperationException();
  }

  @Override
  public boolean contentsAvailableLocally(ByteString digest) {
    return digestToPath.containsKey(digest);
  }

  /**
   * Creates a File object that refers to fileName, if fileName is an absolute path. Otherwise,
   * returns a File object that refers to the fileName appended to the (absolute) current working
   * directory.
   */
  private String fullPath(ActionInput input) {
    String relPath = input.getExecPathString();
    return PathFragment.create(relPath).isAbsolute() ? relPath : new File(cwd, relPath).getPath();
  }

  /** Container class for caching I/O around ActionInputs. */
  private static class ActionInputMetadata {
    private final byte[] digest;
    private final long size;
    private final IOException exceptionOnAccess;

    /** Constructor for a successful lookup. */
    ActionInputMetadata(byte[] digest, long size) {
      this.digest = digest;
      this.size = size;
      this.exceptionOnAccess = null;
    }

    /** Constructor for a failed lookup, size will be 0. */
    ActionInputMetadata(IOException exceptionOnAccess) {
      this.exceptionOnAccess = exceptionOnAccess;
      this.digest = null;
      this.size = 0;
    }

    /** Returns digest or throws the exception encountered calculating it/ */
    byte[] getDigest() throws IOException {
      maybeRaiseException();
      return digest;
    }

    /** Returns the size. */
    long getSize() throws IOException {
      maybeRaiseException();
      return size;
    }

    private void maybeRaiseException() throws IOException {
      if (exceptionOnAccess != null) {
        throw exceptionOnAccess;
      }
    }
  }
}
