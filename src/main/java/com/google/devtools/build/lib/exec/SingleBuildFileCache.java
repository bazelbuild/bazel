// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.Maps;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.DigestOfDirectoryException;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
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
  private final LoadingCache<ActionInput, Pair<ByteString, IOException>> pathToDigest =
      CacheBuilder.newBuilder()
      // We default to 10 disk read threads, but we don't expect them all to edit the map
      // simultaneously.
      .concurrencyLevel(8)
      // Even small-ish builds, as of 11/21/2011 typically have over 10k artifacts, so it's
      // unlikely that this default will adversely affect memory in most cases.
      .initialCapacity(10000)
      .build(new CacheLoader<ActionInput, Pair<ByteString, IOException>>() {
        @Override
        public Pair<ByteString, IOException> load(ActionInput input) {
          Path path = null;
          try {
            path = fs.getPath(fullPath(input));
            BaseEncoding hex = BaseEncoding.base16().lowerCase();
            ByteString digest = ByteString.copyFrom(
                hex.encode(path.getMD5Digest())
                   .getBytes(US_ASCII));
            pathToBytes.put(input, path.getFileSize());
            // Inject reverse mapping. Doing this unconditionally in getDigest() showed up
            // as a hotspot in CPU profiling.
            digestToPath.put(digest, input);
            return Pair.of(digest, null);
          } catch (IOException e) {
            if (path != null && path.isDirectory()) {
              pathToBytes.put(input, 0L);
              return Pair.<ByteString, IOException>of(null, new DigestOfDirectoryException(
                  "Input is a directory: " + input.getExecPathString()));
            }

            // Put value into size map to avoid trying to read file again later.
            pathToBytes.put(input, 0L);
            return Pair.of(null, e);
          }
        }
      });

  private final Map<ByteString, ActionInput> digestToPath = Maps.newConcurrentMap();

  private final Map<ActionInput, Long> pathToBytes = Maps.newConcurrentMap();

  @Nullable
  @Override
  public ActionInput getInputFromDigest(ByteString digest) {
    return digestToPath.get(digest);
  }

  @Override
  public Path getInputPath(ActionInput input) {
    if (input instanceof Artifact) {
      return ((Artifact) input).getPath();
    }
    return fs.getPath(fullPath(input));
  }

  @Override
  public long getSizeInBytes(ActionInput input) throws IOException {
    // TODO(bazel-team): this only works if pathToDigest has already been called.
    Long sz = pathToBytes.get(input);
    if (sz != null) {
      return sz;
    }
    Path path = fs.getPath(fullPath(input));
    sz = path.getFileSize();
    pathToBytes.put(input, sz);
    return sz;
  }

  @Override
  public ByteString getDigest(ActionInput input) throws IOException {
    Pair<ByteString, IOException> result = pathToDigest.getUnchecked(input);
    if (result.second != null) {
      throw result.second;
    }
    return result.first;
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
    return relPath.startsWith("/") ? relPath : new File(cwd, relPath).getPath();
  }
}
