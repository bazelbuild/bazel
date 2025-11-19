// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote.common;

import build.bazel.remote.execution.v2.Digest;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * An exception to indicate cache misses. TODO(olaola): have a class of checked
 * RemoteCacheExceptions.
 */
public final class CacheNotFoundException extends IOException {
  private final Digest missingDigest;
  @Nullable private PathFragment execPath;
  @Nullable private String filename;

  public CacheNotFoundException(Digest missingDigest) {
    this.missingDigest = missingDigest;
  }

  public CacheNotFoundException(Digest missingDigest, PathFragment execPath) {
    this.missingDigest = missingDigest;
    this.execPath = execPath;
  }

  public CacheNotFoundException(Digest missingDigest, String filename) {
    this.missingDigest = missingDigest;
    this.filename = filename;
  }

  // The exec path of the artifact that was not found in the cache if the missing cache entry
  // corresponds to one.
  public void setExecPath(PathFragment execPath) {
    this.execPath = execPath;
  }

  // A human-readable filename only used in error messages.
  public void setFilename(String filename) {
    this.filename = filename;
  }

  public Digest getMissingDigest() {
    return missingDigest;
  }

  @Nullable
  public PathFragment getExecPath() {
    return execPath;
  }

  @Nullable
  public String getFilename() {
    return filename;
  }

  @Override
  public String getMessage() {
    String message =
        "Missing digest: " + missingDigest.getHash() + "/" + missingDigest.getSizeBytes();
    if (execPath != null || filename != null) {
      // Prefer filename over execPath as it contains strictly more information.
      message += " for " + (filename != null ? filename : execPath);
    }
    return message;
  }
}
