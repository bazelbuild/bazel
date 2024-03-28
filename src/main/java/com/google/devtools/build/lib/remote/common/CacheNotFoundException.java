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
import com.google.common.base.Strings;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * An exception to indicate cache misses. TODO(olaola): have a class of checked
 * RemoteCacheExceptions.
 */
public final class CacheNotFoundException extends IOException {
  private final Digest missingDigest;
  @Nullable private String filename;

  public CacheNotFoundException(Digest missingDigest) {
    this.missingDigest = missingDigest;
  }

  public CacheNotFoundException(Digest missingDigest, String filename) {
    this.missingDigest = missingDigest;
    this.filename = filename;
  }

  public void setFilename(@Nullable String filename) {
    this.filename = filename;
  }

  public Digest getMissingDigest() {
    return missingDigest;
  }

  @Override
  public String getMessage() {
    String message =
        "Missing digest: " + missingDigest.getHash() + "/" + missingDigest.getSizeBytes();
    if (!Strings.isNullOrEmpty(filename)) {
      message += " for " + filename;
    }
    return message;
  }
}
