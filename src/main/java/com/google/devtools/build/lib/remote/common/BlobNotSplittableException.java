// Copyright 2026 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.remote.util.DigestUtil;
import java.io.IOException;

/**
 * Indicates that the server cannot describe a blob as a sequence of chunks, either because it does
 * not implement {@code SplitBlob} or because it has no chunks for this particular blob.
 *
 * <p>This says nothing about whether the blob itself is available: the caller is expected to fall
 * back to downloading it as a whole. It is only thrown before any chunk data has been produced, so
 * that such a fallback can safely start from the beginning of the blob.
 */
public final class BlobNotSplittableException extends IOException {
  private final Digest blobDigest;

  public BlobNotSplittableException(Digest blobDigest) {
    this.blobDigest = blobDigest;
  }

  public Digest getBlobDigest() {
    return blobDigest;
  }

  @Override
  public String getMessage() {
    return "Not splittable into chunks: %s".formatted(DigestUtil.toString(blobDigest));
  }
}
