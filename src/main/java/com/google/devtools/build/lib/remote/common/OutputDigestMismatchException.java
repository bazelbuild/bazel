// Copyright 2021 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import javax.annotation.Nullable;

/** An exception to indicate the digest of downloaded output does not match the expected value. */
public class OutputDigestMismatchException extends IOException {
  private final Digest expected;
  @Nullable private final Digest actual;
  private final long receivedSizeBytes;

  private Path localPath;
  private String outputPath;

  public OutputDigestMismatchException(Digest expected, Digest actual) {
    this.expected = expected;
    this.actual = actual;
    this.receivedSizeBytes = actual.getSizeBytes();
  }

  /**
   * Indicates a download whose digest was not verified, but whose size doesn't match the size of
   * the expected digest.
   */
  public OutputDigestMismatchException(Digest expected, long receivedSizeBytes) {
    this.expected = expected;
    this.actual = null;
    this.receivedSizeBytes = receivedSizeBytes;
  }

  public void setOutputPath(String outputPath) {
    this.outputPath = outputPath;
  }

  public String getOutputPath() {
    return outputPath;
  }

  public Path getLocalPath() {
    return localPath;
  }

  public void setLocalPath(Path localPath) {
    this.localPath = localPath;
  }

  @Override
  public String getMessage() {
    if (actual == null) {
      return String.format(
          "Output %s download failed: Expected digest '%s/%d', but received %d bytes.",
          outputPath, expected.getHash(), expected.getSizeBytes(), receivedSizeBytes);
    }
    return String.format(
        "Output %s download failed: Expected digest '%s/%d' does not match "
            + "received digest '%s/%d'.",
        outputPath,
        expected.getHash(),
        expected.getSizeBytes(),
        actual.getHash(),
        actual.getSizeBytes());
  }
}
