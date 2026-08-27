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

package com.google.devtools.build.lib.unix;

import com.google.devtools.build.lib.skybridge.SkybridgeInterface;

/** Exception thrown when a POSIX filesystem operation fails. */
@SkybridgeInterface
public final class NativePosixFilesException extends Exception {

  /** Typesafe representation of POSIX error codes. */
  public static final class PosixError {
    public static final PosixError ENOENT = new PosixError("ENOENT");
    public static final PosixError EACCES = new PosixError("EACCES");
    public static final PosixError ELOOP = new PosixError("ELOOP");
    public static final PosixError ETIMEDOUT = new PosixError("ETIMEDOUT");
    public static final PosixError OTHER = new PosixError("OTHER");

    private final String name;

    private PosixError(String name) {
      this.name = name;
    }

    @Override
    public String toString() {
      return name;
    }
  }

  private final PosixError error;

  public NativePosixFilesException(String message, PosixError error) {
    super(message);
    this.error = error != null ? error : PosixError.OTHER;
  }

  public PosixError getError() {
    return error;
  }
}
