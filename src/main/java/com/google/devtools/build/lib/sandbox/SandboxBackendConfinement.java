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

package com.google.devtools.build.lib.sandbox;

import com.google.devtools.build.lib.util.OS;

/**
 * Bazel's built-in OS jails for sandbox backends. Which one to apply is Bazel's decision, not the
 * backend's: a backend either brings its own jail (a {@code CustomConfinement} on the wire) or
 * leaves confinement to Bazel, in which case {@link #platformDefault} picks the mechanism. Purely
 * Bazel-internal — it never travels on the wire.
 */
final class SandboxBackendConfinement {
  private SandboxBackendConfinement() {}

  /** A built-in OS jail Bazel knows how to apply. */
  enum Builtin {
    /** No OS jail — only the backend's filesystem view. */
    NONE,
    /** macOS Seatbelt (the {@code sandbox-exec} / TrustedBSD MAC layer). */
    SEATBELT,
    /** Linux namespaces (user/mount/pid), via the {@code linux-sandbox} helper. */
    LINUX_NAMESPACES,
  }

  /**
   * The jail to apply when the backend leaves confinement to Bazel: the strongest mechanism
   * available on this host ({@link Builtin#NONE} where none exists).
   */
  static Builtin platformDefault() {
    return switch (OS.getCurrent()) {
      case DARWIN -> Builtin.SEATBELT;
      case LINUX -> Builtin.LINUX_NAMESPACES;
      default -> Builtin.NONE;
    };
  }
}
