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
package com.google.devtools.build.lib.vfs;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.util.OS;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.Locale;

/**
 * Platform-aware containment check between two paths.
 *
 * <p>Pure {@link PathFragment#startsWith} suffices on Linux and Windows. On macOS / APFS, distinct
 * Java strings can resolve to the same inode via NFC/NFD normalization, ligature folding, and
 * case-insensitive matching, so a byte-prefix check alone misses real containment.
 *
 * <p>The {@link #DARWIN} policy is best-effort: Java's {@link Normalizer} ships with the JDK's
 * Unicode tables, which are not guaranteed to match Apple's kernel-level fold rules byte for
 * byte. Use it as a fast-fail signal alongside, not in place of, byte-level checks.
 */
public interface PathContainmentPolicy {

  /** Returns true iff {@code child} is contained in (or equals) {@code parent}. */
  boolean isContained(PathFragment child, PathFragment parent);

  /** Returns true iff {@code child} is contained in (or equals) {@code parent}. */
  default boolean isContained(
      com.google.devtools.build.lib.vfs.Path child,
      com.google.devtools.build.lib.vfs.Path parent) {
    return isContained(child.asFragment(), parent.asFragment());
  }

  /** Default policy: byte-level prefix containment. */
  PathContainmentPolicy DEFAULT = (child, parent) -> child.startsWith(parent);

  /** macOS / APFS policy: byte-level fast path, then NFKC + full case fold fallback. */
  PathContainmentPolicy DARWIN =
      new PathContainmentPolicy() {
        @Override
        public boolean isContained(PathFragment child, PathFragment parent) {
          if (child.startsWith(parent)) {
            return true;
          }
          if (child.isAbsolute() != parent.isAbsolute()) {
            return false;
          }
          String c = canonicalizeForDarwin(child.getPathString());
          String p = canonicalizeForDarwin(parent.getPathString());
          if (p.length() > c.length() || !c.startsWith(p)) {
            return false;
          }
          return c.length() == p.length() || p.equals("/") || c.charAt(p.length()) == '/';
        }
      };

  /**
   * Folds {@code path} into APFS-equivalence form: NFKC compatibility decomposition (handles
   * NFD/NFC variance and ligatures), followed by upper-then-lower casing to perform full Unicode
   * case folding (handles case-insensitive volumes).
   */
  @VisibleForTesting
  static String canonicalizeForDarwin(String path) {
    return Normalizer.normalize(path, Normalizer.Form.NFKC)
        .toUpperCase(Locale.ROOT)
        .toLowerCase(Locale.ROOT);
  }

  /**
   * Policy for the OS the Bazel server is running on.
   *
   * <p>On macOS, the case-fold component is dropped when the boot filesystem is detected as
   * case-sensitive (rare, but a supported APFS configuration), since folding case there would
   * accept symlink targets that the kernel would resolve to a different inode.
   */
  PathContainmentPolicy HOST_POLICY = forHost();

  static PathContainmentPolicy forOs(OS os) {
    return os == OS.DARWIN ? DARWIN : DEFAULT;
  }

  /**
   * Selects a policy for the running JVM. The {@code bazel.darwin.case_sensitive} system property
   * forces the byte-level policy on macOS for operators on case-sensitive APFS volumes; otherwise
   * a one-time filesystem probe of {@code java.io.tmpdir} decides.
   */
  private static PathContainmentPolicy forHost() {
    if (OS.getCurrent() != OS.DARWIN) {
      return DEFAULT;
    }
    if (Boolean.getBoolean("bazel.darwin.case_sensitive")) {
      return DEFAULT;
    }
    return probeCaseSensitive() ? DEFAULT : DARWIN;
  }

  /**
   * Returns true iff {@code java.io.tmpdir} is on a case-sensitive filesystem. Falls back to
   * {@code false} (assume case-insensitive, the macOS default) if the probe cannot be completed.
   */
  @VisibleForTesting
  static boolean probeCaseSensitive() {
    Path probe = null;
    try {
      probe = Files.createTempFile("bazel_case_probe_", ".tmp");
      String name = probe.getFileName().toString();
      Path upper = probe.resolveSibling(name.toUpperCase(Locale.ROOT));
      if (upper.equals(probe)) {
        // Generated name had no lower-case characters; cannot decide.
        return false;
      }
      // Case-sensitive: the upper-cased sibling does not exist, so Files.exists is false.
      // Case-insensitive: the FS resolves it to the same inode, so Files.exists is true and
      // isSameFile confirms the inode equality.
      if (!Files.exists(upper)) {
        return true;
      }
      return !Files.isSameFile(probe, upper);
    } catch (IOException | UnsupportedOperationException e) {
      return false;
    } finally {
      if (probe != null) {
        try {
          Files.deleteIfExists(probe);
        } catch (IOException ignored) {
          // best-effort cleanup
        }
      }
    }
  }
}
