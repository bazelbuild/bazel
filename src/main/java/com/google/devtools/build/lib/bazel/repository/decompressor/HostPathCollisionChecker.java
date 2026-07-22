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

package com.google.devtools.build.lib.bazel.repository.decompressor;

import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.text.Normalizer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Rejects two archive entries whose extraction paths differ byte-for-byte but resolve to the same
 * file on the host filesystem.
 *
 * <p>Bazel keeps archive entry names as raw bytes and compares them byte-for-byte, which is exact
 * on Linux and Windows. macOS/APFS resolves names through Unicode canonical normalization (so an
 * NFC and an NFD spelling of an accented name are one file) and, on the default case-insensitive
 * volume, through full case folding (so {@code SECRET} and {@code secret}, the sharp s and {@code
 * ss}, and the Latin ligatures and their component letters are one file). Two such entries would
 * otherwise extract on top of each other, letting the second silently replace the contents written
 * for the first under a name the byte-level code never sees as equal.
 *
 * <p>The containment checks in the decompressors stay byte-level: they are conservative in the
 * escaping direction and folding them would only widen what they accept. This check instead closes
 * the collision that the byte-level logic cannot see.
 *
 * <p>One instance tracks a single extraction and is not thread-safe.
 */
final class HostPathCollisionChecker {

  private final Map<String, String> firstPathByResolvedForm = new HashMap<>();
  private final boolean normalizeUnicode;
  private final boolean foldCase;

  @VisibleForTesting
  HostPathCollisionChecker(boolean normalizeUnicode, boolean foldCase) {
    this.normalizeUnicode = normalizeUnicode;
    this.foldCase = foldCase;
  }

  /**
   * Returns a checker configured for the host the Bazel server runs on. It is a no-op on every
   * platform except macOS. On macOS canonical normalization always applies; case folding applies
   * unless {@code -Dbazel.darwin.case_sensitive=true} declares the volume case-sensitive.
   */
  static HostPathCollisionChecker create() {
    if (OS.getCurrent() != OS.DARWIN) {
      return new HostPathCollisionChecker(false, false);
    }
    return new HostPathCollisionChecker(true, !Boolean.getBoolean("bazel.darwin.case_sensitive"));
  }

  /**
   * Records {@code relativePath} as an extraction target and throws if a byte-distinct target that
   * resolves to the same host file was already recorded during this extraction. Two entries that
   * share the exact same bytes are left to the existing last-writer-wins behavior.
   */
  void checkAndRecord(PathFragment relativePath) throws IOException {
    String path = relativePath.getPathString();
    String existing = firstPathByResolvedForm.putIfAbsent(resolvedForm(path), path);
    if (existing != null && !existing.equals(path)) {
      throw new IOException(
          String.format(
              "Failed to extract %s, it resolves to the same file as %s on this filesystem",
              path, existing));
    }
  }

  /**
   * Maps a Bazel-internal path string to a representative that is equal for two paths iff the host
   * filesystem resolves them to the same file.
   */
  @VisibleForTesting
  String resolvedForm(String path) {
    if (!normalizeUnicode) {
      return path;
    }
    String unicode = recoverUnicode(path);
    if (unicode == null) {
      // Not valid UTF-8; the filesystem treats such a name as opaque bytes, so compare raw bytes.
      return path;
    }
    String resolved = Normalizer.normalize(unicode, Normalizer.Form.NFC);
    if (foldCase) {
      // Upper- then lower-casing folds case across the whole string the way a case-insensitive
      // volume equates names, including the sharp s (to "ss") and the Latin ligatures.
      resolved = resolved.toUpperCase(Locale.ROOT).toLowerCase(Locale.ROOT);
    }
    return resolved;
  }

  /**
   * Recovers the Unicode name from Bazel's internal path representation, in which archive entry
   * names are the original bytes held one per char as ISO-8859-1. Returns the argument unchanged if
   * it already holds decoded Unicode (any char above 0xFF), or null if the bytes are not UTF-8.
   */
  private static String recoverUnicode(String internalPath) {
    for (int i = 0; i < internalPath.length(); i++) {
      if (internalPath.charAt(i) > 0xFF) {
        return internalPath;
      }
    }
    byte[] bytes = internalPath.getBytes(ISO_8859_1);
    String candidate = new String(bytes, UTF_8);
    return Arrays.equals(candidate.getBytes(UTF_8), bytes) ? candidate : null;
  }
}
