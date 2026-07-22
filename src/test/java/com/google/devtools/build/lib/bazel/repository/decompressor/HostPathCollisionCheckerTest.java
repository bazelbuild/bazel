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
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link HostPathCollisionChecker}. Non-ASCII entry names use unicode escapes. */
@RunWith(JUnit4.class)
public final class HostPathCollisionCheckerTest {

  // Builds the internal path representation a tar/zip/7z entry name would have: the name's UTF-8
  // bytes held one per char as ISO-8859-1.
  private static PathFragment entry(String name) {
    return PathFragment.create(new String(name.getBytes(UTF_8), ISO_8859_1));
  }

  private static void assertCollision(
      boolean normalize, boolean foldCase, String first, String second) throws IOException {
    HostPathCollisionChecker checker = new HostPathCollisionChecker(normalize, foldCase);
    checker.checkAndRecord(entry(first));
    assertThrows(IOException.class, () -> checker.checkAndRecord(entry(second)));
  }

  private static void assertNoCollision(boolean normalize, boolean foldCase, String... names)
      throws IOException {
    HostPathCollisionChecker checker = new HostPathCollisionChecker(normalize, foldCase);
    for (String name : names) {
      checker.checkAndRecord(entry(name));
    }
  }

  @Test
  public void caseInsensitive_foldsSharpSLigaturesAndCase() throws Exception {
    assertCollision(true, true, "ss_output.txt", "\u00df_output.txt"); // ss vs sharp s
    assertCollision(true, true, "fi_c", "\ufb01_c"); // fi ligature
    assertCollision(true, true, "ff_d", "\ufb00_d");
    assertCollision(true, true, "fl_l", "\ufb02_l");
    assertCollision(true, true, "ffi_t", "\ufb03_t");
    assertCollision(true, true, "ffl_r", "\ufb04_r");
    assertCollision(true, true, "st_k", "\ufb05_k"); // long-s t ligature
    assertCollision(true, true, "SECRET.env", "secret.env");
    assertCollision(true, true, "CONFIG.json", "config.json");
  }

  @Test
  public void caseInsensitive_foldsCanonicalNfcNfd() throws Exception {
    assertCollision(true, true, "\u00e9_p", "e\u0301_p"); // e-acute NFC vs NFD
    assertCollision(true, true, "\u00f1_p", "n\u0303_p"); // n-tilde NFC vs NFD
  }

  @Test
  public void caseInsensitive_nestedFoldedDirectoryCollides() throws Exception {
    assertCollision(true, true, "dir_\u00df/x", "dir_ss/x");
  }

  @Test
  public void caseInsensitive_doesNotOverfoldCompatibilityForms() throws Exception {
    // APFS keeps these distinct; NFKC would wrongly fold them, so the check must not.
    assertNoCollision(true, true, "\uff21_x", "A_x"); // fullwidth A vs A
    assertNoCollision(true, true, "x\u00b2", "x2"); // superscript two
    assertNoCollision(true, true, "\u2460_w", "1_w"); // circled one
    assertNoCollision(true, true, "\u216b_v", "XII_v"); // roman numeral twelve
  }

  @Test
  public void caseSensitive_foldsNormalizationOnly() throws Exception {
    assertCollision(true, false, "\u00e9_p", "e\u0301_p"); // NFC/NFD still collide
    assertNoCollision(true, false, "ss_output.txt", "\u00df_output.txt"); // case fold off
    assertNoCollision(true, false, "SECRET.env", "secret.env");
    assertNoCollision(true, false, "fi_c", "\ufb01_c");
  }

  @Test
  public void nonDarwin_identityFoldsNothing() throws Exception {
    assertNoCollision(false, false, "ss_output.txt", "\u00df_output.txt");
    assertNoCollision(false, false, "SECRET.env", "secret.env");
    assertNoCollision(false, false, "\u00e9_p", "e\u0301_p");
  }

  @Test
  public void identicalBytePathIsAllowed() throws Exception {
    assertNoCollision(true, true, "dup.txt", "dup.txt");
  }

  @Test
  public void distinctNamesAreAllowed() throws Exception {
    assertNoCollision(true, true, "a.txt", "b.txt", "sub/c.txt", "\u00df_only.txt");
  }

  @Test
  public void create_returnsUsableCheckerOnAnyHost() throws Exception {
    HostPathCollisionChecker checker = HostPathCollisionChecker.create();
    checker.checkAndRecord(PathFragment.create("a.txt"));
    checker.checkAndRecord(PathFragment.create("b.txt"));
  }
}
