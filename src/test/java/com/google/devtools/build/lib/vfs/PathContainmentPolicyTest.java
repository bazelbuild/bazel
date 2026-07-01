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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.vfs.PathFragment.create;

import com.google.devtools.build.lib.util.OS;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PathContainmentPolicy}. */
@RunWith(JUnit4.class)
public final class PathContainmentPolicyTest {

  private static final PathContainmentPolicy DEFAULT = PathContainmentPolicy.DEFAULT;
  private static final PathContainmentPolicy DARWIN = PathContainmentPolicy.DARWIN;

  @Test
  public void defaultPolicy_byteLevelPrefix() {
    assertThat(DEFAULT.isContained(create("/a/b/c"), create("/a/b"))).isTrue();
    assertThat(DEFAULT.isContained(create("/a/b"), create("/a/b"))).isTrue();
    assertThat(DEFAULT.isContained(create("/a/bc"), create("/a/b"))).isFalse();
    assertThat(DEFAULT.isContained(create("/x"), create("/a"))).isFalse();
  }

  @Test
  public void defaultPolicy_doesNotFoldUnicode() {
    assertThat(DEFAULT.isContained(create("/p/file_ss.txt"), create("/p/file_ß.txt")))
        .isFalse();
    assertThat(DEFAULT.isContained(create("/p/SECRET"), create("/p/secret"))).isFalse();
  }

  @Test
  public void darwin_caseInsensitiveContainment() {
    assertThat(DARWIN.isContained(create("/p/secret.txt"), create("/p/SECRET.txt"))).isTrue();
    assertThat(DARWIN.isContained(create("/p/SECRET.txt"), create("/p/secret.txt"))).isTrue();
    assertThat(DARWIN.isContained(create("/p/CONFIG.json"), create("/p/config.json"))).isTrue();
  }

  @Test
  public void darwin_ligatureFolding() {
    assertThat(DARWIN.isContained(create("/p/file_ss.txt"), create("/p/file_ß.txt")))
        .isTrue();
    assertThat(DARWIN.isContained(create("/p/file_fi.txt"), create("/p/file_ﬁ.txt")))
        .isTrue();
    assertThat(DARWIN.isContained(create("/p/file_fl.txt"), create("/p/file_ﬂ.txt")))
        .isTrue();
    assertThat(DARWIN.isContained(create("/p/file_ff.txt"), create("/p/file_ﬀ.txt")))
        .isTrue();
    assertThat(DARWIN.isContained(create("/p/file_ffi.txt"), create("/p/file_ﬃ.txt")))
        .isTrue();
    assertThat(DARWIN.isContained(create("/p/file_ffl.txt"), create("/p/file_ﬄ.txt")))
        .isTrue();
    assertThat(DARWIN.isContained(create("/p/file_st.txt"), create("/p/file_ﬅ.txt")))
        .isTrue();
  }

  @Test
  public void darwin_nfcEqualsNfd() {
    String nfc = "/p/file_é.txt"; // é precomposed
    String nfd = "/p/file_é.txt"; // e + combining acute
    assertThat(DARWIN.isContained(create(nfd), create(nfc))).isTrue();
    assertThat(DARWIN.isContained(create(nfc), create(nfd))).isTrue();

    String nfcN = "/p/file_ñ.txt"; // ñ precomposed
    String nfdN = "/p/file_ñ.txt"; // n + combining tilde
    assertThat(DARWIN.isContained(create(nfdN), create(nfcN))).isTrue();
  }

  @Test
  public void darwin_segmentBoundaryPreserved() {
    assertThat(DARWIN.isContained(create("/root/foobar"), create("/root/foo"))).isFalse();
    assertThat(DARWIN.isContained(create("/root/secrethouse"), create("/root/SECRET"))).isFalse();
    assertThat(DARWIN.isContained(create("/root/file_ssss"), create("/root/file_ß")))
        .isFalse();
  }

  @Test
  public void darwin_subdirectoryUnderFoldedPrefix() {
    assertThat(DARWIN.isContained(create("/root/dir_ss/inner"), create("/root/dir_ß")))
        .isTrue();
    assertThat(DARWIN.isContained(create("/root/SECRET/file"), create("/root/secret"))).isTrue();
  }

  @Test
  public void darwin_absoluteVsRelativeMismatch() {
    assertThat(DARWIN.isContained(create("rel/path"), create("/root"))).isFalse();
    assertThat(DARWIN.isContained(create("/root"), create("rel/path"))).isFalse();
  }

  @Test
  public void darwin_emptyAndRoot() {
    assertThat(DARWIN.isContained(create("/anything/else"), create("/"))).isTrue();
    assertThat(DARWIN.isContained(create("/"), create("/"))).isTrue();
    assertThat(DARWIN.isContained(create(""), create(""))).isTrue();
  }

  @Test
  public void darwin_fastPathMatchesByteLevel() {
    // When byte-level startsWith already passes, DARWIN must agree with DEFAULT.
    PathFragment child = create("/a/b/c/d");
    PathFragment parent = create("/a/b");
    assertThat(DARWIN.isContained(child, parent)).isEqualTo(DEFAULT.isContained(child, parent));
  }

  @Test
  public void canonicalizeForDarwin_isIdempotent() {
    String once = PathContainmentPolicy.canonicalizeForDarwin("/Foo/ß/ﬁ");
    String twice = PathContainmentPolicy.canonicalizeForDarwin(once);
    assertThat(twice).isEqualTo(once);
  }

  @Test
  public void hostPolicy_matchesCurrentOs() {
    if (OS.getCurrent() != OS.DARWIN) {
      assertThat(PathContainmentPolicy.HOST_POLICY).isSameInstanceAs(DEFAULT);
      return;
    }
    // On macOS, the policy depends on the boot volume's case sensitivity. Either DARWIN
    // (case-insensitive APFS, the default) or DEFAULT (case-sensitive APFS) is valid.
    assertThat(PathContainmentPolicy.HOST_POLICY).isAnyOf(DARWIN, DEFAULT);
  }

  @Test
  public void forOs_returnsDefaultForNonDarwin() {
    assertThat(PathContainmentPolicy.forOs(OS.LINUX)).isSameInstanceAs(DEFAULT);
    assertThat(PathContainmentPolicy.forOs(OS.WINDOWS)).isSameInstanceAs(DEFAULT);
    assertThat(PathContainmentPolicy.forOs(OS.DARWIN)).isSameInstanceAs(DARWIN);
  }

  @Test
  public void probeCaseSensitive_completesWithoutThrowing() {
    // Either answer is acceptable, what matters is that the probe never fails fatally.
    boolean unused = PathContainmentPolicy.probeCaseSensitive();
  }
}
