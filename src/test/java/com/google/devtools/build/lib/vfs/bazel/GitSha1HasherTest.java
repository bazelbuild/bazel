// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.vfs.bazel;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.hash.Hasher;
import com.google.devtools.build.lib.vfs.GitSha1HashFunction;
import com.google.devtools.build.lib.vfs.GitSha1MessageDigest;
import java.nio.charset.StandardCharsets;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link GitSha1MessageDigest}. */
@RunWith(JUnit4.class)
public class GitSha1HasherTest {
  @Test
  public void emptyHash() {
    Hasher h = GitSha1HashFunction.INSTANCE.newHasher();

    byte[] data = new byte[0];
    h.putBytes(data);

    assertThat(h.hash().toString()).isEqualTo("e69de29bb2d1d6434b8b29ae775ad8c2e48c5391");
  }

  @Test
  public void helloWorld() {
    Hasher h = GitSha1HashFunction.INSTANCE.newHasher();

    byte[] data = "hello world".getBytes(StandardCharsets.US_ASCII);
    h.putBytes(data);

    assertThat(h.hash().toString()).isEqualTo("95d09f2b10159347eece71399a7e2e907ea3df4f");
  }
}
