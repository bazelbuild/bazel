// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for Digest.
 *
 */
@RunWith(JUnit4.class)
@TestSpec(size = Suite.SMALL_TESTS)
public class DigestTest {

  private static final String UGLY = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLM"
      + "NOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";
  private static final String UGLY_DIGEST = "e5df5a39f2b8cb71b24e1d8038f93131";

  @Test
  public void testFromString() {
    assertThat(Digest.fromContent("".getBytes(UTF_8)).toStringUtf8())
        .isEqualTo("d41d8cd98f00b204e9800998ecf8427e");
    assertThat(Digest.fromBuffer(ByteBuffer.wrap(new byte[] {})).toStringUtf8())
        .isEqualTo("d41d8cd98f00b204e9800998ecf8427e");

    // Whitespace counts.
    assertThat(Digest.fromContent(" ".getBytes(UTF_8)).toStringUtf8())
        .isEqualTo("7215ee9c7d9dc229d2921a40e899ec5f");
    assertThat(Digest.fromContent("  ".getBytes(UTF_8)).toStringUtf8())
        .isEqualTo("23b58def11b45727d3351702515f86af");

    assertThat(Digest.fromContent("Hello".getBytes(UTF_8)).toStringUtf8())
        .isEqualTo("8b1a9953c4611296a827abf8c47804d7");

    assertThat(Digest.fromContent(UGLY.getBytes(UTF_8)).toStringUtf8()).isEqualTo(UGLY_DIGEST);

    // ByteBuffer digest not idempotent because ByteBuffer manages a "position" internally.
    ByteBuffer buffer = ByteBuffer.wrap(UGLY.getBytes(UTF_8));
    assertThat(Digest.fromBuffer(buffer).toStringUtf8()).isEqualTo(UGLY_DIGEST);
    assertThat(Digest.fromBuffer(buffer).toStringUtf8())
        .isEqualTo("d41d8cd98f00b204e9800998ecf8427e");
    buffer.rewind();
    assertThat(Digest.fromBuffer(buffer).toStringUtf8())
        .isEqualTo("e5df5a39f2b8cb71b24e1d8038f93131");
  }

  @Test
  public void testEmpty() {
    assertThat(Digest.isEmpty(ByteString.EMPTY)).isFalse();
    assertThat(Digest.isEmpty(ByteString.copyFromUtf8("d41d8cd98f00b204e9800998ecf8427e")))
        .isTrue();
    assertThat(Digest.isEmpty(Digest.EMPTY_DIGEST)).isTrue();
    assertThat(Digest.isEmpty(ByteString.copyFromUtf8("xyz"))).isFalse();
    assertThat(Digest.isEmpty(Digest.fromContent(" ".getBytes(UTF_8)))).isFalse();
    assertThat(Digest.EMPTY_DIGEST.toStringUtf8()).isEqualTo("d41d8cd98f00b204e9800998ecf8427e");
  }

  @Test
  public void testIsDigest() {
    assertThat(Digest.isDigest(null)).isFalse();
    assertThat(Digest.isDigest(ByteString.EMPTY)).isFalse();
    assertThat(Digest.isDigest(ByteString.copyFromUtf8("a"))).isFalse();
    assertThat(Digest.isDigest(ByteString.copyFromUtf8("xyz"))).isFalse();
    assertThat(Digest.isDigest(ByteString.copyFromUtf8("8b1a9953c4611296a827abf8c47804d7")))
        .isTrue();
  }

  @Test
  public void testFromVirtualInput() throws Exception{
    Pair<ByteString, Long> result =
        Digest.fromVirtualActionInput(
            new VirtualActionInput() {
              @Override
              public void writeTo(OutputStream out) throws IOException {
                out.write(UGLY.getBytes(UTF_8));
              }

              @Override
              public ByteString getBytes() throws IOException {
                ByteString.Output out = ByteString.newOutput();
                writeTo(out);
                return out.toByteString();
              }

              @Override
              public String getExecPathString() {
                throw new UnsupportedOperationException();
              }

              @Override
              public PathFragment getExecPath() {
                throw new UnsupportedOperationException();
              }
            });
    assertThat(result.first.toStringUtf8()).isEqualTo(UGLY_DIGEST);
    assertThat(result.second.longValue()).isEqualTo(UGLY.length());
  }
}

