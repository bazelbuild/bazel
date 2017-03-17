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

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

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
    assertEquals(
        "d41d8cd98f00b204e9800998ecf8427e", Digest.fromContent("".getBytes(UTF_8)).toStringUtf8());
    assertEquals("d41d8cd98f00b204e9800998ecf8427e",
        Digest.fromBuffer(ByteBuffer.wrap(new byte[]{})).toStringUtf8());

    // Whitespace counts.
    assertEquals(
        "7215ee9c7d9dc229d2921a40e899ec5f", Digest.fromContent(" ".getBytes(UTF_8)).toStringUtf8());
    assertEquals(
        "23b58def11b45727d3351702515f86af",
        Digest.fromContent("  ".getBytes(UTF_8)).toStringUtf8());

    assertEquals(
        "8b1a9953c4611296a827abf8c47804d7",
        Digest.fromContent("Hello".getBytes(UTF_8)).toStringUtf8());

    assertEquals(UGLY_DIGEST, Digest.fromContent(UGLY.getBytes(UTF_8)).toStringUtf8());

    // ByteBuffer digest not idempotent because ByteBuffer manages a "position" internally.
    ByteBuffer buffer = ByteBuffer.wrap(UGLY.getBytes(UTF_8));
    assertEquals(UGLY_DIGEST, Digest.fromBuffer(buffer).toStringUtf8());
    assertEquals("d41d8cd98f00b204e9800998ecf8427e",
        Digest.fromBuffer(buffer).toStringUtf8());
    buffer.rewind();
    assertEquals("e5df5a39f2b8cb71b24e1d8038f93131",
                 Digest.fromBuffer(buffer).toStringUtf8());
  }

  @Test
  public void testEmpty() {
    assertFalse(Digest.isEmpty(ByteString.EMPTY));
    assertTrue(Digest.isEmpty(ByteString.copyFromUtf8("d41d8cd98f00b204e9800998ecf8427e")));
    assertTrue(Digest.isEmpty(Digest.EMPTY_DIGEST));
    assertFalse(Digest.isEmpty(ByteString.copyFromUtf8("xyz")));
    assertFalse(Digest.isEmpty(Digest.fromContent(" ".getBytes(UTF_8))));
    assertEquals("d41d8cd98f00b204e9800998ecf8427e", Digest.EMPTY_DIGEST.toStringUtf8());
  }

  @Test
  public void testIsDigest() {
    assertFalse(Digest.isDigest(null));
    assertFalse(Digest.isDigest(ByteString.EMPTY));
    assertFalse(Digest.isDigest(ByteString.copyFromUtf8("a")));
    assertFalse(Digest.isDigest(ByteString.copyFromUtf8("xyz")));
    assertTrue(Digest.isDigest(ByteString.copyFromUtf8("8b1a9953c4611296a827abf8c47804d7")));
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
              public String getExecPathString() {
                throw new UnsupportedOperationException();
              }

              @Override
              public PathFragment getExecPath() {
                throw new UnsupportedOperationException();
              }
            });
    assertEquals(UGLY_DIGEST, result.first.toStringUtf8());
    assertEquals(UGLY.length(), result.second.longValue());
  }
}

