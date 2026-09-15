// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.downloader;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.io.CharStreams;
import com.google.devtools.build.lib.bazel.repository.cache.DownloadCache.KeyType;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link HashInputStream}. */
@RunWith(JUnit4.class)
@SuppressWarnings("resource")
public class HashInputStreamTest {

  @Rule
  public final ExpectedException thrown = ExpectedException.none();

  @Test
  public void validChecksum_readsOk() throws Exception {
    try (InputStreamReader reader =
        new InputStreamReader(
            new HashInputStream(
                new ByteArrayInputStream("hello".getBytes(UTF_8)),
                Checksum.fromString(KeyType.SHA1, "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d")),
            UTF_8)) {
      assertThat(CharStreams.toString(reader)).isEqualTo("hello");
    }
  }

  @Test
  public void badChecksum_throwsIOException() throws Exception {
    thrown.expect(IOException.class);
    thrown.expectMessage("Checksum");
    try (InputStreamReader reader =
        new InputStreamReader(
            new HashInputStream(
                new ByteArrayInputStream("hello".getBytes(UTF_8)),
                Checksum.fromString(KeyType.SHA1, "0000000000000000000000000000000000000000")),
            UTF_8)) {
      assertThat(CharStreams.toString(reader))
          .isNull(); // Only here to make @CheckReturnValue happy.
    }
  }
}
