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
package com.google.devtools.build.lib.compress;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Method;
import org.junit.Test;

public final class CompressionServiceImplTest {

  private final CompressionService compressionService = new CompressionServiceImpl();

  @Test
  public void zstdStreams_roundTripNonEmptyPayload() throws Exception {
    byte[] payload = "zstd round-trip payload ".repeat(128).getBytes(UTF_8);
    ByteArrayOutputStream compressed = new ByteArrayOutputStream();

    try (OutputStream out = compressionService.newZstdOutputStream(compressed)) {
      out.write(payload);
    }
    byte[] decompressed;
    try (InputStream in =
        compressionService.newZstdInputStream(new ByteArrayInputStream(compressed.toByteArray()))) {
      decompressed = in.readAllBytes();
    }

    assertThat(decompressed).isEqualTo(payload);
  }

  @Test
  public void zstdStreams_roundTripEmptyPayload() throws Exception {
    ByteArrayOutputStream compressed = new ByteArrayOutputStream();

    try (OutputStream out = compressionService.newZstdOutputStream(compressed)) {
      // Writes an empty zstd frame on close.
    }
    byte[] decompressed;
    try (InputStream in =
        compressionService.newZstdInputStream(new ByteArrayInputStream(compressed.toByteArray()))) {
      decompressed = in.readAllBytes();
    }

    assertThat(decompressed).isEmpty();
  }

  @Test
  public void compressZstd_roundTripsThroughNewZstdInputStream() throws Exception {
    byte[] payload = "one-shot zstd compression ".repeat(128).getBytes(UTF_8);

    byte[] compressed = compressionService.compressZstd(payload, /* level= */ 3);
    byte[] decompressed;
    try (InputStream in =
        compressionService.newZstdInputStream(new ByteArrayInputStream(compressed))) {
      decompressed = in.readAllBytes();
    }

    assertThat(decompressed).isEqualTo(payload);
  }

  @Test
  public void newZstdInputStream_doesNotOverrideFinalize() throws Exception {
    try (InputStream in =
        compressionService.newZstdInputStream(new ByteArrayInputStream(new byte[0]))) {
      assertThat(declaresFinalizer(in.getClass())).isFalse();
    }
  }

  @Test
  public void newZstdOutputStream_doesNotOverrideFinalize() throws Exception {
    try (OutputStream out = compressionService.newZstdOutputStream(new ByteArrayOutputStream())) {
      assertThat(declaresFinalizer(out.getClass())).isFalse();
    }
  }

  private static boolean declaresFinalizer(Class<?> clazz) {
    for (Class<?> current = clazz;
        current != null && current != Object.class;
        current = current.getSuperclass()) {
      for (Method method : current.getDeclaredMethods()) {
        if (method.getName().equals("finalize") && method.getParameterCount() == 0) {
          return true;
        }
      }
    }
    return false;
  }
}
