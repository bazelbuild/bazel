package com.google.devtools.build.lib.remote.zstd;

import com.github.luben.zstd.Zstd;
import com.google.common.io.ByteStreams;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Random;

import static com.google.common.truth.Truth.assertThat;

@RunWith(JUnit4.class)
public class ZstdCompressingInputStreamTest {
  @Test
  public void compressionWorks() throws IOException {
    Random rand = new Random();
    byte[] data = new byte[50];
    rand.nextBytes(data);

    ByteArrayInputStream bais = new ByteArrayInputStream(data);
    ZstdCompressingInputStream zdis = new ZstdCompressingInputStream(bais);

    ByteArrayOutputStream compressed = new ByteArrayOutputStream();
    ByteStreams.copy(zdis, compressed);

    assertThat(Zstd.decompress(compressed.toByteArray(), data.length)).isEqualTo(data);
  }

  @Test
  public void streamCanBeCompressedWithMinimumBufferSize() throws IOException {
    Random rand = new Random();
    byte[] data = new byte[50];
    rand.nextBytes(data);

    ByteArrayInputStream bais = new ByteArrayInputStream(data);
    ZstdCompressingInputStream zdis =
        new ZstdCompressingInputStream(bais, ZstdCompressingInputStream.MIN_BUFFER_SIZE);

    ByteArrayOutputStream compressed = new ByteArrayOutputStream();
    ByteStreams.copy(zdis, compressed);

    assertThat(Zstd.decompress(compressed.toByteArray(), data.length)).isEqualTo(data);
  }
}
