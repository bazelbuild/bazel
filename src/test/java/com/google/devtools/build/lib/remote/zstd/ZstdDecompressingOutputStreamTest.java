package com.google.devtools.build.lib.remote.zstd;

import static com.google.common.truth.Truth.assertThat;

import com.github.luben.zstd.Zstd;
import com.github.luben.zstd.ZstdOutputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Random;

@RunWith(JUnit4.class)
public class ZstdDecompressingOutputStreamTest {
  @Test
  public void decompressionWorks() throws IOException {
    Random rand = new Random();
    byte[] data = new byte[50];
    rand.nextBytes(data);
    byte[] compressed = Zstd.compress(data);

    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    ZstdDecompressingOutputStream zdos = new ZstdDecompressingOutputStream(baos);
    zdos.write(compressed);
    zdos.flush();

    assertThat(baos.toByteArray()).isEqualTo(data);
  }

  @Test
  public void streamCanBeDecompressedOneByteAtATime() throws IOException {
    Random rand = new Random();
    byte[] data = new byte[50];
    rand.nextBytes(data);
    byte[] compressed = Zstd.compress(data);

    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    ZstdDecompressingOutputStream zdos = new ZstdDecompressingOutputStream(baos);
    for (byte b : compressed) {
      zdos.write(b);
    }
    zdos.flush();

    assertThat(baos.toByteArray()).isEqualTo(data);
  }

  @Test
  public void bytesWrittenMatchesDecompressedBytes() throws IOException {
    byte[] data = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".getBytes();

    ByteArrayOutputStream compressed = new ByteArrayOutputStream();
    ZstdOutputStream zos = new ZstdOutputStream(compressed);
    zos.setCloseFrameOnFlush(true);
    for (int i = 0; i < data.length; i++) {
      zos.write(data[i]);
      if (i % 5 == 0) {
        // Create multiple frames of 5 bytes each.
        zos.flush();
      }
    }
    zos.close();

    ByteArrayOutputStream decompressed = new ByteArrayOutputStream();
    ZstdDecompressingOutputStream zdos = new ZstdDecompressingOutputStream(decompressed);
    for (byte b : compressed.toByteArray()) {
      zdos.write(b);
      zdos.flush();
      assertThat(zdos.getBytesWritten()).isEqualTo(decompressed.toByteArray().length);
    }
    assertThat(decompressed.toByteArray()).isEqualTo(data);
  }
}
