package com.google.devtools.build.lib.remote.zstd;

import com.github.luben.zstd.ZstdInputStream;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class ZstdDecompressingOutputStream extends OutputStream {
  private final ByteArrayOutputStream baos;
  private final ZstdInputStream zis;
  private final OutputStream out;
  private int lastByte;

  public ZstdDecompressingOutputStream(OutputStream out) throws IOException {
    this.out = out;
    baos = new ByteArrayOutputStream();
    zis =
        new ZstdInputStream(
            new InputStream() {
              @Override
              public int read() throws IOException {
                int value = lastByte;
                lastByte = -1;
                return value;
              }
            });
    zis.setContinuous(true);
  }

  @Override
  public void write(int b) throws IOException {
    lastByte = b;
    int c;
    while ((c = zis.read()) != -1) {
      out.write(c);
    }
  }
}
