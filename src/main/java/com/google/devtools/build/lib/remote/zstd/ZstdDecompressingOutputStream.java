package com.google.devtools.build.lib.remote.zstd;

import com.github.luben.zstd.ZstdInputStream;
import com.google.protobuf.ByteString;

import java.io.ByteArrayInputStream;
import java.io.FilterOutputStream;
import java.io.InputStream;
import java.io.IOException;
import java.io.OutputStream;


public class ZstdDecompressingOutputStream extends FilterOutputStream {
  private ByteArrayInputStream inner;
  private final ZstdInputStream zis;

  public ZstdDecompressingOutputStream(OutputStream out) throws IOException {
    super(out);
    zis =
        new ZstdInputStream(
            new InputStream() {
              @Override
              public int read() {
                return inner.read();
              }
            });
    zis.setContinuous(true);
  }

  @Override
  public void write(int b) throws IOException {
    inner = new ByteArrayInputStream(new byte[]{(byte) b});
    ByteString.readFrom(zis).writeTo(out);
  }

  @Override
  public void write(byte[] b, int off, int len) throws IOException {
    inner = new ByteArrayInputStream(b, off, len);
    ByteString.readFrom(zis).writeTo(out);
  }
}
