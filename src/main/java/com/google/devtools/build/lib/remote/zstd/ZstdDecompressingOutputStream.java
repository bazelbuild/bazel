package com.google.devtools.build.lib.remote.zstd;

import com.github.luben.zstd.ZstdInputStream;
import com.google.protobuf.ByteString;
import org.apache.commons.compress.utils.CountingOutputStream;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.io.IOException;
import java.io.OutputStream;


public class ZstdDecompressingOutputStream extends CountingOutputStream {
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
    write(new byte[]{(byte) b}, 0, 1);
  }

  @Override
  public void write(byte[] b) throws IOException {
    write(b, 0, b.length);
  }

  @Override
  public void write(byte[] b, int off, int len) throws IOException {
    inner = new ByteArrayInputStream(b, off, len);
    byte[] data = ByteString.readFrom(zis).toByteArray();
    super.write(data, 0, data.length);
  }
}
