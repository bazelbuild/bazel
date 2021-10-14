package com.google.devtools.build.lib.remote.zstd;

import com.github.luben.zstd.ZstdOutputStream;

import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;

public class ZstdCompressingInputStream extends FilterInputStream {
    private final PipedInputStream pis;
    private ZstdOutputStream zos;
    private final int size;

    public ZstdCompressingInputStream(InputStream in) throws IOException {
        this(in, 512);
    }

    ZstdCompressingInputStream(InputStream in, int size) throws IOException {
        super(in);
        this.size = size;
        this.pis = new PipedInputStream(size);
        this.zos = new ZstdOutputStream(new PipedOutputStream(pis));
    }

    private void reFill() throws IOException {
      byte[] buf = new byte[size];
      int len = super.read(buf, 0 , Math.max(0, size - pis.available()));
      if (len == -1) {
        zos.close();
        zos = null;
      } else {
        zos.write(buf, 0, len);
        zos.flush();
      }
    }

    @Override
    public int read() throws IOException {
        if (pis.available() == 0) {
            if (zos == null) {
                return -1;
            }
            reFill();
        }
        return pis.read();
    }

    @Override
    public int read(byte[] b) throws IOException {
        return read(b, 0, b.length);
    }

    @Override
    public int read(byte[] b, int off, int len) throws IOException {
        int count = 0;
        int n = len > 0 ? -1 : 0;
        while (count < len && (pis.available() > 0 || zos != null)) {
            if (pis.available() == 0) {
                reFill();
            }
            n = pis.read(b, count + off, len - count);
            count += Math.max(0, n);
        }
        return count > 0 ? count : n;
    }

    @Override
    public void close() throws IOException {
        if (zos != null) {
            zos.close();
        }
        in.close();
    }
}
