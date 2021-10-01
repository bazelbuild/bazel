package com.google.devtools.build.lib.remote.zstd;

import com.github.luben.zstd.ZstdOutputStream;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

public class ZstdCompressingInputStream extends InputStream {
    private final InputStream in;
    private final ByteArrayOutputStream baos;
    private final int size;
    private ZstdOutputStream zos;
    private byte[] buffer;
    private int count;

    public ZstdCompressingInputStream(InputStream in) throws IOException {
        this(in, 512);
    }

    ZstdCompressingInputStream(InputStream in, int size) throws IOException {
        this.in = in;
        this.baos = new ByteArrayOutputStream();
        this.zos = new ZstdOutputStream(baos);
        this.count = 0;
        this.size = size;
        this.buffer = new byte[size];
    }

    @Override
    public int read() throws IOException {
        if (zos == null && count >= baos.size()) {
            return -1;
        }
        if (count == baos.size()) {
            baos.reset();
            count = 0;
            int c = in.read(buffer, 0, buffer.length);
            if (c == -1) {
                if (zos == null) {
                    return -1;
                }
                zos.close();
                zos = null;
            } else {
                zos.write(buffer, 0, c);
                zos.flush();
            }
            baos.flush();
            if (baos.size() > buffer.length) {
                // This can happen if the buffer size is smaller than Zstd header
                // In that case, expand the buffer to fit it.
                buffer = new byte[baos.size()];
            }
            System.arraycopy(baos.toByteArray(), 0, buffer, 0, baos.size());
        }
        return buffer[count++] & 255;
    }

    @Override
    public void close() throws IOException {
        if (zos != null) {
            zos.close();
        }
        baos.close();
        in.close();
    }
}
