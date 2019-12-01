package com.github.luben.zstd;

import java.io.OutputStream;
import java.io.FilterOutputStream;
import java.io.IOException;

import com.github.luben.zstd.util.Native;

/**
 * OutputStream filter that compresses the data using Zstd compression
 */
public class ZstdOutputStream extends FilterOutputStream {

    static {
        Native.load();
    }

    /* Opaque pointer to Zstd context object */
    private final long stream;
    private long srcPos = 0;
    private long dstPos = 0;
    private final byte[] dst;
    private boolean isClosed = false;
    private boolean finalize = true;
    private static final int dstSize = (int) recommendedCOutSize();
    private boolean closeFrameOnFlush;
    private boolean frameClosed = true;

    /* JNI methods */
    private static native long recommendedCOutSize();
    private static native long createCStream();
    private static native int  freeCStream(long ctx);
    private native int resetCStream(long ctx);
    private native int compressStream(long ctx, byte[] dst, int dst_size, byte[] src, int src_size);
    private native int flushStream(long ctx, byte[] dst, int dst_size);
    private native int endStream(long ctx, byte[] dst, int dst_size);


    /**
     *  @deprecated
     *  Use ZstdOutputStream() or ZstdOutputStream(level) and set the other params with the setters
     **/
    public ZstdOutputStream(OutputStream outStream, int level, boolean closeFrameOnFlush, boolean useChecksums) throws IOException {
        this(outStream);
        this.closeFrameOnFlush = closeFrameOnFlush;
        Zstd.setCompressionLevel(this.stream, level);
        Zstd.setCompressionChecksums(this.stream, useChecksums);
    }

    /**
     *  @deprecated
     *  Use ZstdOutputStream() or ZstdOutputStream(level) and set the other params with the setters
     **/
    public ZstdOutputStream(OutputStream outStream, int level, boolean closeFrameOnFlush) throws IOException {
        this(outStream);
        this.closeFrameOnFlush = closeFrameOnFlush;
        Zstd.setCompressionLevel(this.stream, level);
    }

    /* The constuctor */
    public ZstdOutputStream(OutputStream outStream, int level) throws IOException {
        this(outStream);
        this.closeFrameOnFlush = false;
        Zstd.setCompressionLevel(this.stream, level);
    }

    /* The constuctor */
    public ZstdOutputStream(OutputStream outStream) throws IOException {
        super(outStream);
        // create compression context
        this.stream = createCStream();
        this.closeFrameOnFlush = false;
        this.dst = new byte[(int) dstSize];
    }

    public synchronized ZstdOutputStream setChecksum(boolean useChecksums) throws IOException {
        if (!frameClosed) {
            throw new IOException("Change of parameter on initialized stream");
        }
        int size = Zstd.setCompressionChecksums(stream, useChecksums);
        if (Zstd.isError(size)) {
            throw new IOException("Compression param: " + Zstd.getErrorName(size));
        }
        return this;
    }

    public synchronized ZstdOutputStream setLevel(int level) throws IOException {
        if (!frameClosed) {
            throw new IOException("Change of parameter on initialized stream");
        }
        int size = Zstd.setCompressionLevel(stream, level);
        if (Zstd.isError(size)) {
            throw new IOException("Compression param: " + Zstd.getErrorName(size));
        }
        return this;
    }

    public synchronized ZstdOutputStream setWorkers(int n) throws IOException {
        if (!frameClosed) {
            throw new IOException("Change of parameter on initialized stream");
        }
        int size = Zstd.setCompressionWorkers(stream, n);
        if (Zstd.isError(size)) {
            throw new IOException("Compression param: " + Zstd.getErrorName(size));
        }
        return this;
    }

    public synchronized ZstdOutputStream setCloseFrameOnFlush(boolean closeOnFlush) throws IOException {
        if (!frameClosed) {
            throw new IOException("Change of parameter on initialized stream");
        }
        this.closeFrameOnFlush = closeOnFlush;
        return this;
    }

    public synchronized ZstdOutputStream setDict(byte[] dict) throws IOException {
        if (!frameClosed) {
            throw new IOException("Change of parameter on initialized stream");
        }
        int size = Zstd.loadDictCompress(stream, dict, dict.length);
        if (Zstd.isError(size)) {
            throw new IOException("Compression param: " + Zstd.getErrorName(size));
        }
        return this;
    }

    public synchronized ZstdOutputStream setDict(ZstdDictCompress dict) throws IOException {
        if (!frameClosed) {
            throw new IOException("Change of parameter on initialized stream");
        }
        int size = Zstd.loadFastDictCompress(stream, dict);
        if (Zstd.isError(size)) {
            throw new IOException("Compression param: " + Zstd.getErrorName(size));
        }
        return this;
    }

    /**
     * Enable or disable class finalizers
     *
     * If finalizers are disabled the responsibility fir calling the `close` method is on the consumer.
     *
     * @param finalize, default `true` - finalizers are enabled
     */
    public void setFinalize(boolean finalize) {
        this.finalize = finalize;
    }


    public synchronized void write(byte[] src, int offset, int len) throws IOException {
        if (isClosed) {
            throw new IOException("Stream closed");
        }
        if (frameClosed) {
            int size = resetCStream(this.stream);
            if (Zstd.isError(size)) {
                throw new IOException("Compression error: cannot create header: " + Zstd.getErrorName(size));
            }
            frameClosed = false;
        }
        int srcSize = offset + len;
        srcPos = offset;
        while (srcPos < srcSize) {
            int size = compressStream(stream, dst, dstSize, src, srcSize);
            if (Zstd.isError(size)) {
                throw new IOException("Compression error: " + Zstd.getErrorName(size));
            }
            if (dstPos > 0) {
                out.write(dst, 0, (int) dstPos);
            }
        }
    }

    public void write(int i) throws IOException {
        byte[] oneByte = new byte[1];
        oneByte[0] = (byte) i;
        write(oneByte, 0, 1);
    }

    /**
     * Flushes the output
     */
    public synchronized void flush() throws IOException {
        if (isClosed) {
            throw new IOException("Stream closed");
        }
        if (!frameClosed) {
            if (closeFrameOnFlush) {
                // compress the remaining output and close the frame
                int size;
                do {
                    size = endStream(stream, dst, dstSize);
                    if (Zstd.isError(size)) {
                        throw new IOException("Compression error: " + Zstd.getErrorName(size));
                    }
                    out.write(dst, 0, (int) dstPos);
                } while (size > 0);
                frameClosed = true;
            } else {
                // compress the remaining input
                int size;
                do {
                    size = flushStream(stream, dst, dstSize);
                    if (Zstd.isError(size)) {
                        throw new IOException("Compression error: " + Zstd.getErrorName(size));
                    }
                    out.write(dst, 0, (int) dstPos);
                } while (size > 0);
            }
            out.flush();
        }
    }

    public synchronized void close() throws IOException {
        if (isClosed) {
            return;
        }
        if (!frameClosed) {
            // compress the remaining input and close the frame
            int size;
            do {
                size = endStream(stream, dst, dstSize);
                if (Zstd.isError(size)) {
                    throw new IOException("Compression error: " + Zstd.getErrorName(size));
                }
                out.write(dst, 0, (int) dstPos);
            } while (size > 0);
        }
        // release the resources
        freeCStream(stream);
        out.close();
        isClosed = true;
    }

    @Override
    protected void finalize() throws Throwable {
        if (finalize) {
            close();
        }
    }
}
