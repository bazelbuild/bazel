package com.github.luben.zstd;

import com.github.luben.zstd.util.Native;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;

public class ZstdDirectBufferDecompressingStream implements Closeable {

    static {
        Native.load();
    }

    /**
     * Override this method in case the byte buffer passed to the constructor might not contain the full compressed stream
     * @param toRefill current buffer
     * @return either the current buffer (but refilled and flipped if there was new content) or a new buffer.
     */
    protected ByteBuffer refill(ByteBuffer toRefill) {
        return toRefill;
    }

    private ByteBuffer source;
    private final long stream;
    private boolean finishedFrame = false;
    private boolean closed = false;
    private boolean streamEnd = false;
    private boolean finalize = true;

    private static native int recommendedDOutSize();
    private static native long createDStream();
    private static native int  freeDStream(long stream);
    private native int initDStream(long stream);
    private native long decompressStream(long stream, ByteBuffer dst, int dstOffset, int dstSize, ByteBuffer src, int srcOffset, int srcSize);

    public ZstdDirectBufferDecompressingStream(ByteBuffer source) {
        if (!source.isDirect()) {
            throw new IllegalArgumentException("Source buffer should be a direct buffer");
        }
        synchronized(this) {
            this.source = source;
            stream = createDStream();
            initDStream(stream);
        }
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

    public synchronized boolean hasRemaining() {
        return !streamEnd && (source.hasRemaining() || !finishedFrame);
    }

    public static int recommendedTargetBufferSize() {
        return (int) recommendedDOutSize();
    }

    public synchronized ZstdDirectBufferDecompressingStream setDict(byte[] dict) throws IOException {
        int size = Zstd.loadDictDecompress(stream, dict, dict.length);
        if (Zstd.isError(size)) {
            throw new IOException("Decompression error: " + Zstd.getErrorName(size));
        }
        return this;
    }

    public synchronized ZstdDirectBufferDecompressingStream setDict(ZstdDictDecompress dict) throws IOException {
        dict.acquireSharedLock();
        try {
            int size = Zstd.loadFastDictDecompress(stream, dict);
            if (Zstd.isError(size)) {
                throw new IOException("Decompression error: " + Zstd.getErrorName(size));
            }
        } finally {
            dict.releaseSharedLock();
        }
        return this;
    }

    private int consumed;
    private int produced;
    public synchronized int read(ByteBuffer target) throws IOException {
        if (!target.isDirect()) {
            throw new IllegalArgumentException("Target buffer should be a direct buffer");
        }
        if (closed) {
            throw new IOException("Stream closed");
        }
        if (streamEnd) {
            return 0;
        }

        long remaining = decompressStream(stream, target, target.position(), target.remaining(), source, source.position(), source.remaining());
        if (Zstd.isError(remaining)) {
            throw new IOException(Zstd.getErrorName(remaining));
        }

        source.position(source.position() + consumed);
        target.position(target.position() + produced);

        if (!source.hasRemaining()) {
            source = refill(source);
            if (!source.isDirect()) {
                throw new IllegalArgumentException("Source buffer should be a direct buffer");
            }
        }

        finishedFrame = remaining == 0;
        if (finishedFrame) {
            // nothing left, so at end of the stream
            streamEnd = !source.hasRemaining();
        }

        return produced;
    }


    @Override
    public synchronized void close() throws IOException {
        if (!closed) {
            try {
                freeDStream(stream);
            }
            finally {
                closed = true;
                source = null; // help GC with realizing the buffer can be released
            }
        }
    }

    @Override
    protected void finalize() throws Throwable {
        if (finalize) {
            close();
        }
    }
}
