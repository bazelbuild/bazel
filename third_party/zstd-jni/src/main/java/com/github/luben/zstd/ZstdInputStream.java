package com.github.luben.zstd;

import java.io.InputStream;
import java.io.FilterInputStream;
import java.io.IOException;
import java.lang.IndexOutOfBoundsException;

import com.github.luben.zstd.util.Native;

/**
 * InputStream filter that decompresses the data provided
 * by the underlying InputStream using Zstd compression.
 *
 * It does not support mark/reset methods
 *
 */

public class ZstdInputStream extends FilterInputStream {

    static {
        Native.load();
    }

    // Opaque pointer to Zstd context object
    private final long stream;
    private long dstPos = 0;
    private long srcPos = 0;
    private long srcSize = 0;
    private boolean needRead = true;
    private boolean finalize = true;
    private byte[] src;
    private static final int srcBuffSize = (int) recommendedDInSize();

    private boolean isContinuous = false;
    private boolean frameFinished = true;
    private boolean isClosed = false;

    /* JNI methods */
    private static native long recommendedDInSize();
    private static native long recommendedDOutSize();
    private static native long createDStream();
    private static native int  freeDStream(long stream);
    private native int  initDStream(long stream);
    private native int  decompressStream(long stream, byte[] dst, int dst_size, byte[] src, int src_size);

    // The main constuctor / legacy version dispatcher
    public ZstdInputStream(InputStream inStream) throws IOException {
        // FilterInputStream constructor
        super(inStream);

        // allocate input buffer with max frame header size
        // no need to synchronize as these a finals
        this.src = new byte[srcBuffSize];
        // memory barrier
        synchronized(this) {
            this.stream = createDStream();
            initDStream(stream);
        }
    }

    /**
     * Don't break on unfinished frames
     *
     * Use case: decompressing files that are not
     * yet finished writing and compressing
     */
    public synchronized ZstdInputStream setContinuous(boolean b) {
        isContinuous = b;
        return this;
    }

    public synchronized boolean getContinuous() {
        return this.isContinuous;
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

    public synchronized ZstdInputStream setDict(byte[] dict) throws IOException {
        int size = Zstd.loadDictDecompress(stream, dict, dict.length);
        if (Zstd.isError(size)) {
            throw new IOException("Decompression error: " + Zstd.getErrorName(size));
        }
        return this;
    }

    public synchronized ZstdInputStream setDict(ZstdDictDecompress dict) throws IOException {

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

    public synchronized int read(byte[] dst, int offset, int len) throws IOException {
        // guard agains buffer overflows
        if (offset < 0 || len > dst.length - offset) {
            throw new IndexOutOfBoundsException("Requested lenght " + len
                    + " from offset " + offset + " in buffer of size " + dst.length);
        }
        if (len == 0) {
            return 0;
        } else {
            int result = 0;
            while (result == 0) {
                result = readInternal(dst, offset, len);
            }
            return result;
        }
    }

    int readInternal(byte[] dst, int offset, int len) throws IOException {

        if (isClosed) {
            throw new IOException("Stream closed");
        }

        // guard agains buffer overflows
        if (offset < 0 || len > dst.length - offset) {
            throw new IndexOutOfBoundsException("Requested lenght " + len
                    + " from offset " + offset + " in buffer of size " + dst.length);
        }
        int dstSize = offset + len;
        dstPos = offset;
        long lastDstPos = -1;

        while (dstPos < dstSize && lastDstPos < dstPos) {
            // we will read only if data from the upstream is available OR
            // we have not yet produced any output
            if (needRead && (in.available() > 0 || dstPos == offset)) {
                srcSize = in.read(src, 0, srcBuffSize);
                srcPos = 0;
                if (srcSize < 0) {
                    srcSize = 0;
                    if (frameFinished) {
                        return -1;
                    } else if (isContinuous) {
                        return (int)(dstPos - offset);
                    } else {
                        throw new IOException("Read error or truncated source");
                    }
                }
                frameFinished = false;
            }

            lastDstPos = dstPos;
            int size = decompressStream(stream, dst, dstSize, src, (int) srcSize);

            if (Zstd.isError(size)) {
                throw new IOException("Decompression error: " + Zstd.getErrorName(size));
            }

            // we have completed a frame
            if (size == 0) {
                frameFinished = true;
                // we need to read from the upstream only if we have not consumed
                // fully the source buffer
                needRead = srcPos == srcSize;
                return (int)(dstPos - offset);
            } else {
                // size > 0, so more input is required but there is data left in
                // the decompressor buffers if we have not filled the dst buffer
                needRead = dstPos < dstSize;
            }
        }
        return (int)(dstPos - offset);
    }

    public synchronized int read() throws IOException {
        byte[] oneByte = new byte[1];
        int result = 0;
        while (result == 0) {
            result = readInternal(oneByte, 0, 1);
        }
        if (result == 1) {
            return oneByte[0] & 0xff;
        } else {
            return -1;
        }
    }

    public synchronized int available() throws IOException {
        if (isClosed) {
            throw new IOException("Stream closed");
        }
        if (!needRead) {
            return 1;
        } else {
            return in.available();
        }
    }

    /* we don't support mark/reset */
    public boolean markSupported() {
        return false;
    }

    /* we can skip forward */
    public synchronized long skip(long numBytes) throws IOException {
        if (isClosed) {
            throw new IOException("Stream closed");
        }
        if (numBytes <= 0) {
            return 0;
        }
        long toSkip = numBytes;
        int bufferLen = (int) Math.min(recommendedDOutSize(), toSkip);
        byte data[] = new byte[bufferLen];
        while (toSkip > 0) {
            int read = read(data, 0, (int) Math.min((long) bufferLen, toSkip));
            if (read < 0) {
                break;
            }
            toSkip -= read;
        }
        return numBytes - toSkip;
    }

    public synchronized void close() throws IOException {
        if (isClosed) {
            return;
        }
        freeDStream(stream);
        in.close();
        isClosed = true;
    }

    @Override
    protected void finalize() throws Throwable {
        if (finalize) {
            close();
        }
    }
}
