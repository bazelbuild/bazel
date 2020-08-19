package com.github.luben.zstd;

import com.github.luben.zstd.util.Native;
import com.github.luben.zstd.ZstdDictCompress;

import java.nio.ByteBuffer;
import java.util.Arrays;

public class ZstdCompressCtx extends AutoCloseBase {

    static {
        Native.load();
    }

    private long nativePtr = 0;

    private ZstdDictCompress compression_dict = null;

    private native void init();

    private native void free();

    /**
     * Create a context for faster compress operations
     * One such context is required for each thread - put this in a ThreadLocal.
     */
    public ZstdCompressCtx() {
        init();
        if (0 == nativePtr) {
            throw new IllegalStateException("ZSTD_createCompressCtx failed");
        }
        storeFence();
    }

    void  doClose() {
        if (nativePtr != 0) {
            free();
            nativePtr = 0;
        }
    }

    /**
     * Set compression level
     * @param level compression level, default: 3
     */
    public ZstdCompressCtx setLevel(int level) {
        if (nativePtr == 0) {
            throw new IllegalStateException("Compression context is closed");
        }
        acquireSharedLock();
        setLevel0(level);
        releaseSharedLock();
        return this;
    }

    private native void setLevel0(int level);

    /**
     * Enable or disable compression checksums
     * @param checksumFlag A 32-bits checksum of content is written at end of frame, default: false
     */
    public ZstdCompressCtx setChecksum(boolean checksumFlag) {
        if (nativePtr == 0) {
            throw new IllegalStateException("Compression context is closed");
        }
        acquireSharedLock();
        setChecksum0(checksumFlag);
        releaseSharedLock();
        return this;
    }
    private native void setChecksum0(boolean checksumFlag);

    /**
     * Enable or disable content size
     * @param contentSizeFlag Content size will be written into frame header _whenever known_, default: true
     */
    public ZstdCompressCtx setContentSize(boolean contentSizeFlag) {
        if (nativePtr == 0) {
            throw new IllegalStateException("Compression context is closed");
        }
        acquireSharedLock();
        setContentSize0(contentSizeFlag);
        releaseSharedLock();
        return this;
    }
    private native void setContentSize0(boolean contentSizeFlag);

    /**
     * Enable or disable dictID
     * @param dictIDFlag When applicable, dictionary's ID is written into frame header, default: true
     */
    public ZstdCompressCtx setDictID(boolean dictIDFlag) {
        if (nativePtr == 0) {
            throw new IllegalStateException("Compression context is closed");
        }
        acquireSharedLock();
        setDictID0(dictIDFlag);
        releaseSharedLock();
        return this;
    }
    private native void setDictID0(boolean dictIDFlag);

    /**
     * Load compression dictionary to be used for subsequently compressed frames.
     *
     * @param dict the dictionary or `null` to remove loaded dictionary
     */
    public ZstdCompressCtx loadDict(ZstdDictCompress dict) {
        if (nativePtr == 0) {
            throw new IllegalStateException("Compression context is closed");
        }

        acquireSharedLock();
        dict.acquireSharedLock();
        try {
            long result = loadCDictFast0(dict);
            if (Zstd.isError(result)) {
                throw new ZstdException(result);
            }
            // keep a reference to the dictionary so it's not garbage collected
            compression_dict = dict;
        } finally {
            dict.releaseSharedLock();
            releaseSharedLock();
        }
        return this;
    }
    private native long loadCDictFast0(ZstdDictCompress dict);

    /**
     * Load compression dictionary to be used for subsequently compressed frames.
     *
     * @param dict the dictionary or `null` to remove loaded dictionary
     */
    public ZstdCompressCtx loadDict(byte[] dict) {
        if (nativePtr == 0) {
            throw new IllegalStateException("Compression context is closed");
        }
        acquireSharedLock();
        try {
            long result = loadCDict0(dict);
            if (Zstd.isError(result)) {
                throw new ZstdException(result);
            }
            compression_dict = null;
        } finally {
            releaseSharedLock();
        }
        return this;
    }
    private native long loadCDict0(byte[] dict);

    /**
     * Compresses buffer 'srcBuff' into buffer 'dstBuff' reusing this ZstdCompressCtx.
     *
     * Destination buffer should be sized to handle worst cases situations (input
     * data not compressible). Worst case size evaluation is provided by function
     * ZSTD_compressBound(). This is a low-level function that does not take into
     * account or affect the `limit` or `position` of source or destination buffers.
     *
     * @param dstBuff the destination buffer - must be direct
     * @param dstOffset the start offset of 'dstBuff'
     * @param dstSize the size of 'dstBuff'
     * @param srcBuff the source buffer - must be direct
     * @param srcOffset the start offset of 'srcBuff'
     * @param srcSize the length of 'srcBuff'
     * @return  the number of bytes written into buffer 'dstBuff'.
     */
    public int compressDirectByteBuffer(ByteBuffer dstBuff, int dstOffset, int dstSize, ByteBuffer srcBuff, int srcOffset, int srcSize) {
        if (nativePtr == 0) {
            throw new IllegalStateException("Compression context is closed");
        }
        if (!srcBuff.isDirect()) {
            throw new IllegalArgumentException("srcBuff must be a direct buffer");
        }
        if (!dstBuff.isDirect()) {
            throw new IllegalArgumentException("dstBuff must be a direct buffer");
        }

        acquireSharedLock();

        try {
            long size = compressDirectByteBuffer0(dstBuff, dstOffset, dstSize, srcBuff, srcOffset, srcSize);
            if (Zstd.isError(size)) {
                throw new ZstdException(size);
            }
            if (size > Integer.MAX_VALUE) {
                throw new ZstdException(Zstd.errGeneric(), "Output size is greater than MAX_INT");
            }
            return (int) size;
        } finally {
            releaseSharedLock();
        }
    }

    private native long compressDirectByteBuffer0(ByteBuffer dst, int dstOffset, int dstSize, ByteBuffer src, int srcOffset, int srcSize);

    /**
     * Compresses byte array 'srcBuff' into byte array 'dstBuff' reusing this ZstdCompressCtx.
     *
     * Destination buffer should be sized to handle worst cases situations (input
     * data not compressible). Worst case size evaluation is provided by function
     * ZSTD_compressBound().
     *
     * @param dstBuff the destination buffer (byte array)
     * @param dstOffset the start offset of 'dstBuff'
     * @param dstSize the size of 'dstBuff'
     * @param srcBuff the source buffer (byte array)
     * @param srcOffset the start offset of 'srcBuff'
     * @param srcSize the length of 'srcBuff'
     * @return  the number of bytes written into buffer 'dstBuff'.
     */
    public int compressByteArray(byte[] dstBuff, int dstOffset, int dstSize, byte[] srcBuff, int srcOffset, int srcSize) {
        if (nativePtr == 0) {
            throw new IllegalStateException("Compression context is closed");
        }

        acquireSharedLock();

        try {
            long size = compressByteArray0(dstBuff, dstOffset, dstSize, srcBuff, srcOffset, srcSize);
            if (Zstd.isError(size)) {
                throw new ZstdException(size);
            }
            if (size > Integer.MAX_VALUE) {
                throw new ZstdException(Zstd.errGeneric(), "Output size is greater than MAX_INT");
            }
            return (int) size;
        } finally {
            releaseSharedLock();
        }
    }

    private native long compressByteArray0(byte[] dst, int dstOffset, int dstSize, byte[] src, int srcOffset, int srcSize);

    /** Convenience methods */

    /**
     * Compresses the data in buffer 'srcBuf'
     *
     * @param dstBuf the destination buffer - must be direct. It is assumed that the `position()` of this buffer marks the offset
     *               at which the compressed data are to be written, and that the `limit()` of this buffer is the maximum
     *               compressed data size to allow.
     *               <p>
     *               When this method returns successfully, its `position()` will be set to its current `position()` plus the
     *               compressed size of the data.
     *               </p>
     * @param srcBuf the source buffer - must be direct. It is assumed that the `position()` of this buffer marks the beginning of the
     *               uncompressed data to be compressed, and that the `limit()` of this buffer marks its end.
     *               <p>
     *               When this method returns successfully, its `position()` will be set to the initial `limit()`.
     *               </p>
     * @return the size of the compressed data
     */
    public int compress(ByteBuffer dstBuf, ByteBuffer srcBuf) {

        int size = compressDirectByteBuffer(dstBuf, // compress into dstBuf
                dstBuf.position(),                   // write compressed data starting at offset position()
                dstBuf.limit() - dstBuf.position(),  // write no more than limit() - position() bytes
                srcBuf,                              // read data to compress from srcBuf
                srcBuf.position(),                   // start reading at position()
                srcBuf.limit() - srcBuf.position()   // read limit() - position() bytes
            );
        srcBuf.position(srcBuf.limit());
        dstBuf.position(dstBuf.position() + size);
        return size;
    }

    /**
     * Compresses the data in buffer 'srcBuf'
     *
     * @param srcBuf the source buffer - must be direct. It is assumed that the `position()` of the
     *               buffer marks the beginning of the uncompressed data to be compressed, and that
     *               the `limit()` of this buffer marks its end.
     *               <p>
     *               When this method returns successfully, its `position()` will be set to its initial `limit()`.
     *               </p>
     * @return A newly allocated direct ByteBuffer containing the compressed data.
     */
    public ByteBuffer compress(ByteBuffer srcBuf) throws ZstdException {
        long maxDstSize = Zstd.compressBound((long)(srcBuf.limit() - srcBuf.position()));
        if (maxDstSize > Integer.MAX_VALUE) {
            throw new ZstdException(Zstd.errGeneric(), "Max output size is greater than MAX_INT");
        }
        ByteBuffer dstBuf = ByteBuffer.allocateDirect((int) maxDstSize);
        int size = compressDirectByteBuffer(dstBuf,    // compress into dstBuf
                  0,                                   // starting at offset 0
                  (int) maxDstSize,                    // writing no more than maxDstSize
                  srcBuf,                              // read data to be compressed from srcBuf
                  srcBuf.position(),                   // start reading at offset position()
                  srcBuf.limit() - srcBuf.position()   // read limit() - position() bytes
            );
        srcBuf.position(srcBuf.limit());

        dstBuf.limit(size);
        // Since we allocated the buffer ourselves, we know it cannot be used to hold any further compressed data,
        // so leave the position at zero where the caller surely wants it, ready to read

        return dstBuf;
    }

    public int compress(byte[] dst, byte[] src) {
        return compressByteArray(dst, 0, dst.length, src, 0, src.length);
    }

    public byte[] compress(byte[] src) {
        long maxDstSize = Zstd.compressBound(src.length);
        if (maxDstSize > Integer.MAX_VALUE) {
            throw new ZstdException(Zstd.errGeneric(), "Max output size is greater than MAX_INT");
        }
        byte[] dst = new byte[(int) maxDstSize];
        int size = compressByteArray(dst, 0, dst.length, src, 0, src.length);
        return Arrays.copyOfRange(dst, 0, size);
    }
}
