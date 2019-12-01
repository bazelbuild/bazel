package com.github.luben.zstd;

import com.github.luben.zstd.util.Native;
import com.github.luben.zstd.ZstdDictCompress;

import java.nio.ByteBuffer;

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
    public void setLevel(int level) {
        if (nativePtr == 0) {
            throw new IllegalStateException("Compression context is closed");
        }
        acquireSharedLock();
        setLevel0(level);
        releaseSharedLock();
    }
    private native void setLevel0(int level);

    /**
     * Enable or disable compression checksums
     * @param checksumFlag A 32-bits checksum of content is written at end of frame, default: false
     */
    public void setChecksum(boolean checksumFlag) {
        if (nativePtr == 0) {
            throw new IllegalStateException("Compression context is closed");
        }
        acquireSharedLock();
        setChecksum0(checksumFlag);
        releaseSharedLock();
    }
    private native void setChecksum0(boolean checksumFlag);

    /**
     * Enable or disable content size
     * @param contentSizeFlag Content size will be written into frame header _whenever known_, default: true
     */
    public void setContentSize(boolean contentSizeFlag) {
        if (nativePtr == 0) {
            throw new IllegalStateException("Compression context is closed");
        }
        acquireSharedLock();
        setContentSize0(contentSizeFlag);
        releaseSharedLock();
    }
    private native void setContentSize0(boolean contentSizeFlag);

    /**
     * Enable or disable dictID
     * @param dictIDFlag When applicable, dictionary's ID is written into frame header, default: true
     */
    public void setDictID(boolean dictIDFlag) {
        if (nativePtr == 0) {
            throw new IllegalStateException("Compression context is closed");
        }
        acquireSharedLock();
        setDictID0(dictIDFlag);
        releaseSharedLock();
    }
    private native void setDictID0(boolean dictIDFlag);

    /**
     * Load compression dictionary to be used for subsequently compressed frames.
     *
     * @param dict the dictionary or `null` to remove loaded dictionary
     */
    public void loadDict(ZstdDictCompress dict) {
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
    }
    private native long loadCDictFast0(ZstdDictCompress dict);

    /**
     * Load compression dictionary to be used for subsequently compressed frames.
     *
     * @param dict the dictionary or `null` to remove loaded dictionary
     */
    public void loadDict(byte[] dict) {
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
    }
    private native long loadCDict0(byte[] dict);

    /**
     * Compresses buffer 'srcBuff' into buffer 'dstBuff' reusing this ZstdCompressCtx.
     *
     * Destination buffer should be sized to handle worst cases situations (input
     * data not compressible). Worst case size evaluation is provided by function
     * ZSTD_compressBound().
     * Note: compression level and parameters are
     *
     * @param dstBuff the destination buffer - must be direct
     * @param dstOffset the start offset of 'dstBuff'
     * @param dstSize the size of 'dstBuff'
     * @param srcBuff the source buffer - must be direct
     * @param srcOffset the start offset of 'srcBuff'
     * @param srcSize the length of 'srcBuff'
     * @return  the number of bytes written into buffer 'dstBuff'.
     */
    public long compressDirectByteBuffer(ByteBuffer dstBuff, int dstOffset, int dstSize, ByteBuffer srcBuff, int srcOffset, int srcSize) {
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
            long result = compressDirectByteBuffer0(dstBuff, dstOffset, dstSize, srcBuff, srcOffset, srcSize);
            if (Zstd.isError(result)) {
                throw new ZstdException(result);
            }
            return result;

        } finally {
            releaseSharedLock();
        }
    }

    private native long compressDirectByteBuffer0(ByteBuffer dst, int dstOffset, int dstSize, ByteBuffer src, int srcOffset, int srcSize);
}
