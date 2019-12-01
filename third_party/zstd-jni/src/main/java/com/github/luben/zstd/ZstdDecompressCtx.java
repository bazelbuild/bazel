package com.github.luben.zstd;

import com.github.luben.zstd.util.Native;
import com.github.luben.zstd.ZstdDictDecompress;

import java.nio.ByteBuffer;

public class ZstdDecompressCtx extends AutoCloseBase {

    static {
        Native.load();
    }

    private long nativePtr = 0;
    private ZstdDictDecompress decompression_dict = null;

    private native void init();

    private native void free();

    /**
     * Create a context for faster compress operations
     * One such context is required for each thread - put this in a ThreadLocal.
     */
    public ZstdDecompressCtx() {
        init();
        if (0 == nativePtr) {
            throw new IllegalStateException("ZSTD_createDeCompressCtx failed");
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
     * Load decompression dictionary
     *
     * @param dict the dictionary or `null` to remove loaded dictionary
     */
    public void loadDict(ZstdDictDecompress dict) {
        if (nativePtr == 0) {
            throw new IllegalStateException("Decompression context is closed");
        }
        acquireSharedLock();
        dict.acquireSharedLock();
        try {
            long result = loadDDictFast0(dict);
            if (Zstd.isError(result)) {
                throw new ZstdException(result);
            }
            // keep a reference to the dictionary so it's not garbage collected
            decompression_dict = dict;
        } finally {
            dict.releaseSharedLock();
            releaseSharedLock();
        }
    }
    private native long loadDDictFast0(ZstdDictDecompress dict);

    /**
     * Load decompression dictionary.
     *
     * @param dict the dictionary or `null` to remove loaded dictionary
     */
    public void loadDict(byte[] dict) {
        if (nativePtr == 0) {
            throw new IllegalStateException("Compression context is closed");
        }
        acquireSharedLock();
        try {
            long result = loadDDict0(dict);
            if (Zstd.isError(result)) {
                throw new ZstdException(result);
            }
            decompression_dict = null;
        } finally {
            releaseSharedLock();
        }
    }
    private native long loadDDict0(byte[] dict);

    /**
     * Decompresses buffer 'srcBuff' into buffer 'dstBuff' using this ZstdDecompressCtx.
     *
     * Destination buffer should be sized to be larger of equal to the originalSize
     *
     * @param dstBuff the destination buffer - must be direct
     * @param dstOffset the start offset of 'dstBuff'
     * @param dstSize the size of 'dstBuff'
     * @param srcBuff the source buffer - must be direct
     * @param srcOffset the start offset of 'srcBuff'
     * @param srcSize the size of 'srcBuff'
     * @return the number of bytes decompressed into destination buffer (originalSize)
     */
    public int decompressDirectByteBuffer(ByteBuffer dstBuff, int dstOffset, int dstSize, ByteBuffer srcBuff, int srcOffset, int srcSize) {
        if (nativePtr == 0) {
            throw new IllegalStateException("Decompression context is closed");
        }
        if (!srcBuff.isDirect()) {
            throw new IllegalArgumentException("srcBuff must be a direct buffer");
        }
        if (!dstBuff.isDirect()) {
            throw new IllegalArgumentException("dstBuff must be a direct buffer");
        }

        acquireSharedLock();

        try {
            long result = decompressDirectByteBuffer0(dstBuff, dstOffset, dstSize, srcBuff, srcOffset, srcSize);
            if (Zstd.isError(result)) {
                throw new ZstdException(result);
            }
            return (int) result;

        } finally {
            releaseSharedLock();
        }
    }

    private native long decompressDirectByteBuffer0(ByteBuffer dst, int dstOffset, int dstSize, ByteBuffer src, int srcOffset, int srcSize);
}
