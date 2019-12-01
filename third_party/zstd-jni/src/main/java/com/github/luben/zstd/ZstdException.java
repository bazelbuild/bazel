package com.github.luben.zstd;

public class ZstdException extends RuntimeException {
    private long code;

    /**
     * Construct a ZstdException from the result of a Zstd library call.
     *
     * The error code and message are automatically looked up using
     * Zstd.getErrorCode and Zstd.getErrorName.
     *
     * @param result the return value of a Zstd library call
     */
    public ZstdException(long result) {
        this(Zstd.getErrorCode(result), Zstd.getErrorName(result));
    }

    /**
     * Construct a ZstdException with a manually-specified error code and message.
     *
     * No transformation of either the code or message is done. It is advised
     * that one of the Zstd.err*() is used to obtain a stable error code.
     *
     * @param code a Zstd error code
     * @param message the exception's message
     */
    public ZstdException(long code, String message) {
        super(message);
        this.code = code;
    }

    /**
     * Get the Zstd error code that caused the exception.
     *
     * This will likely correspond to one of the Zstd.err*() methods, but the
     * Zstd library may return error codes that are not yet stable. In such
     * cases, this method will return the code reported by Zstd, but it will
     * not correspond to any of the Zstd.err*() methods.
     *
     * @return a Zstd error code
     */
    public long getErrorCode() {
        return code;
    }
}
