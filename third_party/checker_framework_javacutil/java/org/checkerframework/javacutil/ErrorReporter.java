package org.checkerframework.javacutil;

/**
 * Handle errors detected in utility classes. By default, the error reporter throws a
 * RuntimeException, but clients of the utility library may register a handler to change the
 * behavior. For example, type checkers can direct errors to the
 * org.checkerframework.framework.source.SourceChecker class.
 */
public class ErrorReporter {

    protected static ErrorHandler handler = null;

    /** Register a handler to customize error reporting. */
    public static void setHandler(ErrorHandler h) {
        handler = h;
    }

    /**
     * Log an error message and abort processing. Call this method instead of raising an exception.
     *
     * @param msg the error message to log
     */
    public static void errorAbort(String msg) {
        if (handler != null) {
            handler.errorAbort(msg);
        } else {
            throw new RuntimeException(msg, new Throwable());
        }
    }

    /**
     * Log an error message use {@link String#format(String, Object...)}} and abort processing. Call
     * this method instead of raising an exception.
     *
     * @param format a format string
     * @param args arguments to the format string
     */
    public static void errorAbort(String format, Object... args) {
        String formattedMsg = String.format(format, args);
        if (handler != null) {
            handler.errorAbort(formattedMsg);
        } else {
            throw new RuntimeException(formattedMsg, new Throwable());
        }
    }

    public static void errorAbort(String msg, Throwable cause) {
        if (handler != null) {
            handler.errorAbort(msg, cause);
        } else {
            throw new RuntimeException(msg, cause);
        }
    }
}
