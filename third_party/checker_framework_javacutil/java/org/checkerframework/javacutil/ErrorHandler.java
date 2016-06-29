package org.checkerframework.javacutil;

/**
 * An implementation of the ErrorHandler interface can be registered
 * with the ErrorReporter class to change the default behavior on
 * errors.
 */
public interface ErrorHandler {

    /**
     * Log an error message and abort processing.
     *
     * @param msg the error message to log
     */
    public void errorAbort(String msg);

    public void errorAbort(String msg, Throwable cause);
}
