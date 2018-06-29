package com.google.devtools.build.lib.skyframe;

/** Base class for exceptions that happen during toolchain resolution. */
public class ToolchainException extends Exception {

  public ToolchainException(String message) {
    super(message);
  }

  public ToolchainException(Throwable cause) {
    super(cause);
  }

  public ToolchainException(String message, Throwable cause) {
    super(message, cause);
  }
}
