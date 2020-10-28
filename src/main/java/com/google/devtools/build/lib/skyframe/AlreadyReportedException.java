package com.google.devtools.build.lib.skyframe;

/**
 * A marker class for exceptions that are already reported. Once caught, these exceptions shouldn't
 * be reported again.
 */
public class AlreadyReportedException extends Exception {
  public AlreadyReportedException(String message, Throwable cause) {
    super(message, cause);
  }
}
