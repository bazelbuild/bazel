package org.brotli.integration;

/**
 * Optionally loads brotli JNI wrapper native library.
 */
public class BrotliJniTestBase {
  static {
    String jniLibrary = System.getProperty("BROTLI_JNI_LIBRARY");
    if (jniLibrary != null) {
      System.load(new java.io.File(jniLibrary).getAbsolutePath());
    }
  }
}