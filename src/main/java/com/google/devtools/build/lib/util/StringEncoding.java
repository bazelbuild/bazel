package com.google.devtools.build.lib.util;

import com.google.devtools.build.lib.unsafe.StringUnsafe;
import com.google.protobuf.ByteString;
import java.nio.charset.Charset;


import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.US_ASCII;
import static java.nio.charset.StandardCharsets.UTF_8;

public final class StringEncoding {

  /**
   * Even though Bazel attempts to force the default charset to be ISO-8859-1, which makes String
   * identical to the "bag of raw bytes" that a UNIX path is, the JVM may still use a different
   * encoding for file paths and command-line arguments, e.g. on macOS (always UTF-8). When
   * interoperating with Java APIs, we thus need to reencode paths to the JVM's native encoding.
   */
  private static final Charset SUN_JNU_ENCODING =
      Charset.forName(System.getProperty("sun.jnu.encoding"));

  // This only exists for RemoteWorker, which directly uses the RE APIs UTF-8-encoded string with
  // the JavaIoFileSystem and thus shouldn't be subject to any reencoding.
  private static final boolean BAZEL_UNICODE_STRINGS =
      Boolean.getBoolean("bazel.internal.UseUtf8ForStrings");

  private StringEncoding() {}

  /**
   * Reencodes a string using Bazel's internal raw byte encoding into the equivalent representation
   * for Java stdlib functions, if necessary.
   */
  public static String internalStringToPlatformString(String s) {
    return needsReencodeForPlatform(s) ? new String(s.getBytes(ISO_8859_1), UTF_8) : s;
  }

  /**
   * Reencodes a string obtained from Java stdlib functions into Bazel's internal raw byte encoding,
   * if necessary.
   */
  public static String platformStringToInternalString(String s) {
    return needsReencodeForPlatform(s) ? new String(s.getBytes(UTF_8), ISO_8859_1) : s;
  }

  public static String platformBytesToInternalString(ByteString bytes) {
    return bytes.toString(ISO_8859_1);
  }

  public static String internalBytesToPlatformString(ByteString bytes) {
    return internalStringToPlatformString(bytes.toString(ISO_8859_1));
  }

  private static boolean needsReencodeForPlatform(String s) {
    // The comparisons below are expected to be constant-folded by the JIT.
    if (BAZEL_UNICODE_STRINGS) {
      return false;
    }
    if (SUN_JNU_ENCODING == US_ASCII) {
      return false;
    }
    if (SUN_JNU_ENCODING == ISO_8859_1 && OS.getCurrent() != OS.WINDOWS) {
      return false;
    }
    return !StringUnsafe.getInstance().isAscii(s);
  }

  /**
   * Decode a String that might actually be UTF-8, in which case each input character will be
   * treated as a byte.
   *
   * <p>Several Bazel subsystems, including Starlark, store bytes in `String` values where each
   * `char` stores one `byte` in its lower 8 bits. This function converts its input to a `[]byte`,
   * then decodes that byte array as UTF-8.
   *
   * <p>Using U+2049 (EXCLAMATION QUESTION MARK) as an example:
   *
   * <p>"\u2049".getBytes(UTF_8) == [0xE2, 0x81, 0x89]
   *
   * <p>decodeBytestringUtf8("\u00E2\u0081\u0089") == "\u2049"
   *
   * <p>The return value is suitable for passing to Protobuf string fields or printing to the
   * terminal.
   */
  public static String reencodeInternalToUtf8(String maybeUtf8) {
    if (StringUnsafe.getInstance().isAscii(maybeUtf8)) {
      return maybeUtf8;
    }

    // Try our best to get a valid Unicode string, assuming that the input
    // is either UTF-8 (from Starlark or a UNIX file path) or already valid
    // Unicode (from a Windows file path).
    if (maybeUtf8.chars().anyMatch(c -> c > 0xFF)) {
      return maybeUtf8;
    }

    final byte[] utf8 = maybeUtf8.getBytes(ISO_8859_1);
    final String decoded = new String(utf8, UTF_8);

    // If the input was Unicode that happens to contain only codepoints in
    // the ISO-8859-1 range, then it will probably have a partial decoding
    // failure.
    if (decoded.chars().anyMatch(c -> c == 0xFFFD)) {
      return maybeUtf8;
    }

    return decoded;
  }

  /**
   * Encodes a String to UTF-8, then converts those UTF-8 bytes to a String by zero-extending each
   * `byte` into a `char`.
   *
   * <p>Using U+2049 (EXCLAMATION QUESTION MARK) as an example:
   *
   * <p>"\u2049".getBytes(UTF_8) == [0xE2, 0x81, 0x89]
   *
   * <p>encodeBytestringUtf8("\u2049") == "\u00E2\u0081\u0089"
   *
   * <p>See {@link #reencodeInternalToUtf8} for motivation.
   */
  public static String reencodeUtf8ToInternal(String unicode) {
    if (StringUnsafe.getInstance().isAscii(unicode)) {
      return unicode;
    }
    final byte[] utf8 = unicode.getBytes(UTF_8);
    return new String(utf8, ISO_8859_1);
  }
}
