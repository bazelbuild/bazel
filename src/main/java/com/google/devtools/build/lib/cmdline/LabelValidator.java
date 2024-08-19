// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.cmdline;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.CharMatcher;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * The canonical place to parse and validate Blaze labels.
 */
public final class LabelValidator {

  // Target names allow all 7-bit ASCII characters except
  // 0-31 (control characters)
  // 92 '\' (backslash) - directory separator (on Windows); may be allowed in the future
  // 127 (delete)
  /** Matches punctuation in target names which requires quoting in a blaze query. */
  private static final CharMatcher PUNCTUATION_REQUIRING_QUOTING =
      CharMatcher.anyOf(" \"#$&'()*+,;<=>?[]{|}~");

  /**
   * Matches punctuation in target names which doesn't require quoting in a blaze query.
   *
   * <p>Note that . is also allowed in target names, and doesn't require quoting, but has
   * restrictions on its surrounding characters; see {@link #validateTargetName(String)}. The same
   * applies to :.
   */
  private static final CharMatcher PUNCTUATION_NOT_REQUIRING_QUOTING = CharMatcher.anyOf("!%-@^_`");

  // Package names allow all 7-bit ASCII characters except
  // 0-31 (control characters)
  // 58 ':' (colon) - target name separator
  // 92 '\' (backslash) - directory separator (on Windows); may be allowed in the future
  // 127 (delete)
  /** Matches characters allowed in package name. */
  private static final CharMatcher ALLOWED_CHARACTERS_IN_PACKAGE_NAME =
      CharMatcher.inRange('0', '9')
          .or(CharMatcher.inRange('a', 'z'))
          .or(CharMatcher.inRange('A', 'Z'))
          .or(CharMatcher.anyOf(" !\"#$%&'()*+,-./:;<=>?@[]^_`{|}~"))
          .precomputed();

  /**
   * Matches characters allowed in target names regardless of context.
   *
   * Note that the only other characters allowed in target names are / and . but they have
   * restrictions around surrounding characters; see {@link #validateTargetName(String)}.
   */
  private static final CharMatcher ALWAYS_ALLOWED_TARGET_CHARACTERS =
      CharMatcher.javaLetterOrDigit()
          .or(PUNCTUATION_REQUIRING_QUOTING)
          .or(PUNCTUATION_NOT_REQUIRING_QUOTING)
          // On unix platforms, strings obtained from readdir() are bytes "decoded" as latin1.  But
          // the user is likely to have stored UTF-8 in them.  So we permit all non-ASCII characters
          // in UTF-8 by allowing all high bytes.
          .or(CharMatcher.inRange((char) 128, (char) 255))
          .precomputed();

  @VisibleForTesting
  static final String PACKAGE_NAME_ERROR =
      "package names may contain A-Z, a-z, 0-9, or any of ' !\"#$%&'()*+,-./;<=>?[]^_`{|}~'"
          + " (most 7-bit ascii characters except 0-31, 127, ':', or '\\')";

  @VisibleForTesting
  static final String PACKAGE_NAME_DOT_ERROR =
      "package name component contains only '.' characters";

  /**
   * Performs validity checking of the specified package name. Returns null on success or an error
   * message otherwise.
   *
   * @param packageName the name of the package
   * @return null if {@code name} is valid or an error string if any part
   *   of the package name is invalid
   */
  @Nullable
  public static String validatePackageName(String packageName) {
    int len = packageName.length();
    if (len == 0) {
      // Empty package name (//:foo).
      return null;
    }

    if (packageName.charAt(0) == '/') {
      return "package names may not start with '/'";
    }

    if (!ALLOWED_CHARACTERS_IN_PACKAGE_NAME.matchesAllOf(packageName)) {
      return PACKAGE_NAME_ERROR;
    }

    if (packageName.charAt(packageName.length() - 1) == '/') {
      return "package names may not end with '/'";
    }
    // Check for empty or dot-only package segment
    boolean nonDot = false;
    boolean lastSlash = true;
    // Going backward and marking the last character as being a / so we detect
    // '.' only package segment.
    for (int i = len - 1; i >= -1; --i) {
      char c = (i >= 0) ? packageName.charAt(i) : '/';
      if (c == '/') {
        if (lastSlash) {
          return "package names may not contain '//' path separators";
        }
        if (!nonDot) {
          return PACKAGE_NAME_DOT_ERROR;
        }
        nonDot = false;
        lastSlash = true;
      } else {
        if (c != '.') {
          nonDot = true;
        }
        lastSlash = false;
      }
    }

    return null; // ok
  }

  /**
   * Performs validity checking of the specified target name. Returns null on success or an error
   * message otherwise.
   */
  @Nullable
  public static String validateTargetName(String targetName) {
    // We allow labels equaling '.' or ending in '/.' for now. If we ever
    // actually configure the target we will report an error, but they are permitted for
    // data directories.

    // Code optimized for the common case: success.
    int len = targetName.length();
    if (len == 0) {
      return "empty target name";
    }
    // Forbidden start chars:
    char c = targetName.charAt(0);
    if (c == '/') {
      return "target names may not start with '/'";
    } else if (c == ':') {
      return "target names may not start with ':'";
    } else if (c == '.') {
      if (targetName.startsWith("../") || targetName.equals("..")) {
        return "target names may not contain up-level references '..'";
      } else if (targetName.equals(".")) {
        return null; // See comment above; ideally should be an error.
      } else if (targetName.startsWith("./")) {
        return "target names may not contain '.' as a path segment";
      }
    }

    // Give a friendly error message on CRs in target names
    if (targetName.endsWith("\r")) {
      return "target names may not end with carriage returns " +
             "(perhaps the input source is CRLF-terminated)";
    }

    for (int ii = 0; ii < len; ++ii) {
      c = targetName.charAt(ii);
      if (ALWAYS_ALLOWED_TARGET_CHARACTERS.matches(c)) {
        continue;
      }
      if (c == '.' || c == ':') {
        continue;
      }
      if (c == '/') {
        if (stringRegionMatch(targetName, "/../", ii)) {
          return "target names may not contain up-level references '..'";
        } else if (stringRegionMatch(targetName, "/./", ii)) {
          return "target names may not contain '.' as a path segment";
        } else if (stringRegionMatch(targetName, "//", ii)) {
          return "target names may not contain '//' path separators";
        }
        continue;
      }
      if (c <= '\u001f' || c == '\u007f') {
        return "target names may not contain non-printable characters: '" +
               String.format("\\x%02X", (int) c) + "'";
      }
      return "target names may not contain '" + c + "'";
    }
    // Forbidden end chars:
    if (c == '.') {
      if (targetName.endsWith("/..")) {
        return "target names may not contain up-level references '..'";
      } else if (targetName.endsWith("/.")) {
        return null; // See comment above; ideally should be an error.
      }
    }
    if (c == '/') {
      return "target names may not end with '/'";
    }
    return null; // ok
  }

  // Prefer this implementation over calls to String#subString(), as the latter implies copying
  // the subregion.
  private static boolean stringRegionMatch(String fullString, String possibleMatch, int offset) {
    return fullString.regionMatches(offset, possibleMatch, 0, possibleMatch.length());
  }

  /**
   * Validate the label and parse it into a pair of package name and target name. If the label is
   * not valid, it throws an {@link BadLabelException}.
   *
   * <p>It accepts these forms of labels:
   * <pre>
   * //foo/bar
   * //foo/bar:quux
   * //foo/bar:      (undocumented, but accepted)
   * </pre>
   */
  public static PackageAndTarget validateAbsoluteLabel(String absName) throws BadLabelException {
    PackageAndTarget result = parseAbsoluteLabel(absName);
    String packageName = result.getPackageName();
    String targetName = result.getTargetName();
    String error = validatePackageName(packageName);
    if (error != null) {
      error = "invalid package name '" + packageName + "': " + error;
      // This check is just for a more helpful error message,
      // i.e. valid target name, invalid package name, colon-free label form
      // used => probably they meant "//foo:bar.c" not "//foo/bar.c".
      if (packageName.endsWith("/" + targetName)) {
        error += " (perhaps you meant \":" + targetName + "\"?)";
      }
      throw new BadLabelException(error);
    }
    error = validateTargetName(targetName);
    if (error != null) {
      error = "invalid target name '" + targetName + "': " + error;
      throw new BadLabelException(error);
    }
    return result;
  }

  /**
   * Returns if the label starts with a repository (@whatever) or a package (//whatever).
   */
  public static boolean isAbsolute(String label) {
    return label.startsWith("//") || label.startsWith("@");
  }

  /**
   * Parses the given absolute label by verifying that it starts with "//". If it contains a ':',
   * then the part after that is the target name within the package, and the part before that (but
   * without the leading "//") is the package name. However, it performs no validation on these two
   * pieces.
   *
   * <p>Use of this method is generally not recommended.
   *
   * @throws NullPointerException if {@code absName} is {@code null}
   * @throws BadLabelException if {@code absName} starts with "//"
   */
  public static PackageAndTarget parseAbsoluteLabel(String absName) throws BadLabelException {
    if (!isAbsolute(absName)) {
      throw new BadLabelException("invalid label: " + absName);
    }
    if (absName.startsWith("@")) {
      int endOfRepo = absName.indexOf("//");
      if (endOfRepo < 0) {
        return new PackageAndTarget("", absName.substring(1));
      }
      absName = absName.substring(endOfRepo);
    }
    // Find the package/suffix separation:
    int colonIndex = absName.indexOf(':');
    int splitAt = colonIndex >= 0 ? colonIndex : absName.length();
    String packageName = absName.substring("//".length(), splitAt);
    String suffix = absName.substring(splitAt);
    // ('suffix' is empty, or starts with a colon.)

    // "If packagename and version are elided, the colon is not necessary."
    String targetName = suffix.isEmpty()
        // Target name is last package segment: (works in slash-free case too.)
        ? packageName.substring(packageName.lastIndexOf('/') + 1)
        // Target name is what's after colon:
        : suffix.substring(1);

    return new PackageAndTarget(packageName, targetName);
  }

  /**
   * A pair of package and target names. Note that having an instance of this does not imply that
   * the package or target names are actually valid.
   */
  public static class PackageAndTarget {
    private final String packageName;
    private final String targetName;

    public PackageAndTarget(String packageName, String targetName) {
      this.packageName = packageName;
      this.targetName = targetName;
    }

    public String getPackageName() {
      return packageName;
    }

    public String getTargetName() {
      return targetName;
    }

    @Override
    public String toString() {
      return "//" + packageName + ":" + targetName;
    }

    @Override
    public int hashCode() {
      return Objects.hash(packageName, targetName);
    }

    @Override
    public boolean equals(Object o) {
      if (o == null || o.getClass() != getClass()) {
        return false;
      }
      PackageAndTarget otherTarget = (PackageAndTarget) o;
      return Objects.equals(otherTarget.targetName, targetName)
          && Objects.equals(otherTarget.packageName, packageName);
    }
  }

  /**
   * An exception to notify the caller that a label could not be parsed.
   */
  public static class BadLabelException extends Exception {
    public BadLabelException(String msg) {
      super(msg);
    }
  }
}
