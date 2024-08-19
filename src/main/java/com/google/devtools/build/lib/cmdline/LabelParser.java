// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.errorprone.annotations.FormatMethod;
import javax.annotation.Nullable;

/** Utilities to help parse labels. */
final class LabelParser {
  private LabelParser() {}

  /**
   * Contains the parsed elements of a label string. The parts are validated (they don't contain
   * invalid characters). See {@link #parse} for valid label patterns.
   */
  @AutoValue
  abstract static class Parts {
    /**
     * The {@code @repo} or {@code @@canonical_repo} part of the string (sans any leading
     * {@literal @}s); can be null if it doesn't have such a part (i.e. if it doesn't start with a
     * {@literal @}).
     */
    @Nullable
    abstract String repo();
    /**
     * Whether the repo part is using the canonical repo syntax (two {@literal @}s) or not (one
     * {@literal @}). If there is no repo part, this is false.
     */
    abstract boolean repoIsCanonical();
    /**
     * Whether the package part of the string is prefixed by double-slash. This can only be false if
     * the repo part is missing.
     */
    abstract boolean pkgIsAbsolute();
    /**
     * The package part of the string (sans the leading double-slash, if present; also sans the
     * final '...' segment, if present).
     */
    abstract String pkg();
    /** Whether the package part of the string ends with a '...' segment. */
    abstract boolean pkgEndsWithTripleDots();
    /** The target part of the string (sans colon). */
    abstract String target();
    /** The original unparsed raw string. */
    abstract String raw();

    @VisibleForTesting
    static Parts validateAndCreate(
        @Nullable String repo,
        boolean repoIsCanonical,
        boolean pkgIsAbsolute,
        String pkg,
        boolean pkgEndsWithTripleDots,
        String target,
        String raw)
        throws LabelSyntaxException {
      validateRepoName(repo);
      validatePackageName(pkg, target);
      return new AutoValue_LabelParser_Parts(
          repo,
          repoIsCanonical,
          pkgIsAbsolute,
          pkg,
          pkgEndsWithTripleDots,
          validateAndProcessTargetName(pkg, target, pkgEndsWithTripleDots),
          raw);
    }

    /**
     * Parses a raw label string into parts. The logic can be summarized by the following table:
     *
     * <pre>{@code
     *  raw                  | repo   | repoIs-   | pkgIs-   | pkg       | pkgEndsWith- | target
     *                       |        | Canonical | Absolute |           | TripleDots   |
     * ----------------------+--------+-----------+----------+-----------+--------------+-----------
     * "foo/bar"             | null   | false     | false    | ""        | false        | "foo/bar"
     * "..."                 | null   | false     | false    | ""        | true         | ""
     * "...:all"             | null   | false     | false    | ""        | true         | "all"
     * "foo/..."             | null   | false     | false    | "foo"     | true         | ""
     * "//foo/bar"           | null   | false     | true     | "foo/bar" | false        | "bar"
     * "//foo/..."           | null   | false     | true     | "foo"     | true         | ""
     * "//foo/...:all"       | null   | false     | true     | "foo"     | true         | "all"
     * "//foo/all"           | null   | false     | true     | "foo/all" | false        | "all"
     * "@repo"               | "repo" | false     | true     | ""        | false        | "repo"
     * "@@repo"              | "repo" | true      | true     | ""        | false        | "repo"
     * "@repo//foo/bar"      | "repo" | false     | true     | "foo/bar" | false        | "bar"
     * "@@repo//foo/bar"     | "repo" | true      | true     | "foo/bar" | false        | "bar"
     * ":quux"               | null   | false     | false    | ""        | false        | "quux"
     * ":qu:ux"              | null   | false     | false    | ""        | false        | "qu:ux"
     * "foo/bar:quux"        | null   | false     | false    | "foo/bar" | false        | "quux"
     * "foo/bar:qu:ux"       | null   | false     | false    | "foo/bar" | false        | "qu:ux"
     * "//foo/bar:quux"      | null   | false     | true     | "foo/bar" | false        | "quux"
     * "@repo//foo/bar:quux" | "repo" | false     | true     | "foo/bar" | false        | "quux"
     * }</pre>
     */
    static Parts parse(String rawLabel) throws LabelSyntaxException {
      @Nullable final String repo;
      final boolean repoIsCanonical = rawLabel.startsWith("@@");
      final int startOfPackage;
      final int doubleSlashIndex = rawLabel.indexOf("//");
      final boolean pkgIsAbsolute;
      if (rawLabel.startsWith("@")) {
        if (doubleSlashIndex < 0) {
          // Special case: the label "@foo" is synonymous with "@foo//:foo".
          repo = rawLabel.substring(repoIsCanonical ? 2 : 1);
          return validateAndCreate(
              repo,
              repoIsCanonical,
              /* pkgIsAbsolute= */ true,
              /* pkg= */ "",
              /* pkgEndsWithTripleDots= */ false,
              /* target= */ repo,
              rawLabel);
        } else {
          repo = rawLabel.substring(repoIsCanonical ? 2 : 1, doubleSlashIndex);
          startOfPackage = doubleSlashIndex + 2;
          pkgIsAbsolute = true;
        }
      } else {
        // If the label begins with '//', it's an absolute label. Otherwise, treat it as relative
        // (the command-line kind).
        pkgIsAbsolute = doubleSlashIndex == 0;
        startOfPackage = doubleSlashIndex == 0 ? 2 : 0;
        repo = null;
      }

      final String pkg;
      final String target;
      final int colonIndex = rawLabel.indexOf(':', startOfPackage);
      final String rawPkg =
          rawLabel.substring(startOfPackage, colonIndex >= 0 ? colonIndex : rawLabel.length());
      final boolean pkgEndsWithTripleDots = rawPkg.endsWith("/...") || rawPkg.equals("...");
      if (colonIndex < 0 && pkgEndsWithTripleDots) {
        // Special case: if the entire label ends in '...', the target name is empty.
        pkg = stripTrailingTripleDots(rawPkg);
        target = "";
      } else if (colonIndex < 0 && !pkgIsAbsolute) {
        // Special case: the label "foo/bar" is synonymous with ":foo/bar".
        pkg = "";
        target = rawLabel.substring(startOfPackage);
      } else {
        pkg = stripTrailingTripleDots(rawPkg);
        if (colonIndex >= 0) {
          target = rawLabel.substring(colonIndex + 1);
        } else {
          // Special case: the label "[@repo]//foo/bar" is synonymous with "[@repo]//foo/bar:bar".
          // The target name is the last package segment (works even if `pkg` contains no slash)
          target = pkg.substring(pkg.lastIndexOf('/') + 1);
        }
      }
      return validateAndCreate(
          repo, repoIsCanonical, pkgIsAbsolute, pkg, pkgEndsWithTripleDots, target, rawLabel);
    }

    private static String stripTrailingTripleDots(String pkg) {
      if (pkg.endsWith("/...")) {
        return pkg.substring(0, pkg.length() - 4);
      }
      if (pkg.equals("...")) {
        return "";
      }
      return pkg;
    }

    private static void validateRepoName(@Nullable String repo) throws LabelSyntaxException {
      if (repo != null) {
        RepositoryName.validate(repo);
      }
    }

    private static void validatePackageName(String pkg, String target) throws LabelSyntaxException {
      String pkgError = LabelValidator.validatePackageName(pkg);
      if (pkgError != null) {
        throw syntaxErrorf(
            "invalid package name '%s': %s%s", pkg, pkgError, perhapsYouMeantMessage(pkg, target));
      }
    }

    void checkPkgIsAbsolute() throws LabelSyntaxException {
      if (!pkgIsAbsolute()) {
        throw syntaxErrorf("invalid label '%s': absolute label must begin with '@' or '//'", raw());
      }
    }

    void checkPkgDoesNotEndWithTripleDots() throws LabelSyntaxException {
      if (pkgEndsWithTripleDots()) {
        throw syntaxErrorf("invalid label '%s': package name cannot contain '...'", raw());
      }
    }
  }

  @FormatMethod
  static LabelSyntaxException syntaxErrorf(String format, Object... args) {
    return new LabelSyntaxException(
        StringUtilities.sanitizeControlChars(String.format(format, args)));
  }

  private static String perhapsYouMeantMessage(String pkg, String target) {
    return pkg.endsWith('/' + target) ? " (perhaps you meant \":" + target + "\"?)" : "";
  }

  static String validateAndProcessTargetName(
      String pkg, String target, boolean pkgEndsWithTripleDots) throws LabelSyntaxException {
    if (pkgEndsWithTripleDots && target.isEmpty()) {
      // Allow empty target name if the package part ends in '...'.
      return target;
    }
    String targetError = LabelValidator.validateTargetName(target);
    if (targetError != null) {
      throw syntaxErrorf(
          "invalid target name '%s': %s%s",
          target, targetError, perhapsYouMeantMessage(pkg, target));
    }
    // TODO(bazel-team): This should be an error, but we can't make it one for legacy reasons.
    if (target.endsWith("/.")) {
      return target.substring(0, target.length() - 2);
    }
    return target;
  }
}
