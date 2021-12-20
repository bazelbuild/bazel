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
  static final class Parts {
    /**
     * The {@code @repo} part of the string (sans the {@literal @}); can be null if it doesn't have
     * such a part.
     */
    @Nullable final String repo;
    /** Whether the package part of the string is prefixed by double-slash. */
    final boolean pkgIsAbsolute;
    /** The package part of the string (sans double-slash, if any). */
    final String pkg;
    /** The target part of the string (sans colon). */
    final String target;
    /** The original unparsed raw string. */
    final String raw;

    private Parts(
        @Nullable String repo, boolean pkgIsAbsolute, String pkg, String target, String raw) {
      this.repo = repo;
      this.pkgIsAbsolute = pkgIsAbsolute;
      this.pkg = pkg;
      this.target = target;
      this.raw = raw;
    }

    private static Parts validateAndCreate(
        @Nullable String repo, boolean pkgIsAbsolute, String pkg, String target, String raw)
        throws LabelSyntaxException {
      validateRepoName(repo);
      validatePackageName(pkg, target);
      return new Parts(repo, pkgIsAbsolute, pkg, validateAndProcessTargetName(pkg, target), raw);
    }

    /**
     * Parses a raw label string into parts. The logic can be summarized by the following table:
     *
     * {@code
     *  raw                 | repo   | pkgIsAbsolute | pkg       | target
     * ---------------------+--------+---------------+-----------+-----------
     *  foo/bar             | null   | false         | ""        | "foo/bar"
     *  //foo/bar           | null   | true          | "foo/bar" | "bar"
     *  @repo               | "repo" | true          | ""        | "repo"
     *  @repo//foo/bar      | "repo" | true          | "foo/bar" | "bar"
     *  :quux               | null   | false         | ""        | "quux"
     *  foo/bar:quux        | null   | false         | "foo/bar" | "quux"
     *  //foo/bar:quux      | null   | true          | "foo/bar" | "quux"
     *  @repo//foo/bar:quux | "repo" | true          | "foo/bar" | "quux"
     * }
     */
    static Parts parse(String rawLabel) throws LabelSyntaxException {
      @Nullable final String repo;
      final int startOfPackage;
      final int doubleSlashIndex = rawLabel.indexOf("//");
      final boolean pkgIsAbsolute;
      if (rawLabel.startsWith("@")) {
        if (doubleSlashIndex < 0) {
          // Special case: the label "@foo" is synonymous with "@foo//:foo".
          repo = rawLabel.substring(1);
          return validateAndCreate(
              repo, /*pkgIsAbsolute=*/ true, /*pkg=*/ "", /*target=*/ repo, rawLabel);
        } else {
          repo = rawLabel.substring(1, doubleSlashIndex);
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
      if (colonIndex >= 0) {
        pkg = rawLabel.substring(startOfPackage, colonIndex);
        target = rawLabel.substring(colonIndex + 1);
      } else if (pkgIsAbsolute) {
        // Special case: the label "[@repo]//foo/bar" is synonymous with "[@repo]//foo/bar:bar".
        pkg = rawLabel.substring(startOfPackage);
        // The target name is the last package segment (works even if `pkg` contains no slash)
        target = pkg.substring(pkg.lastIndexOf('/') + 1);
      } else {
        // Special case: the label "foo/bar" is synonymous with ":foo/bar".
        pkg = "";
        target = rawLabel.substring(startOfPackage);
      }
      return validateAndCreate(repo, pkgIsAbsolute, pkg, target, rawLabel);
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
      if (!pkgIsAbsolute) {
        throw syntaxErrorf("invalid label '%s': absolute label must begin with '@' or '//'", raw);
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

  static String validateAndProcessTargetName(String pkg, String target)
      throws LabelSyntaxException {
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
