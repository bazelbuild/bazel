// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Factory class for creating appropriate instances of {@link SkylarkImports}.
 */
public class SkylarkImports {

  private SkylarkImports() {
    throw new IllegalStateException("This class should not be instantiated");
  }

  // Default implementation class for SkylarkImport.
  private abstract static class SkylarkImportImpl implements SkylarkImport {
    protected String importString;

    @Override
    public String getImportString() {
      return importString;
    }

    @Override
    public abstract PathFragment asPathFragment();

    @Override
    public abstract Label getLabel(Label containingFileLabel);

    @Override
    public boolean hasAbsolutePath() {
      return false;
    }

    @Override
    public PathFragment getAbsolutePath() {
      throw new IllegalStateException("can't request absolute path from a non-absolute import");
    }
  }

  private static final class AbsolutePathImport extends SkylarkImportImpl {
    private PathFragment importPath;

    private AbsolutePathImport(String importString, PathFragment importPath) {
      this.importString = importString;
      this.importPath = importPath;
    }

    @Override
    public PathFragment asPathFragment() {
      return importPath;
    }

    @Override
    public Label getLabel(Label containingFileLabel) {
      throw new IllegalStateException("can't request a label from an absolute path import");
    }

    @Override
    public boolean hasAbsolutePath() {
      return true;
    }

    @Override
    public PathFragment getAbsolutePath() {
      return this.importPath;
    }
  }

  private static final class RelativePathImport extends SkylarkImportImpl {
    private String importFile;

    private RelativePathImport(String importString, String importFile) {
      this.importString = importString;
      this.importFile = importFile;
    }

    @Override
    public PathFragment asPathFragment() {
      return new PathFragment(importFile);
    }

    @Override
    public Label getLabel(Label containingFileLabel) {
      // The twistiness of the code below is due to the fact that the containing file may be in
      // a subdirectory of the package that contains it. We need to construct a Label with
      // the imported file in the same subdirectory of the package.
      PathFragment containingDirInPkg =
          (new PathFragment(containingFileLabel.getName())).getParentDirectory();
      String targetNameForImport = containingDirInPkg.getRelative(importFile).toString();
      try {
        return containingFileLabel.getRelative(targetNameForImport);
      } catch (LabelSyntaxException e) {
        // Shouldn't happen because the parent label is assumed to be valid and the target string is
        // validated on construction.
        throw new IllegalStateException(e);
      }
    }
  }

  private static final class AbsoluteLabelImport extends SkylarkImportImpl {
    private Label importLabel;

    private AbsoluteLabelImport(String importString, Label importLabel) {
      this.importString = importString;
      this.importLabel = importLabel;
    }

    @Override
    public PathFragment asPathFragment() {
      return new PathFragment(PathFragment.ROOT_DIR).getRelative(importLabel.toPathFragment());
    }

    @Override
    public Label getLabel(Label containingFileLabel) {
      // When the import label contains no explicit repository identifier, we resolve it relative
      // to the repo of the containing file.
      return containingFileLabel.resolveRepositoryRelative(importLabel);
    }
  }

  private static final class RelativeLabelImport extends SkylarkImportImpl {
    private String importTarget;

    private RelativeLabelImport(String importString, String importTarget) {
      this.importString = importString;
      this.importTarget = importTarget;
    }

    @Override
    public PathFragment asPathFragment() {
      return new PathFragment(importTarget);
    }

    @Override
    public Label getLabel(Label containingFileLabel) {
      // Unlike a relative path import, the import target is relative to the containing package,
      // not the containing directory within the package.
      try {
        return containingFileLabel.getRelative(importTarget);
      } catch (LabelSyntaxException e) {
        // shouldn't happen because the parent label is assumed validated and the target string is
        // validated on construction
        throw new IllegalStateException(e);
      }
    }
  }

  /**
   * Exception raised for syntactically-invalid Skylark load strings.
   */
  public static class SkylarkImportSyntaxException extends Exception {
    public SkylarkImportSyntaxException(String message) {
      super(message);
    }
  }

  @VisibleForTesting
  static final String INVALID_LABEL_PREFIX = "Invalid label: ";

  @VisibleForTesting
  static final String MUST_HAVE_BZL_EXT_MSG =
      "The label must reference a file with extension '.bzl'";

  @VisibleForTesting
  static final String EXTERNAL_PKG_NOT_ALLOWED_MSG =
  "Skylark files may not be loaded from the //external package";

  @VisibleForTesting
  static final String BZL_EXT_IMPLICIT_MSG =
  "The '.bzl' file extension is implicit; remove it from the path";

  @VisibleForTesting
  static final String INVALID_TARGET_PREFIX = "Invalid target: ";

  @VisibleForTesting
  static final String INVALID_FILENAME_PREFIX = "Invalid filename: ";

  @VisibleForTesting
  static final String RELATIVE_PATH_NO_SUBDIRS_MSG =
  "A relative import path may not contain subdirectories";

  /**
   * Creates and syntactically validates a {@link SkylarkImports} instance from a string.
   * <p>
   * There four syntactic import variants: Absolute paths, relative paths, absolute labels, and
   * relative labels
   *
   * @throws SkylarkImportSyntaxException if the string is not a valid Skylark import.
   */
  public static SkylarkImport create(String importString) throws SkylarkImportSyntaxException {
    if (importString.startsWith("//") || importString.startsWith("@")) {
      // Absolute label.
      Label importLabel;
      try {
        importLabel = Label.parseAbsolute(importString);
      } catch (LabelSyntaxException e) {
        throw new SkylarkImportSyntaxException(INVALID_LABEL_PREFIX + e.getMessage());
      }
      String targetName = importLabel.getName();
      if (!targetName.endsWith(".bzl")) {
        throw new SkylarkImportSyntaxException(MUST_HAVE_BZL_EXT_MSG);
      }
      PackageIdentifier packageId = importLabel.getPackageIdentifier();
      if (packageId.equals(Label.EXTERNAL_PACKAGE_IDENTIFIER)) {
        throw new SkylarkImportSyntaxException(EXTERNAL_PKG_NOT_ALLOWED_MSG);
      }
      return new AbsoluteLabelImport(importString, importLabel);
    } else if (importString.startsWith("/")) {
      // Absolute path.
      if (importString.endsWith(".bzl")) {
        throw new SkylarkImportSyntaxException(BZL_EXT_IMPLICIT_MSG);
      }
      PathFragment importPath = new PathFragment(importString + ".bzl");
      return new AbsolutePathImport(importString, importPath);
    } else if (importString.startsWith(":")) {
      // Relative label. We require that relative labels use an explicit ':' prefix to distinguish
      // them from relative paths, which have a different semantics.
      String importTarget = importString.substring(1);
      if (!importTarget.endsWith(".bzl")) {
        throw new SkylarkImportSyntaxException(MUST_HAVE_BZL_EXT_MSG);
      }
      String maybeErrMsg = LabelValidator.validateTargetName(importTarget);
      if (maybeErrMsg != null) {
        // Null indicates successful target validation.
        throw new SkylarkImportSyntaxException(INVALID_TARGET_PREFIX + maybeErrMsg);
      }
      return new RelativeLabelImport(importString, importTarget);
    } else {
      // Relative path.
      if (importString.endsWith(".bzl")) {
        throw new SkylarkImportSyntaxException(BZL_EXT_IMPLICIT_MSG);
      }
      if (importString.contains("/")) {
        throw new SkylarkImportSyntaxException(RELATIVE_PATH_NO_SUBDIRS_MSG);
      }
      String importTarget = importString + ".bzl";
      String maybeErrMsg = LabelValidator.validateTargetName(importTarget);
      if (maybeErrMsg != null) {
        // Null indicates successful target validation.
        throw new SkylarkImportSyntaxException(INVALID_FILENAME_PREFIX + maybeErrMsg);
      }
      return new RelativePathImport(importString, importTarget);
    }
  }
}

