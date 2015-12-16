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

import static com.google.common.truth.Truth.assertThat;
import static org.hamcrest.CoreMatchers.startsWith;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.syntax.SkylarkImports.SkylarkImportSyntaxException;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link SkylarkImports}.
 */
@RunWith(JUnit4.class)
public class SkylarkImportTest {
  @Rule
  public ExpectedException thrown = ExpectedException.none();

  @Test
  public void testValidAbsoluteLabel() throws Exception {
    String labelToTest = "//some/skylark:file.bzl";
    SkylarkImport importForLabel = SkylarkImports.create(labelToTest);

    assertThat(importForLabel.hasAbsolutePath()).isFalse();

    Label irrelevantContainingFile = Label.parseAbsoluteUnchecked("//another/path:BUILD");
    assertThat(importForLabel.getLabel(irrelevantContainingFile))
        .isEqualTo(Label.parseAbsoluteUnchecked("//some/skylark:file.bzl"));

    thrown.expect(IllegalStateException.class);
    importForLabel.getAbsolutePath();
  }


  @Test
  public void testValidAbsoluteLabelWithRepo() throws Exception {
    String labelToTest = "@my_repo//some/skylark:file.bzl";
    SkylarkImport importForLabel = SkylarkImports.create(labelToTest);

    assertThat(importForLabel.hasAbsolutePath()).isFalse();

    Label irrelevantContainingFile = Label.parseAbsoluteUnchecked("//another/path:BUILD");
    assertThat(importForLabel.getLabel(irrelevantContainingFile))
        .isEqualTo(Label.parseAbsoluteUnchecked("@my_repo//some/skylark:file.bzl"));

    thrown.expect(IllegalStateException.class);
    importForLabel.getAbsolutePath();
  }

  @Test
  public void testValidAbsolutePath() throws Exception {
    String pathToTest = "/some/skylark/file";
    SkylarkImport importForPath = SkylarkImports.create(pathToTest);

    assertThat(importForPath.hasAbsolutePath()).isTrue();

    Label irrelevantContainingFile = Label.parseAbsoluteUnchecked("//another/path:BUILD");
    assertThat(importForPath.getAbsolutePath()).isEqualTo(new PathFragment(pathToTest));

    thrown.expect(IllegalStateException.class);
    importForPath.getLabel(irrelevantContainingFile);
  }

  @Test
  public void testValidRelativeSimpleLabelInPackageDir() throws Exception {
    String labelToTest = ":file.bzl";
    SkylarkImport importForLabel = SkylarkImports.create(labelToTest);

    assertThat(importForLabel.hasAbsolutePath()).isFalse();

    // The import label is relative to the parent's package, not the parent's directory.
    Label containingFile = Label.parseAbsolute("//some/skylark:BUILD");
    assertThat(importForLabel.getLabel(containingFile))
        .isEqualTo(Label.parseAbsolute("//some/skylark:file.bzl"));

    thrown.expect(IllegalStateException.class);
    importForLabel.getAbsolutePath();
  }

  @Test
  public void testValidRelativeSimpleLabelInPackageSubdir() throws Exception {
    String labelToTest = ":file.bzl";
    SkylarkImport importForLabel = SkylarkImports.create(labelToTest);

    assertThat(importForLabel.hasAbsolutePath()).isFalse();

    // The import label is relative to the parent's package, not the parent's directory.
    Label containingFile = Label.parseAbsolute("//some/path/to:skylark/parent.bzl");
    assertThat(importForLabel.getLabel(containingFile))
        .isEqualTo(Label.parseAbsolute("//some/path/to:file.bzl"));

    thrown.expect(IllegalStateException.class);
    importForLabel.getAbsolutePath();
  }

  @Test
  public void testValidRelativeComplexLabelInPackageDir() throws Exception {
    String labelToTest = ":subdir/containing/file.bzl";
    SkylarkImport importForLabel = SkylarkImports.create(labelToTest);

    assertThat(importForLabel.hasAbsolutePath()).isFalse();

    // The import label is relative to the parent's package, not the parent's directory.
    Label containingFile = Label.parseAbsolute("//some/skylark:BUILD");
    assertThat(importForLabel.getLabel(containingFile))
        .isEqualTo(Label.parseAbsolute("//some/skylark:subdir/containing/file.bzl"));

    thrown.expect(IllegalStateException.class);
    importForLabel.getAbsolutePath();
  }

  @Test
  public void testValidRelativeComplexLabelInPackageSubdir() throws Exception {
    String labelToTest = ":subdir/containing/file.bzl";
    SkylarkImport importForLabel = SkylarkImports.create(labelToTest);

    assertThat(importForLabel.hasAbsolutePath()).isFalse();

    // The import label is relative to the parent's package, not the parent's directory.
    Label containingFile = Label.parseAbsolute("//some/path/to:skylark/parent.bzl");
    assertThat(importForLabel.getLabel(containingFile))
        .isEqualTo(Label.parseAbsolute("//some/path/to:subdir/containing/file.bzl"));

    thrown.expect(IllegalStateException.class);
    importForLabel.getAbsolutePath();
  }

  @Test
  public void testValidRelativePathInPackageDir() throws Exception {
    String pathToTest = "file";
    SkylarkImport importForPath = SkylarkImports.create(pathToTest);

    assertThat(importForPath.hasAbsolutePath()).isFalse();

    // The import label is relative to the parent's directory not the parent's package.
    Label containingFile = Label.parseAbsolute("//some/skylark:BUILD");
    assertThat(importForPath.getLabel(containingFile))
        .isEqualTo(Label.parseAbsolute("//some/skylark:file.bzl"));

    thrown.expect(IllegalStateException.class);
    importForPath.getAbsolutePath();
  }

  @Test
  public void testValidRelativePathInPackageSubdir() throws Exception {
    String pathToTest = "file";
    SkylarkImport importForPath = SkylarkImports.create(pathToTest);
    assertThat(importForPath.hasAbsolutePath()).isFalse();

    // The import label is relative to the parent's directory not the parent's package.
    Label containingFile = Label.parseAbsolute("//some/path/to:skylark/parent.bzl");
    assertThat(importForPath.getLabel(containingFile))
        .isEqualTo(Label.parseAbsolute("//some/path/to:skylark/file.bzl"));

    thrown.expect(IllegalStateException.class);
    importForPath.getAbsolutePath();
  }

  @Test
  public void testInvalidAbsoluteLabelSyntax() throws Exception {
    String labelToTest = "//some/skylark/:file.bzl"; // final '/' is illegal
    
    thrown.expect(SkylarkImportSyntaxException.class);
    thrown.expectMessage(startsWith(SkylarkImports.INVALID_LABEL_PREFIX));
    SkylarkImports.create(labelToTest);
  }

  @Test
  public void testInvalidAbsoluteLabelSyntaxWithRepo() throws Exception {
    String labelToTest = "@my_repo//some/skylark/:file.bzl"; // final '/' is illegal
    
    thrown.expect(SkylarkImportSyntaxException.class);
    thrown.expectMessage(startsWith(SkylarkImports.INVALID_LABEL_PREFIX));
    SkylarkImports.create(labelToTest);
  }

  @Test
  public void tesInvalidAbsoluteLabelMissingBzlExt() throws Exception {
    String labelToTest = "//some/skylark:file";
    
    thrown.expect(SkylarkImportSyntaxException.class);
    thrown.expectMessage(SkylarkImports.MUST_HAVE_BZL_EXT_MSG);
    SkylarkImports.create(labelToTest);
  }

  @Test
  public void tesInvalidAbsoluteLabelReferencesExternalPkg() throws Exception {
    String labelToTest = "//external:file.bzl";
    
    thrown.expect(SkylarkImportSyntaxException.class);
    thrown.expectMessage(SkylarkImports.EXTERNAL_PKG_NOT_ALLOWED_MSG);
    SkylarkImports.create(labelToTest);
  }

 @Test
  public void tesInvalidAbsolutePathBzlExtImplicit() throws Exception {
    String labelToTest = "/some/skylark/file.bzl";
    
    thrown.expect(SkylarkImportSyntaxException.class);
    thrown.expectMessage(SkylarkImports.BZL_EXT_IMPLICIT_MSG);
    SkylarkImports.create(labelToTest);
  }

  @Test
  public void testInvalidRelativeLabelMissingBzlExt() throws Exception {
    String labelToTest = ":file";
    
    thrown.expect(SkylarkImportSyntaxException.class);
    thrown.expectMessage(SkylarkImports.MUST_HAVE_BZL_EXT_MSG);
    SkylarkImports.create(labelToTest);
  }

  @Test
  public void testInvalidRelativeLabelSyntax() throws Exception {
    String labelToTest = "::file.bzl";
    
    thrown.expect(SkylarkImportSyntaxException.class);
    thrown.expectMessage(startsWith(SkylarkImports.INVALID_TARGET_PREFIX));
    SkylarkImports.create(labelToTest);
  }

  @Test
  public void testInvalidRelativePathBzlExtImplicit() throws Exception {
    String labelToTest = "file.bzl";
    
    thrown.expect(SkylarkImportSyntaxException.class);
    thrown.expectMessage(SkylarkImports.BZL_EXT_IMPLICIT_MSG);
    SkylarkImports.create(labelToTest);
  }

  @Test
  public void testInvalidRelativePathNoSubdirs() throws Exception {
    String labelToTest = "path/to/file";
    
    thrown.expect(SkylarkImportSyntaxException.class);
    thrown.expectMessage(SkylarkImports.RELATIVE_PATH_NO_SUBDIRS_MSG);
    SkylarkImports.create(labelToTest);
  }

  @Test
  public void testInvalidRelativePathInvalidFilename() throws Exception {
    String labelToTest = "\tfile"; // tab character is invalid
    
    thrown.expect(SkylarkImportSyntaxException.class);
    thrown.expectMessage(startsWith(SkylarkImports.INVALID_FILENAME_PREFIX));
    SkylarkImports.create(labelToTest);
  }
}

