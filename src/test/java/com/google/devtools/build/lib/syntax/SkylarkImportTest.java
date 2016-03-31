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

  private void validAbsoluteLabelTest(String labelString, String expectedLabelString,
      String expectedPathString) throws Exception {
    SkylarkImport importForLabel = SkylarkImports.create(labelString);

    assertThat(importForLabel.hasAbsolutePath()).named("hasAbsolutePath()").isFalse();
    assertThat(importForLabel.getImportString()).named("getIMportString()").isEqualTo(labelString);

    Label irrelevantContainingFile = Label.parseAbsoluteUnchecked("//another/path:BUILD");
    assertThat(importForLabel.getLabel(irrelevantContainingFile)).named("getLabel()")
        .isEqualTo(Label.parseAbsoluteUnchecked(expectedLabelString));
 
    assertThat(importForLabel.asPathFragment()).named("asPathFragment()")
        .isEqualTo(new PathFragment(expectedPathString));

    thrown.expect(IllegalStateException.class);
    importForLabel.getAbsolutePath();   
  }

  @Test
  public void testValidAbsoluteLabel() throws Exception {
    validAbsoluteLabelTest("//some/skylark:file.bzl",
        /*expected label*/ "//some/skylark:file.bzl",
        /*expected path*/ "/some/skylark/file.bzl");
  }

  @Test
  public void testValidAbsoluteLabelWithRepo() throws Exception {
    validAbsoluteLabelTest("@my_repo//some/skylark:file.bzl",
        /*expected label*/ "@my_repo//some/skylark:file.bzl",
        /*expected path*/ "/some/skylark/file.bzl");
  }

  @Test
  public void testValidAbsolutePath() throws Exception {
    String pathToTest = "/some/skylark/file";
    SkylarkImport importForPath = SkylarkImports.create(pathToTest);

    assertThat(importForPath.hasAbsolutePath()).named("hasAbsolutePath()").isTrue();
    assertThat(importForPath.getImportString()).named("getImportString()").isEqualTo(pathToTest);

    Label irrelevantContainingFile = Label.parseAbsoluteUnchecked("//another/path:BUILD");
    assertThat(importForPath.getAbsolutePath()).named("getAbsolutePath()")
        .isEqualTo(new PathFragment("//some/skylark/file.bzl"));

     assertThat(importForPath.asPathFragment()).named("asPathFragment()")
        .isEqualTo(new PathFragment("/some/skylark/file.bzl"));

    thrown.expect(IllegalStateException.class);
    importForPath.getLabel(irrelevantContainingFile);
  }

  private void validRelativeLabelTest(String labelString,
      String containingLabelString, String expectedLabelString, String expectedPathString)
          throws Exception {
    SkylarkImport importForLabel = SkylarkImports.create(labelString);

    assertThat(importForLabel.hasAbsolutePath()).named("hasAbsolutePath()").isFalse();
    assertThat(importForLabel.getImportString()).named("getImportString()").isEqualTo(labelString);

    // The import label is relative to the parent's package, not the parent's directory.
    Label containingLabel = Label.parseAbsolute(containingLabelString);
    assertThat(importForLabel.getLabel(containingLabel)).named("getLabel()")
        .isEqualTo(Label.parseAbsolute(expectedLabelString));

    assertThat(importForLabel.asPathFragment()).named("asPathFragment()")
        .isEqualTo(new PathFragment(expectedPathString));

   thrown.expect(IllegalStateException.class);
    importForLabel.getAbsolutePath();
  }

  @Test
  public void testValidRelativeSimpleLabelInPackageDir() throws Exception {
    validRelativeLabelTest(":file.bzl",
        /*containing*/ "//some/skylark:BUILD",
        /*expected label*/ "//some/skylark:file.bzl",
        /*expected path*/ "file.bzl");
  }

  @Test
  public void testValidRelativeSimpleLabelInPackageSubdir() throws Exception {
    validRelativeLabelTest(":file.bzl",
        /*containing*/ "//some/path/to:skylark/parent.bzl",
        /*expected label*/ "//some/path/to:file.bzl",
        /*expected path*/ "file.bzl");
  }

  @Test
  public void testValidRelativeComplexLabelInPackageDir() throws Exception {
    validRelativeLabelTest(":subdir/containing/file.bzl",
        /*containing*/ "//some/skylark:BUILD",
        /*expected label*/ "//some/skylark:subdir/containing/file.bzl",
        /*expected path*/ "subdir/containing/file.bzl");
  }

  @Test
  public void testValidRelativeComplexLabelInPackageSubdir() throws Exception {
    validRelativeLabelTest(":subdir/containing/file.bzl",
        /*containing*/ "//some/path/to:skylark/parent.bzl",
        /*expected label*/ "//some/path/to:subdir/containing/file.bzl",
        /*expected path*/ "subdir/containing/file.bzl");
  }

  private void validRelativePathTest(String pathString, String containingLabelString,
      String expectedLabelString, String expectedPathString) throws Exception {
    SkylarkImport importForPath = SkylarkImports.create(pathString);

    assertThat(importForPath.hasAbsolutePath()).named("hasAbsolutePath()").isFalse();

    // The import label is relative to the parent's directory not the parent's package.
    Label containingLabel = Label.parseAbsolute(containingLabelString);
    assertThat(importForPath.getLabel(containingLabel)).named("getLabel()")
        .isEqualTo(Label.parseAbsolute(expectedLabelString));
    
    assertThat(importForPath.asPathFragment()).named("asPathFragment()")
        .isEqualTo(new PathFragment(expectedPathString));

    thrown.expect(IllegalStateException.class);
    importForPath.getAbsolutePath();
  }

  @Test
  public void testValidRelativePathInPackageDir() throws Exception {
    validRelativePathTest("file",
        /*containing*/ "//some/skylark:BUILD",
        /*expected label*/ "//some/skylark:file.bzl",
        /*expected path*/ "file.bzl");
  }

  @Test
  public void testValidRelativePathInPackageSubdir() throws Exception {
    validRelativePathTest("file",
        /*containing*/ "//some/path/to:skylark/parent.bzl",
        /*expected label*/ "//some/path/to:skylark/file.bzl",
        /*expected path*/ "file.bzl");
  }

  private void invalidImportTest(String importString, String expectedMsgPrefix) throws Exception {
    thrown.expect(SkylarkImportSyntaxException.class);
    thrown.expectMessage(startsWith(expectedMsgPrefix));
    SkylarkImports.create(importString);   
  }

  @Test
  public void testInvalidAbsoluteLabelSyntax() throws Exception {
    // final '/' is illegal
    invalidImportTest("//some/skylark/:file.bzl", SkylarkImports.INVALID_LABEL_PREFIX);
  }

  @Test
  public void testInvalidAbsoluteLabelSyntaxWithRepo() throws Exception {
    // final '/' is illegal
    invalidImportTest("@my_repo//some/skylark/:file.bzl", SkylarkImports.INVALID_LABEL_PREFIX);
  }

  @Test
  public void tesInvalidAbsoluteLabelMissingBzlExt() throws Exception {
    invalidImportTest("//some/skylark:file", SkylarkImports.MUST_HAVE_BZL_EXT_MSG);
  }

  @Test
  public void tesInvalidAbsoluteLabelReferencesExternalPkg() throws Exception {
    invalidImportTest("//external:file.bzl", SkylarkImports.EXTERNAL_PKG_NOT_ALLOWED_MSG);
  }

  @Test
  public void tesInvalidAbsolutePathBzlExtImplicit() throws Exception {
    invalidImportTest("/some/skylark/file.bzl", SkylarkImports.INVALID_PATH_SYNTAX);
  }

  @Test
  public void testInvalidRelativeLabelMissingBzlExt() throws Exception {
    invalidImportTest(":file", SkylarkImports.MUST_HAVE_BZL_EXT_MSG);
  }

  @Test
  public void testInvalidRelativeLabelSyntax() throws Exception {
    invalidImportTest("::file.bzl", SkylarkImports.INVALID_TARGET_PREFIX);
  }

  @Test
  public void testInvalidRelativePathBzlExtImplicit() throws Exception {
    invalidImportTest("file.bzl", SkylarkImports.INVALID_PATH_SYNTAX);
  }

  @Test
  public void testInvalidRelativePathNoSubdirs() throws Exception {
    invalidImportTest("path/to/file", SkylarkImports.INVALID_PATH_SYNTAX);
  }

  @Test
  public void testInvalidRelativePathInvalidFilename() throws Exception {
    // tab character is invalid
    invalidImportTest("\tfile", SkylarkImports.INVALID_FILENAME_PREFIX);
  }
}
