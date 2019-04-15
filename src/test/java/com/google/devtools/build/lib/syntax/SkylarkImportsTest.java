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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.syntax.SkylarkImports.SkylarkImportSyntaxException;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link SkylarkImports}.
 */
@RunWith(JUnit4.class)
public class SkylarkImportsTest {
  @Rule
  public ExpectedException thrown = ExpectedException.none();

  private void validAbsoluteLabelTest(String labelString, String expectedLabelString)
      throws Exception {
    SkylarkImport importForLabel = SkylarkImports.create(labelString);

    assertThat(importForLabel.getImportString()).named("getImportString()").isEqualTo(labelString);

    Label irrelevantContainingFile = Label.parseAbsoluteUnchecked("//another/path:BUILD");
    assertThat(importForLabel.getLabel(irrelevantContainingFile)).named("getLabel()")
        .isEqualTo(Label.parseAbsoluteUnchecked(expectedLabelString));
  }

  @Test
  public void testValidAbsoluteLabel() throws Exception {
    validAbsoluteLabelTest("//some/skylark:file.bzl", /*expected label*/ "//some/skylark:file.bzl");
  }

  @Test
  public void testValidAbsoluteLabelWithRepo() throws Exception {
    validAbsoluteLabelTest(
        "@my_repo//some/skylark:file.bzl", /*expected label*/ "@my_repo//some/skylark:file.bzl");
  }

  @Test
  public void testValidAbsoluteLabelWithRepoRemapped() throws Exception {
    String labelString = "@orig_repo//some/skylark:file.bzl";
    String remappedLabelString = "@new_repo//some/skylark:file.bzl";
    ImmutableMap<RepositoryName, RepositoryName> repositoryMapping =
        ImmutableMap.of(RepositoryName.create("@orig_repo"), RepositoryName.create("@new_repo"));
    SkylarkImport importForLabel = SkylarkImports.create(labelString, repositoryMapping);

    assertThat(importForLabel.getImportString()).named("getImportString()").isEqualTo(labelString);

    Label irrelevantContainingFile = Label.parseAbsoluteUnchecked("//another/path:BUILD");
    assertThat(importForLabel.getLabel(irrelevantContainingFile))
        .named("getLabel()")
        .isEqualTo(Label.parseAbsoluteUnchecked(remappedLabelString));
  }

  private void validRelativeLabelTest(
      String labelString, String containingLabelString, String expectedLabelString)
      throws Exception {
    SkylarkImport importForLabel = SkylarkImports.create(labelString);

    assertThat(importForLabel.getImportString()).named("getImportString()").isEqualTo(labelString);

    // The import label is relative to the parent's package, not the parent's directory.
    Label containingLabel = Label.parseAbsolute(containingLabelString, ImmutableMap.of());
    assertThat(importForLabel.getLabel(containingLabel))
        .named("getLabel()")
        .isEqualTo(Label.parseAbsolute(expectedLabelString, ImmutableMap.of()));
  }

  @Test
  public void testValidRelativeSimpleLabelInPackageDir() throws Exception {
    validRelativeLabelTest(
        ":file.bzl",
        /*containing*/ "//some/skylark:BUILD",
        /*expected label*/ "//some/skylark:file.bzl");
  }

  @Test
  public void testValidRelativeSimpleLabelInPackageSubdir() throws Exception {
    validRelativeLabelTest(
        ":file.bzl",
        /*containing*/ "//some/path/to:skylark/parent.bzl",
        /*expected label*/ "//some/path/to:file.bzl");
  }

  @Test
  public void testValidRelativeComplexLabelInPackageDir() throws Exception {
    validRelativeLabelTest(
        ":subdir/containing/file.bzl",
        /*containing*/ "//some/skylark:BUILD",
        /*expected label*/ "//some/skylark:subdir/containing/file.bzl");
  }

  @Test
  public void testValidRelativeComplexLabelInPackageSubdir() throws Exception {
    validRelativeLabelTest(
        ":subdir/containing/file.bzl",
        /*containing*/ "//some/path/to:skylark/parent.bzl",
        /*expected label*/ "//some/path/to:subdir/containing/file.bzl");
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
  public void testInvalidPathSyntax() throws Exception {
    invalidImportTest("some/path/foo.bzl", SkylarkImports.INVALID_PATH_SYNTAX);
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
    invalidImportTest("path/to/file.bzl", SkylarkImports.INVALID_PATH_SYNTAX);
  }

  @Test
  public void testInvalidRelativePathInvalidFilename() throws Exception {
    // tab character is invalid
    invalidImportTest("\tfile.bzl", SkylarkImports.INVALID_PATH_SYNTAX);
  }

  @Test
  public void serialization() throws Exception {
    new SerializationTester(
            SkylarkImports.create("//some/skylark:file.bzl"),
            SkylarkImports.create(":subdirectory/containing/file.bzl"))
        .runTests();
  }
}
