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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;

import java.util.UUID;

/**
 * Tests for SkylarkImportLookupFunction.
 */
public class SkylarkImportLookupFunctionTest extends BuildViewTestCase {

  @Override
  public void setUp() throws Exception {
    super.setUp();
    Path alternativeRoot = scratch.dir("/root_2");
    getSkyframeExecutor()
        .preparePackageLoading(
            new PathPackageLocator(outputBase, ImmutableList.of(rootDirectory, alternativeRoot)),
            ConstantRuleVisibility.PUBLIC,
            true,
            7,
            "",
            UUID.randomUUID());
  }

  public void testSkylarkImportLabels() throws Exception {
    scratch.file("pkg1/BUILD");
    scratch.file("pkg1/ext.bzl");
    checkLabel("//pkg1:ext.bzl", "//pkg1:ext.bzl");

    scratch.file("pkg2/BUILD");
    scratch.file("pkg2/dir/ext.bzl");
    checkLabel("//pkg2:dir/ext.bzl", "//pkg2:dir/ext.bzl");

    scratch.file("dir/pkg3/BUILD");
    scratch.file("dir/pkg3/dir/ext.bzl");
    checkLabel("//dir/pkg3:dir/ext.bzl", "//dir/pkg3:dir/ext.bzl");
  }

  public void testSkylarkImportLabelsAlternativeRoot() throws Exception {
    scratch.file("/root_2/pkg4/BUILD");
    scratch.file("/root_2/pkg4/ext.bzl");
    checkLabel("//pkg4:ext.bzl", "//pkg4:ext.bzl");
  }

  public void testSkylarkImportLabelsMultipleBuildFiles() throws Exception {
    scratch.file("dir1/BUILD");
    scratch.file("dir1/dir2/BUILD");
    scratch.file("dir1/dir2/ext.bzl");
    checkLabel("//dir1/dir2:ext.bzl", "//dir1/dir2:ext.bzl");
  }

  public void testLoadRelativePath() throws Exception {
    scratch.file("pkg/BUILD");
    scratch.file("pkg/ext1.bzl", "a = 1");
    scratch.file("pkg/ext2.bzl", "load('ext1', 'a')");
    get(key("//pkg:ext2.bzl"));
  }

  public void testLoadAbsolutePath() throws Exception {
    scratch.file("pkg2/BUILD");
    scratch.file("pkg3/BUILD");
    scratch.file("pkg2/ext.bzl", "b = 1");
    scratch.file("pkg3/ext.bzl", "load('/pkg2/ext', 'b')");
    get(key("//pkg3:ext.bzl"));
  }

  private EvaluationResult<SkylarkImportLookupValue> get(SkyKey skylarkImportLookupKey)
      throws Exception {
    EvaluationResult<SkylarkImportLookupValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skylarkImportLookupKey, /*keepGoing=*/ false, reporter);
    if (result.hasError()) {
      fail(result.getError(skylarkImportLookupKey).getException().getMessage());
    }
    return result;
  }

  private SkyKey key(String label) throws Exception {
    return SkylarkImportLookupValue.key(Label.parseAbsoluteUnchecked(label));
  }

  private void checkLabel(String file, String label) throws Exception {
    SkyKey skylarkImportLookupKey = key(file);
    EvaluationResult<SkylarkImportLookupValue> result = get(skylarkImportLookupKey);
    assertEquals(label, result.get(skylarkImportLookupKey).getDependency().getLabel().toString());
  }

  public void testSkylarkImportLookupNoBuildFile() throws Exception {
    scratch.file("pkg/ext.bzl", "");
    SkyKey skylarkImportLookupKey =
        SkylarkImportLookupValue.key(Label.parseAbsoluteUnchecked("//pkg:ext.bzl"));
    EvaluationResult<SkylarkImportLookupValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skylarkImportLookupKey, /*keepGoing=*/ false, reporter);
    assertTrue(result.hasError());
    ErrorInfo errorInfo = result.getError(skylarkImportLookupKey);
    String errorMessage = errorInfo.getException().getMessage();
    assertEquals("Extension file not found. Unable to load package for '//pkg:ext.bzl': "
        + "BUILD file not found on package path", errorMessage);
  }

  public void testSkylarkAbsoluteImportFilenameWithControlChars() throws Exception {
    scratch.file("pkg/BUILD", "");
    scratch.file("pkg/ext.bzl", "load('/pkg/oops\u0000', 'a')");
    SkyKey skylarkImportLookupKey =
        SkylarkImportLookupValue.key(Label.parseAbsoluteUnchecked("//pkg:ext.bzl"));
    EvaluationResult<SkylarkImportLookupValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skylarkImportLookupKey, /*keepGoing=*/ false, reporter);
    assertTrue(result.hasError());
    ErrorInfo errorInfo = result.getError(skylarkImportLookupKey);
    String errorMessage = errorInfo.getException().getMessage();
    assertEquals("invalid target name 'oops<?>.bzl': "
        + "target names may not contain non-printable characters: '\\x00'", errorMessage);
  }

  public void testSkylarkRelativeImportFilenameWithControlChars() throws Exception {
    scratch.file("pkg/BUILD", "");
    scratch.file("pkg/ext.bzl", "load('oops\u0000', 'a')");
    SkyKey skylarkImportLookupKey =
        SkylarkImportLookupValue.key(Label.parseAbsoluteUnchecked("//pkg:ext.bzl"));
    EvaluationResult<SkylarkImportLookupValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skylarkImportLookupKey, /*keepGoing=*/ false, reporter);
    assertTrue(result.hasError());
    ErrorInfo errorInfo = result.getError(skylarkImportLookupKey);
    String errorMessage = errorInfo.getException().getMessage();
    assertEquals("invalid target name 'oops<?>.bzl': "
        + "target names may not contain non-printable characters: '\\x00'", errorMessage);
  }

}
