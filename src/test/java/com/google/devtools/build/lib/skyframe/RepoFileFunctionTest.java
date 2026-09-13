// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;

import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.Type;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class RepoFileFunctionTest extends BuildViewTestCase {

  @Test
  public void defaultVisibility() throws Exception {
    scratch.overwriteFile("REPO.bazel", "repo(default_visibility=['//some:thing'])");
    scratch.overwriteFile("p/BUILD", "filegroup(name = 't')");
    invalidatePackages();
    Target t = getTarget("//p:t");
    assertThat(t.getVisibility().getDeclaredLabels())
        .containsExactly(Label.parseCanonical("//some:thing"));
  }

  @Test
  public void repoFileInTheMainRepo() throws Exception {
    scratch.overwriteFile("REPO.bazel", "repo(default_deprecation='EVERYTHING IS DEPRECATED')");
    scratch.overwriteFile("abc/def/BUILD", "filegroup(name='what')");
    invalidatePackages();
    assertThat(
            getRuleContext(getConfiguredTarget("//abc/def:what"))
                .attributes()
                .get("deprecation", Type.STRING))
        .isEqualTo("EVERYTHING IS DEPRECATED");
  }

  @Test
  public void repoFileInAnExternalRepo() throws Exception {
    scratch.overwriteFile("MODULE.bazel", "bazel_dep(name='foo',version='1.0')");
    scratch.overwriteFile("abc/def/BUILD", "filegroup(name='what')");
    registry.addModule(createModuleKey("foo", "1.0"), "module(name='foo',version='1.0')");
    scratch.overwriteFile(
        moduleRoot.getRelative("foo+1.0/REPO.bazel").getPathString(),
        "repo(default_deprecation='EVERYTHING IS DEPRECATED')");
    scratch.overwriteFile(
        moduleRoot.getRelative("foo+1.0/abc/def/BUILD").getPathString(), "filegroup(name='what')");

    invalidatePackages();

    assertThat(
            getRuleContext(getConfiguredTarget("//abc/def:what"))
                .attributes()
                .get("deprecation", Type.STRING))
        .isNull();
    assertThat(
            getRuleContext(getConfiguredTarget("@@foo+//abc/def:what"))
                .attributes()
                .get("deprecation", Type.STRING))
        .isEqualTo("EVERYTHING IS DEPRECATED");
  }

  @Test
  public void cantCallRepoTwice() throws Exception {
    scratch.overwriteFile(
        "REPO.bazel",
        "repo(default_deprecation='EVERYTHING IS DEPRECATED')",
        "repo(features=['abc'])");
    scratch.overwriteFile("abc/def/BUILD", "filegroup(name='what')");
    reporter.removeHandler(failFastHandler);
    try {
      invalidatePackages();
    } catch (
        @SuppressWarnings("InterruptedExceptionSwallowed")
        Exception e) {
      // Ignore any errors.
    }
    assertTargetError("//abc/def:what", "'repo' can only be called once");
  }

  @Test
  public void featureMerger() throws Exception {
    scratch.overwriteFile("REPO.bazel", "repo(features=['a', 'b', 'c', '-d'])");
    scratch.overwriteFile(
        "abc/def/BUILD",
        """
        package(features = [
            "-a",
            "-b",
            "d",
        ])

        filegroup(
            name = "what",
            features = ["b"],
        )
        """);
    invalidatePackages();
    RuleContext ruleContext = getRuleContext(getConfiguredTarget("//abc/def:what"));
    assertThat(ruleContext.getFeatures()).containsExactly("b", "c", "d");
    assertThat(ruleContext.getDisabledFeatures()).containsExactly("a");
  }

  @Test
  public void cantCallTraversalIgnoreDirectoriesTwice() throws Exception {
    assertTraversalIgnoreDirectoriesError(
        "traversal_ignore_directories(['a'])\ntraversal_ignore_directories(['b'])",
        "'traversal_ignore_directories()' can only be called once");
  }

  @Test
  public void traversalIgnoreDirectoriesMustFollowRepo() throws Exception {
    assertTraversalIgnoreDirectoriesError(
        "traversal_ignore_directories(['a'])\nrepo(features=['abc'])",
        "if repo() is called, it must be called before any other functions");
  }

  @Test
  public void traversalIgnoreDirectoriesRejectsPackageLabel() throws Exception {
    assertTraversalIgnoreDirectoriesError(
        "traversal_ignore_directories(['//experimental'])",
        "invalid traversal_ignore_directories entry '//experimental': path must be normalized"
            + " and relative to the workspace");
  }

  @Test
  public void traversalIgnoreDirectoriesRejectsAbsolutePath() throws Exception {
    assertTraversalIgnoreDirectoriesError(
        "traversal_ignore_directories(['/experimental'])",
        "invalid traversal_ignore_directories entry '/experimental': path must be normalized"
            + " and relative to the workspace");
  }

  @Test
  public void traversalIgnoreDirectoriesRejectsRootDirectory() throws Exception {
    assertTraversalIgnoreDirectoriesError(
        "traversal_ignore_directories([''])",
        "invalid traversal_ignore_directories entry '': ignores the entire workspace");
  }

  @Test
  public void traversalIgnoreDirectoriesRejectsDotDirectory() throws Exception {
    assertTraversalIgnoreDirectoriesError(
        "traversal_ignore_directories(['.'])",
        "invalid traversal_ignore_directories entry '.': ignores the entire workspace");
  }

  @Test
  public void traversalIgnoreDirectoriesRejectsUnnormalizedPath() throws Exception {
    assertTraversalIgnoreDirectoriesError(
        "traversal_ignore_directories(['experimental/./foo'])",
        "invalid traversal_ignore_directories entry 'experimental/./foo': path must be"
            + " normalized and relative to the workspace");
  }

  @Test
  public void traversalIgnoreDirectoriesRejectsTrailingSlash() throws Exception {
    assertTraversalIgnoreDirectoriesError(
        "traversal_ignore_directories(['experimental/'])",
        "invalid traversal_ignore_directories entry 'experimental/': path must be normalized"
            + " and relative to the workspace");
  }

  @Test
  public void traversalIgnoreDirectoriesRejectsUplevelReference() throws Exception {
    assertTraversalIgnoreDirectoriesError(
        "traversal_ignore_directories(['../experimental'])",
        "invalid traversal_ignore_directories entry '../experimental': path must be normalized"
            + " and relative to the workspace");
  }

  @Test
  public void traversalIgnoreDirectoriesRejectsTrailingUplevelReference() throws Exception {
    assertTraversalIgnoreDirectoriesError(
        "traversal_ignore_directories(['experimental/..'])",
        "invalid traversal_ignore_directories entry 'experimental/..': path must be normalized"
            + " and relative to the workspace");
  }

  @Test
  public void traversalIgnoreDirectoriesRejectsTargetPattern() throws Exception {
    assertTraversalIgnoreDirectoriesError(
        "traversal_ignore_directories(['experimental/...'])",
        "invalid traversal_ignore_directories entry 'experimental/...': package name component"
            + " contains only '.' characters");
  }

  @Test
  public void traversalIgnoreDirectoriesRejectsTarget() throws Exception {
    assertTraversalIgnoreDirectoriesError(
        "traversal_ignore_directories(['experimental:thing'])",
        "invalid traversal_ignore_directories entry 'experimental:thing': package names may"
            + " contain A-Z, a-z, 0-9, or any of ' !\"#$%&'()*+,-./;<=>?[]^_`{|}~' (any ASCII"
            + " character except 0-31, 127, ':', or '\\')");
  }

  @Test
  public void traversalIgnoreDirectoriesRejectsInvalidPackageName() throws Exception {
    // "\n" is a normalized relative path (a control character isn't a path separator or a
    // '.'/'..' segment), but control characters aren't allowed in a package name, so this is
    // caught by the LabelValidator check.
    assertTraversalIgnoreDirectoriesError(
        "traversal_ignore_directories(['foo\\nbar'])",
        "invalid traversal_ignore_directories entry 'foo\nbar': package names may contain"
            + " A-Z, a-z, 0-9, or any of ' !\"#$%&'()*+,-./;<=>?[]^_`{|}~' (any ASCII character"
            + " except 0-31, 127, ':', or '\\')");
  }

  private void assertTraversalIgnoreDirectoriesError(String repoFileContents, String expectedError)
      throws Exception {
    scratch.overwriteFile("REPO.bazel", repoFileContents);
    scratch.overwriteFile("abc/def/BUILD", "filegroup(name='what')");
    reporter.removeHandler(failFastHandler);
    try {
      invalidatePackages();
    } catch (
        @SuppressWarnings("InterruptedExceptionSwallowed")
        Exception e) {
      // Ignore any errors.
    }
    assertTargetError("//abc/def:what", expectedError);
  }

  @Test
  public void restrictedSyntax() throws Exception {
    scratch.overwriteFile(
        "REPO.bazel", "if 3+5>7: repo(default_deprecation='EVERYTHING IS DEPRECATED')");
    scratch.overwriteFile("abc/def/BUILD", "filegroup(name='what')");
    reporter.removeHandler(failFastHandler);
    try {
      invalidatePackages();
    } catch (
        @SuppressWarnings("InterruptedExceptionSwallowed")
        Exception e) {
      // Ignore any errors.
    }
    assertTargetError(
        "//abc/def:what",
        "`if` statements are not allowed in REPO.bazel files. You may use an `if` expression for"
            + " simple cases.");
  }
}
