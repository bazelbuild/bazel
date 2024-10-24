// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Functions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.io.FileSymlinkInfiniteExpansionException;
import com.google.devtools.build.lib.packages.Globber;
import com.google.devtools.build.lib.packages.Globber.Operation;
import com.google.devtools.build.lib.skyframe.GlobsValue.GlobRequest;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public final class GlobsFunctionTest extends GlobTestBase {
  @Override
  protected void createGlobSkyFunction(Map<SkyFunctionName, SkyFunction> skyFunctions) {
    skyFunctions.put(SkyFunctions.GLOBS, new GlobsFunction());
  }

  @Override
  protected void assertSingleGlobMatches(
      String pattern, Globber.Operation globberOperation, String... expecteds) throws Exception {
    ImmutableSet<PathFragment> matchesInPathFragment =
        runSingleGlob(pattern, globberOperation).getMatches();
    assertThat(
            matchesInPathFragment.stream()
                .map(PathFragment::getPathString)
                .collect(toImmutableSet()))
        .isEqualTo(ImmutableSet.copyOf(expecteds));
  }

  @Override
  protected GlobsValue runSingleGlob(String pattern, Globber.Operation globberOperation)
      throws Exception {
    GlobRequest globRequest = GlobRequest.create(pattern, globberOperation);
    return queryGlobsValue(
        GlobsValue.key(PKG_ID, Root.fromPath(root), ImmutableSet.of(globRequest)));
  }

  private GlobsValue queryGlobsValue(GlobsValue.Key globsKey) throws Exception {
    EvaluationResult<SkyValue> result =
        evaluator.evaluate(ImmutableList.of(globsKey), EVALUATION_OPTIONS);
    if (result.hasError()) {
      throw result.getError().getException();
    }

    SkyValue skyValue = result.get(globsKey);
    assertThat(skyValue).isInstanceOf(GlobsValue.class);
    return (GlobsValue) skyValue;
  }

  @Override
  protected void assertIllegalPattern(String pattern) {
    assertThrows(
        "invalid pattern not detected: " + pattern,
        InvalidGlobPatternException.class,
        () -> GlobsValue.GlobRequest.create(pattern, Globber.Operation.FILES_AND_DIRS));
  }

  @Override
  protected GlobsValue.Key createdGlobRelatedSkyKey(
      String pattern, Globber.Operation globberOperation) throws InvalidGlobPatternException {
    return GlobsValue.key(
        PKG_ID,
        Root.fromPath(root),
        ImmutableSet.of(GlobRequest.create(pattern, globberOperation)));
  }

  @Override
  protected Iterable<String> getSubpackagesMatches(String pattern) throws Exception {
    SkyValue skyValue = runSingleGlob(pattern, Globber.Operation.SUBPACKAGES);
    assertThat(skyValue).isInstanceOf(GlobsValue.class);
    return Iterables.transform(((GlobsValue) skyValue).getMatches(), Functions.toStringFunction());
  }

  // The test cases below cover scenario when there are multiple GlobRequests defined.
  @Test
  public void testGlobs_allGlobRequestsAllSucceeds() throws Exception {
    GlobRequest globRequest1 = GlobRequest.create("foo/barnacle/**", Operation.FILES_AND_DIRS);
    GlobRequest globRequest2 = GlobRequest.create("foo/bar/**", Operation.FILES);
    GlobRequest globRequest3 = GlobRequest.create("a2/**", Operation.SUBPACKAGES);

    GlobsValue globsValue =
        queryGlobsValue(
            GlobsValue.key(
                PKG_ID,
                Root.fromPath(root),
                ImmutableSet.of(globRequest1, globRequest2, globRequest3)));
    assertThat(
            globsValue.getMatches().stream()
                .map(PathFragment::getPathString)
                .collect(toImmutableSet()))
        .containsExactly("a2/b2", "foo/barnacle", "foo/barnacle/wiz", "foo/bar/wiz/file");
  }

  @Test
  public void testGlobs_oneGoodAndOneGlobUnboundedSymlink() throws Exception {
    pkgPath.getRelative("parent/sub").createDirectoryAndParents();
    FileSystemUtils.ensureSymbolicLink(
        pkgPath.getRelative("parent/sub/symlink"), pkgPath.getRelative("parent"));

    GlobRequest unboundedSymlinksGlobRequest =
        GlobRequest.create("parent/sub/*", Operation.FILES_AND_DIRS);
    GlobRequest goodGlobRequest = GlobRequest.create("foo/bar/**", Operation.FILES);

    GlobsValue.Key globsKey =
        GlobsValue.key(
            PKG_ID,
            Root.fromPath(root),
            ImmutableSet.of(unboundedSymlinksGlobRequest, goodGlobRequest));
    EvaluationResult<GlobValue> result =
        evaluator.evaluate(ImmutableList.of(globsKey), EVALUATION_OPTIONS);

    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(globsKey);
    assertThat(errorInfo.getException()).isInstanceOf(FileSymlinkInfiniteExpansionException.class);
    assertThat(errorInfo.getException()).hasMessageThat().contains("Infinite symlink expansion");
  }

  @Test
  public void testGlobs_bothTwoGlobBothAreSymlinkCycles() throws Exception {
    pkgPath.getRelative("parent/sub1").createDirectoryAndParents();
    FileSystemUtils.ensureSymbolicLink(
        pkgPath.getRelative("parent/sub1/self1"), pkgPath.getRelative("parent"));
    GlobRequest symlinkGlobRequest1 = GlobRequest.create("parent/sub1/*", Operation.FILES_AND_DIRS);

    pkgPath.getRelative("parent/sub2").createDirectoryAndParents();
    FileSystemUtils.ensureSymbolicLink(
        pkgPath.getRelative("parent/sub2/self2"), pkgPath.getRelative("parent"));
    GlobRequest symlinkGlobRequest2 = GlobRequest.create("parent/sub2/*", Operation.FILES_AND_DIRS);

    GlobsValue.Key globsKey =
        GlobsValue.key(
            PKG_ID, Root.fromPath(root), ImmutableSet.of(symlinkGlobRequest1, symlinkGlobRequest2));
    EvaluationResult<GlobValue> result =
        evaluator.evaluate(ImmutableList.of(globsKey), EVALUATION_OPTIONS);

    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(globsKey);
    assertThat(errorInfo.getException()).isInstanceOf(FileSymlinkInfiniteExpansionException.class);
    assertThat(errorInfo.getException()).hasMessageThat().contains("Infinite symlink expansion");

    // The two globs are evaluated in parallel inside GlobsFunction, so it is non-deterministic
    // which SymlinkInfiniteExpansionException is thrown and caught first. So this test only needs
    // to verify the output error is from either one of the SymlinkInfiniteExpansion errors.
    assertThat(((FileSymlinkInfiniteExpansionException) errorInfo.getException()).getChain())
        .containsAnyOf(
            RootedPath.toRootedPath(Root.fromPath(root), pkgPath.getRelative("parent/sub1/self1")),
            RootedPath.toRootedPath(Root.fromPath(root), pkgPath.getRelative("parent/sub2/self2")));
  }
}
