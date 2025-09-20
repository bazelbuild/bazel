// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackagePiece;
import com.google.devtools.build.lib.packages.PackagePieceIdentifier;
import com.google.devtools.build.lib.skyframe.GraphBackedRecursivePackageProvider.UniverseTargetPattern;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link GraphBackedRecursivePackageProvider}. */
@RunWith(TestParameterInjector.class)
public final class GraphBackedRecursivePackageProviderTest extends BuildViewTestCase {

  private WalkableGraph makeWalkableGraph(SkyKey... roots) throws InterruptedException {
    EvaluationResult<?> result =
        getSkyframeExecutor()
            .evaluate(
                ImmutableList.copyOf(roots),
                /* keepGoing= */ true,
                /* numThreads= */ SkyframeExecutor.DEFAULT_THREAD_COUNT,
                reporter);
    return result.getWalkableGraph();
  }

  private GraphBackedRecursivePackageProvider makeGraphBackedRecursivePackageProvider(
      WalkableGraph walkableGraph) throws InterruptedException {
    return new GraphBackedRecursivePackageProvider(
        walkableGraph,
        UniverseTargetPattern.all(),
        getSkyframeExecutor().getPackageManager().getPackagePath(),
        new RecursivePkgValueRootPackageExtractor());
  }

  @Test
  public void getBuildFile_eagerMacroExpansion() throws Exception {
    scratch.file("pkg1/BUILD", "java_library(name = 'foo')");

    PackageIdentifier pkgId = PackageIdentifier.createInMainRepo("pkg1");
    GraphBackedRecursivePackageProvider packageProvider =
        makeGraphBackedRecursivePackageProvider(makeWalkableGraph(pkgId));
    InputFile buildFile =
        packageProvider.getBuildFile(reporter, PackageIdentifier.createInMainRepo("pkg1"));
    assertThat(buildFile.getName()).isEqualTo("BUILD");
    assertThat(buildFile.getPackageoid()).isInstanceOf(Package.class);
  }

  @Test
  public void getBuildFile_lazyMacroExpansion(@TestParameter boolean graphContainsFullPackage)
      throws Exception {
    setPackageOptions("--experimental_lazy_macro_expansion_packages=*");
    scratch.file("pkg1/BUILD", "java_library(name = 'foo')");

    PackageIdentifier pkgId = PackageIdentifier.createInMainRepo("pkg1");
    PackagePieceIdentifier.ForBuildFile packagePieceId =
        new PackagePieceIdentifier.ForBuildFile(pkgId);
    WalkableGraph graph = makeWalkableGraph(graphContainsFullPackage ? pkgId : packagePieceId);
    if (graphContainsFullPackage) {
      assertThat(graph.getValue(pkgId)).isNotNull();
    } else {
      assertThat(graph.getValue(pkgId)).isNull();
    }
    GraphBackedRecursivePackageProvider packageProvider =
        makeGraphBackedRecursivePackageProvider(graph);
    InputFile buildFile =
        packageProvider.getBuildFile(reporter, PackageIdentifier.createInMainRepo("pkg1"));
    assertThat(buildFile.getName()).isEqualTo("BUILD");
    assertThat(buildFile.getPackageoid()).isInstanceOf(PackagePiece.ForBuildFile.class);
  }
}
