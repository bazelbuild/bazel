// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.ideinfo;

import static com.google.common.collect.Iterables.transform;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import com.google.common.collect.Iterables;
import com.google.common.collect.ObjectArrays;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.intellij.ideinfo.IntellijIdeInfo.ArtifactLocation;
import com.google.devtools.intellij.ideinfo.IntellijIdeInfo.CIdeInfo;
import com.google.devtools.intellij.ideinfo.IntellijIdeInfo.CToolchainIdeInfo;
import com.google.devtools.intellij.ideinfo.IntellijIdeInfo.JavaIdeInfo;
import com.google.devtools.intellij.ideinfo.IntellijIdeInfo.TargetIdeInfo;
import com.google.protobuf.ByteString;
import com.google.protobuf.ProtocolStringList;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AndroidStudioInfoAspect} validating proto's contents. */
@RunWith(JUnit4.class)
public class AndroidStudioInfoAspectTest extends AndroidStudioInfoAspectTestBase {

  @Override
  protected final void useConfiguration(String... args) throws Exception {
    super.useConfiguration(ObjectArrays.concat(args, "--java_header_compilation=true"));
  }

  @Test
  public void testSimpleJavaLibrary() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "    name = 'simple',",
        "    srcs = ['simple/Simple.java']",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:simple");
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);
    ArtifactLocation location = targetIdeInfo.getBuildFileArtifactLocation();
    assertThat(Paths.get(location.getRelativePath()).toString())
        .isEqualTo(Paths.get("com/google/example/BUILD").toString());
    assertThat(location.getIsSource()).isTrue();
    assertThat(location.getIsExternal()).isFalse();
    assertThat(targetIdeInfo.getKindString()).isEqualTo("java_library");
    assertThat(relativePathsForJavaSourcesOf(targetIdeInfo))
        .containsExactly("com/google/example/simple/Simple.java");
    assertThat(transform(targetIdeInfo.getJavaIdeInfo().getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(
            jarString(
                "com/google/example", "libsimple.jar", "libsimple-hjar.jar", "libsimple-src.jar"));

    assertThat(getIdeResolveFiles())
        .containsExactly(
            "com/google/example/libsimple.jar",
            "com/google/example/libsimple-hjar.jar",
            "com/google/example/libsimple-src.jar");
    assertThat(targetIdeInfo.getJavaIdeInfo().getJdeps().getRelativePath())
        .isEqualTo("com/google/example/libsimple.jdeps");
  }

  @Test
  public void testPackageManifestCreated() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "    name = 'simple',",
        "    srcs = ['simple/Simple.java']",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:simple");
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);

    ArtifactLocation packageManifest = targetIdeInfo.getJavaIdeInfo().getPackageManifest();
    assertNotNull(packageManifest);

    assertEquals(packageManifest.getRelativePath(), "com/google/example/simple.manifest");
  }

  @Test
  public void testPackageManifestNotCreatedForOnlyGeneratedSources() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "genrule(",
        "   name = 'gen_sources',",
        "   outs = ['Gen.java'],",
        "   cmd = '',",
        ")",
        "java_library(",
        "    name = 'simple',",
        "    srcs = [':gen_sources']",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:simple");
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);
    assertThat(targetIdeInfo.getJavaIdeInfo().hasPackageManifest()).isFalse();
  }

  @Test
  public void testFilteredGenJarNotCreatedForSourceOnlyRule() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "    name = 'simple',",
        "    srcs = ['Test.java']",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:simple");
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);
    assertThat(targetIdeInfo.getJavaIdeInfo().hasFilteredGenJar()).isFalse();
  }

  @Test
  public void testFilteredGenJarNotCreatedForOnlyGenRule() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "genrule(",
        "   name = 'gen_sources',",
        "   outs = ['Gen.java'],",
        "   cmd = '',",
        ")",
        "java_library(",
        "    name = 'simple',",
        "    srcs = [':gen_sources']",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:simple");
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);
    assertThat(targetIdeInfo.getJavaIdeInfo().hasFilteredGenJar()).isFalse();
  }

  @Test
  public void testFilteredGenJar() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "genrule(",
        "   name = 'gen_sources',",
        "   outs = ['Gen.java'],",
        "   cmd = '',",
        ")",
        "genrule(",
        "   name = 'gen_srcjar',",
        "   outs = ['gen.srcjar'],",
        "   cmd = '',",
        ")",
        "java_library(",
        "    name = 'lib',",
        "    srcs = [':gen_sources', ':gen_srcjar', 'Test.java']",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:lib");
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:lib", targetIdeInfos);
    assertThat(targetIdeInfo.getJavaIdeInfo().hasFilteredGenJar()).isTrue();
    assertThat(targetIdeInfo.getJavaIdeInfo().getFilteredGenJar().getJar().getRelativePath())
        .isEqualTo("com/google/example/lib-filtered-gen.jar");
    assertThat(targetIdeInfo.getJavaIdeInfo().getFilteredGenJar().getSourceJar().getRelativePath())
        .isEqualTo("com/google/example/lib-filtered-gen-src.jar");
  }

  @Test
  public void testJavaLibraryWithDependencies() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "    name = 'simple',",
        "    srcs = ['simple/Simple.java']",
        ")",
        "java_library(",
        "    name = 'complex',",
        "    srcs = ['complex/Complex.java'],",
        "    deps = [':simple']",
        ")");

    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:complex");

    getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);
    TargetIdeInfo complexTarget =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:complex", targetIdeInfos);

    assertThat(relativePathsForJavaSourcesOf(complexTarget))
        .containsExactly("com/google/example/complex/Complex.java");
    assertThat(complexTarget.getDependenciesList()).contains("//com/google/example:simple");
  }

  @Test
  public void testJavaLibraryWithTransitiveDependencies() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "    name = 'simple',",
        "    srcs = ['simple/Simple.java']",
        ")",
        "java_library(",
        "    name = 'complex',",
        "    srcs = ['complex/Complex.java'],",
        "    deps = [':simple']",
        ")",
        "java_library(",
        "    name = 'extracomplex',",
        "    srcs = ['extracomplex/ExtraComplex.java'],",
        "    deps = [':complex']",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:extracomplex");

    getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);
    getTargetIdeInfoAndVerifyLabel("//com/google/example:complex", targetIdeInfos);

    TargetIdeInfo extraComplexTarget =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:extracomplex", targetIdeInfos);

    assertThat(relativePathsForJavaSourcesOf(extraComplexTarget))
        .containsExactly("com/google/example/extracomplex/ExtraComplex.java");
    assertThat(extraComplexTarget.getDependenciesList()).contains("//com/google/example:complex");

    assertThat(getIdeResolveFiles())
        .containsExactly(
            "com/google/example/libextracomplex.jar",
            "com/google/example/libextracomplex-hjar.jar",
            "com/google/example/libextracomplex-src.jar",
            "com/google/example/libcomplex.jar",
            "com/google/example/libcomplex-hjar.jar",
            "com/google/example/libcomplex-src.jar",
            "com/google/example/libsimple.jar",
            "com/google/example/libsimple-hjar.jar",
            "com/google/example/libsimple-src.jar");
  }

  @Test
  public void testJavaLibraryWithDiamondDependencies() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "    name = 'simple',",
        "    srcs = ['simple/Simple.java']",
        ")",
        "java_library(",
        "    name = 'complex',",
        "    srcs = ['complex/Complex.java'],",
        "    deps = [':simple']",
        ")",
        "java_library(",
        "    name = 'complex1',",
        "    srcs = ['complex1/Complex.java'],",
        "    deps = [':simple']",
        ")",
        "java_library(",
        "    name = 'extracomplex',",
        "    srcs = ['extracomplex/ExtraComplex.java'],",
        "    deps = [':complex', ':complex1']",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:extracomplex");

    getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);
    getTargetIdeInfoAndVerifyLabel("//com/google/example:complex", targetIdeInfos);
    getTargetIdeInfoAndVerifyLabel("//com/google/example:complex1", targetIdeInfos);

    TargetIdeInfo extraComplexTarget =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:extracomplex", targetIdeInfos);

    assertThat(relativePathsForJavaSourcesOf(extraComplexTarget))
        .containsExactly("com/google/example/extracomplex/ExtraComplex.java");
    assertThat(extraComplexTarget.getDependenciesList())
        .containsAllOf("//com/google/example:complex", "//com/google/example:complex1");
  }

  @Test
  public void testJavaLibraryWithExports() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "    name = 'simple',",
        "    srcs = ['simple/Simple.java']",
        ")",
        "java_library(",
        "    name = 'complex',",
        "    srcs = ['complex/Complex.java'],",
        "    exports = [':simple'],",
        ")",
        "java_library(",
        "    name = 'extracomplex',",
        "    srcs = ['extracomplex/ExtraComplex.java'],",
        "    deps = [':complex']",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:extracomplex");

    getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);
    getTargetIdeInfoAndVerifyLabel("//com/google/example:complex", targetIdeInfos);

    TargetIdeInfo complexTarget =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:complex", targetIdeInfos);
    TargetIdeInfo extraComplexTarget =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:extracomplex", targetIdeInfos);

    assertThat(complexTarget.getDependenciesList()).contains("//com/google/example:simple");

    assertThat(extraComplexTarget.getDependenciesList())
        .containsAllOf("//com/google/example:simple", "//com/google/example:complex");
    assertThat(getIdeResolveFiles())
        .containsExactly(
            "com/google/example/libextracomplex.jar",
            "com/google/example/libextracomplex-hjar.jar",
            "com/google/example/libextracomplex-src.jar",
            "com/google/example/libcomplex.jar",
            "com/google/example/libcomplex-hjar.jar",
            "com/google/example/libcomplex-src.jar",
            "com/google/example/libsimple.jar",
            "com/google/example/libsimple-hjar.jar",
            "com/google/example/libsimple-src.jar");
  }

  @Test
  public void testJavaLibraryWithTransitiveExports() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "    name = 'simple',",
        "    srcs = ['simple/Simple.java']",
        ")",
        "java_library(",
        "    name = 'complex',",
        "    srcs = ['complex/Complex.java'],",
        "    exports = [':simple'],",
        ")",
        "java_library(",
        "    name = 'extracomplex',",
        "    srcs = ['extracomplex/ExtraComplex.java'],",
        "    exports = [':complex'],",
        ")",
        "java_library(",
        "    name = 'megacomplex',",
        "    srcs = ['megacomplex/MegaComplex.java'],",
        "    deps = [':extracomplex'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:megacomplex");

    getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);
    getTargetIdeInfoAndVerifyLabel("//com/google/example:complex", targetIdeInfos);
    getTargetIdeInfoAndVerifyLabel("//com/google/example:extracomplex", targetIdeInfos);

    TargetIdeInfo megaComplexTarget =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:megacomplex", targetIdeInfos);

    assertThat(relativePathsForJavaSourcesOf(megaComplexTarget))
        .containsExactly("com/google/example/megacomplex/MegaComplex.java");
    assertThat(megaComplexTarget.getDependenciesList())
        .containsAllOf(
            "//com/google/example:simple",
            "//com/google/example:complex",
            "//com/google/example:extracomplex");
  }

  @Test
  public void testJavaImport() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_import(",
        "   name = 'imp',",
        "   jars = ['a.jar', 'b.jar'],",
        "   srcjar = 'impsrc.jar',",
        ")",
        "java_library(",
        "   name = 'lib',",
        "   srcs = ['Lib.java'],",
        "   deps = [':imp'],",
        ")");

    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:lib");
    final TargetIdeInfo libInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:lib", targetIdeInfos);
    TargetIdeInfo impInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:imp", targetIdeInfos);

    assertThat(impInfo.getKindString()).isEqualTo("java_import");
    assertThat(libInfo.getDependenciesList()).contains("//com/google/example:imp");

    JavaIdeInfo javaIdeInfo = impInfo.getJavaIdeInfo();
    assertThat(javaIdeInfo).isNotNull();
    assertThat(transform(javaIdeInfo.getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(
            jarString(
                "com/google/example",
                "a.jar",
                "_ijar/imp/com/google/example/a-ijar.jar",
                "impsrc.jar"),
            jarString(
                "com/google/example",
                "b.jar",
                "_ijar/imp/com/google/example/b-ijar.jar",
                "impsrc.jar"))
        .inOrder();

    assertThat(getIdeResolveFiles())
        .containsExactly(
            "com/google/example/_ijar/imp/com/google/example/a-ijar.jar",
            "com/google/example/_ijar/imp/com/google/example/b-ijar.jar",
            "com/google/example/liblib.jar",
            "com/google/example/liblib-hjar.jar",
            "com/google/example/liblib-src.jar");
  }

  @Test
  public void testJavaImportWithExports() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "   name = 'foobar',",
        "   srcs = ['FooBar.java'],",
        ")",
        "java_import(",
        "   name = 'imp',",
        "   jars = ['a.jar', 'b.jar'],",
        "   deps = [':foobar'],",
        "   exports = [':foobar'],",
        ")",
        "java_library(",
        "   name = 'lib',",
        "   srcs = ['Lib.java'],",
        "   deps = [':imp'],",
        ")");

    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:lib");
    TargetIdeInfo libInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:lib", targetIdeInfos);
    TargetIdeInfo impInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:imp", targetIdeInfos);

    assertThat(impInfo.getKindString()).isEqualTo("java_import");
    assertThat(impInfo.getDependenciesList()).contains("//com/google/example:foobar");
    assertThat(libInfo.getDependenciesList())
        .containsAllOf("//com/google/example:foobar", "//com/google/example:imp");
  }

  @Test
  public void testNoPackageManifestForExports() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "   name = 'foobar',",
        "   srcs = ['FooBar.java'],",
        ")",
        "java_import(",
        "   name = 'imp',",
        "   jars = ['a.jar', 'b.jar'],",
        "   deps = [':foobar'],",
        "   exports = [':foobar'],",
        ")",
        "java_library(",
        "   name = 'lib',",
        "   srcs = ['Lib.java'],",
        "   deps = [':imp'],",
        ")");

    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:lib");
    TargetIdeInfo libInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:lib", targetIdeInfos);
    TargetIdeInfo impInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:imp", targetIdeInfos);

    assertThat(!impInfo.getJavaIdeInfo().hasPackageManifest()).isTrue();
    assertThat(libInfo.getJavaIdeInfo().hasPackageManifest()).isTrue();
  }

  @Test
  public void testGeneratedJavaImportFilesAreAddedToOutputGroup() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_import(",
        "   name = 'imp',",
        "   jars = [':gen_jar'],",
        "   srcjar = ':gen_srcjar',",
        ")",
        "genrule(",
        "   name = 'gen_jar',",
        "   outs = ['gen_jar.jar'],",
        "   cmd = '',",
        ")",
        "genrule(",
        "   name = 'gen_srcjar',",
        "   outs = ['gen_srcjar.jar'],",
        "   cmd = '',",
        ")");
    buildIdeInfo("//com/google/example:imp");
    assertThat(getIdeResolveFiles())
        .containsExactly(
            "com/google/example/_ijar/imp/com/google/example/gen_jar-ijar.jar",
            "com/google/example/gen_jar.jar",
            "com/google/example/gen_srcjar.jar");
  }

  @Test
  public void testAspectIsPropagatedAcrossExports() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "   name = 'foobar',",
        "   srcs = ['FooBar.java'],",
        ")",
        "java_library(",
        "   name = 'lib',",
        "   srcs = ['Lib.java'],",
        "   exports = [':foobar'],",
        ")");

    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:lib");
    getTargetIdeInfoAndVerifyLabel("//com/google/example:foobar", targetIdeInfos);
  }

  @Test
  public void testJavaTest() throws Exception {
    scratch.file(
        "java/com/google/example/BUILD",
        "java_library(",
        "   name = 'foobar',",
        "   srcs = ['FooBar.java'],",
        ")",
        "java_test(",
        "   name = 'FooBarTest',",
        "   srcs = ['FooBarTest.java'],",
        "   size = 'large',",
        "   deps = [':foobar'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos =
        buildIdeInfo("//java/com/google/example:FooBarTest");
    TargetIdeInfo testInfo =
        getTargetIdeInfoAndVerifyLabel("//java/com/google/example:FooBarTest", targetIdeInfos);
    assertThat(testInfo.getKindString()).isEqualTo("java_test");
    assertThat(relativePathsForJavaSourcesOf(testInfo))
        .containsExactly("java/com/google/example/FooBarTest.java");
    assertThat(testInfo.getDependenciesList()).contains("//java/com/google/example:foobar");
    assertThat(transform(testInfo.getJavaIdeInfo().getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(
            jarString("java/com/google/example", "FooBarTest.jar", null, "FooBarTest-src.jar"));

    assertThat(getIdeResolveFiles())
        .containsExactly(
            "java/com/google/example/libfoobar.jar",
            "java/com/google/example/libfoobar-hjar.jar",
            "java/com/google/example/libfoobar-src.jar",
            "java/com/google/example/FooBarTest.jar",
            "java/com/google/example/FooBarTest-src.jar");
    assertThat(testInfo.getJavaIdeInfo().getJdeps().getRelativePath())
        .isEqualTo("java/com/google/example/FooBarTest.jdeps");

    assertThat(testInfo.getTestInfo().getSize()).isEqualTo("large");
  }

  @Test
  public void testJavaBinary() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "   name = 'foobar',",
        "   srcs = ['FooBar.java'],",
        ")",
        "java_binary(",
        "   name = 'foobar-exe',",
        "   main_class = 'MyMainClass',",
        "   srcs = ['FooBarMain.java'],",
        "   deps = [':foobar'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:foobar-exe");
    TargetIdeInfo binaryInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:foobar-exe", targetIdeInfos);

    assertThat(binaryInfo.getKindString()).isEqualTo("java_binary");
    assertThat(relativePathsForJavaSourcesOf(binaryInfo))
        .containsExactly("com/google/example/FooBarMain.java");
    assertThat(binaryInfo.getDependenciesList()).contains("//com/google/example:foobar");

    assertThat(transform(binaryInfo.getJavaIdeInfo().getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(
            jarString("com/google/example", "foobar-exe.jar", null, "foobar-exe-src.jar"));

    assertThat(getIdeResolveFiles())
        .containsExactly(
            "com/google/example/libfoobar.jar",
            "com/google/example/libfoobar-hjar.jar",
            "com/google/example/libfoobar-src.jar",
            "com/google/example/foobar-exe.jar",
            "com/google/example/foobar-exe-src.jar");
    assertThat(binaryInfo.getJavaIdeInfo().getJdeps().getRelativePath())
        .isEqualTo("com/google/example/foobar-exe.jdeps");
  }

  @Test
  public void testJavaToolchain() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "    name = 'a',",
        "    srcs = ['A.java'],",
        "    deps = [':b'],",
        ")",
        "java_library(",
        "    name = 'b',",
        "    srcs = ['B.java'],",
        ")");

    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:a");

    List<TargetIdeInfo> javaToolChainInfos = findJavaToolchain(targetIdeInfos);
    assertThat(javaToolChainInfos).hasSize(1); // Ensure we don't get one instance per java_library
    TargetIdeInfo toolChainInfo = Iterables.getOnlyElement(javaToolChainInfos);
    assertThat(toolChainInfo.getJavaToolchainIdeInfo().getSourceVersion()).isNotEmpty();
    assertThat(toolChainInfo.getJavaToolchainIdeInfo().getTargetVersion()).isNotEmpty();

    TargetIdeInfo a = targetIdeInfos.get("//com/google/example:a");
    assertThat(a.getDependenciesList())
        .containsAllOf("//com/google/example:b", toolChainInfo.getLabel());
  }

  @Test
  public void testJavaToolchainForAndroid() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "android_library(",
        "    name = 'a',",
        "    srcs = ['A.java'],",
        ")");

    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:a");
    assertThat(targetIdeInfos).hasSize(2);

    List<TargetIdeInfo> javaToolChainInfos = findJavaToolchain(targetIdeInfos);
    assertThat(javaToolChainInfos).hasSize(1);
  }

  @Test
  public void testAndroidLibrary() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "android_library(",
        "  name = 'l1',",
        "  manifest = 'AndroidManifest.xml',",
        "  custom_package = 'com.google.example',",
        "  resource_files = ['r1/values/a.xml'],",
        ")",
        "android_library(",
        "  name = 'l',",
        "  srcs = ['Main.java'],",
        "  deps = [':l1'],",
        "  manifest = 'AndroidManifest.xml',",
        "  custom_package = 'com.google.example',",
        "  resource_files = ['res/drawable/a.png', 'res/drawable/b.png'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:l");
    TargetIdeInfo target = getTargetIdeInfoAndVerifyLabel("//com/google/example:l", targetIdeInfos);
    assertThat(target.getKindString()).isEqualTo("android_library");
    assertThat(relativePathsForJavaSourcesOf(target))
        .containsExactly("com/google/example/Main.java");
    assertThat(transform(target.getJavaIdeInfo().getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(
            jarString("com/google/example", "libl.jar", "libl-hjar.jar", "libl-src.jar"),
            jarString("com/google/example", "l_resources.jar", null, "l_resources-src.jar"));
    assertThat(transform(target.getAndroidIdeInfo().getResourcesList(), ARTIFACT_TO_RELATIVE_PATH))
        .containsExactly("com/google/example/res");
    assertThat(target.getAndroidIdeInfo().getManifest().getRelativePath())
        .isEqualTo("com/google/example/AndroidManifest.xml");
    assertThat(target.getAndroidIdeInfo().getJavaPackage()).isEqualTo("com.google.example");
    assertThat(LIBRARY_ARTIFACT_TO_STRING.apply(target.getAndroidIdeInfo().getResourceJar()))
        .isEqualTo(jarString("com/google/example", "l_resources.jar", null, "l_resources-src.jar"));

    assertThat(target.getDependenciesList()).contains("//com/google/example:l1");
    assertThat(getIdeResolveFiles())
        .containsExactly(
            "com/google/example/libl.jar",
            "com/google/example/libl-hjar.jar",
            "com/google/example/libl-src.jar",
            "com/google/example/l_resources.jar",
            "com/google/example/l_resources-src.jar",
            "com/google/example/libl1.jar",
            "com/google/example/libl1-src.jar",
            "com/google/example/l1_resources.jar",
            "com/google/example/l1_resources-src.jar");
    assertThat(target.getJavaIdeInfo().getJdeps().getRelativePath())
        .isEqualTo("com/google/example/libl.jdeps");
  }

  @Test
  public void testAndroidBinary() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "android_library(",
        "  name = 'l1',",
        "  manifest = 'AndroidManifest.xml',",
        "  custom_package = 'com.google.example',",
        "  resource_files = ['r1/values/a.xml'],",
        ")",
        "android_binary(",
        "  name = 'b',",
        "  srcs = ['Main.java'],",
        "  deps = [':l1'],",
        "  manifest = 'AndroidManifest.xml',",
        "  custom_package = 'com.google.example',",
        "  resource_files = ['res/drawable/a.png', 'res/drawable/b.png'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:b");
    TargetIdeInfo target = getTargetIdeInfoAndVerifyLabel("//com/google/example:b", targetIdeInfos);

    assertThat(target.getKindString()).isEqualTo("android_binary");
    assertThat(relativePathsForJavaSourcesOf(target))
        .containsExactly("com/google/example/Main.java");
    assertThat(transform(target.getJavaIdeInfo().getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(
            jarString("com/google/example", "libb.jar", "libb-hjar.jar", "libb-src.jar"),
            jarString("com/google/example", "b_resources.jar", null, "b_resources-src.jar"));

    assertThat(transform(target.getAndroidIdeInfo().getResourcesList(), ARTIFACT_TO_RELATIVE_PATH))
        .containsExactly("com/google/example/res");
    assertThat(target.getAndroidIdeInfo().getManifest().getRelativePath())
        .isEqualTo("com/google/example/AndroidManifest.xml");
    assertThat(target.getAndroidIdeInfo().getJavaPackage()).isEqualTo("com.google.example");
    assertThat(target.getAndroidIdeInfo().getApk().getRelativePath())
        .isEqualTo("com/google/example/b.apk");

    assertThat(target.getDependenciesList()).contains("//com/google/example:l1");

    assertThat(getIdeResolveFiles())
        .containsExactly(
            "com/google/example/libb.jar",
            "com/google/example/libb-hjar.jar",
            "com/google/example/libb-src.jar",
            "com/google/example/b_resources.jar",
            "com/google/example/b_resources-src.jar",
            "com/google/example/libl1.jar",
            "com/google/example/libl1-src.jar",
            "com/google/example/l1_resources.jar",
            "com/google/example/l1_resources-src.jar");
    assertThat(target.getJavaIdeInfo().getJdeps().getRelativePath())
        .isEqualTo("com/google/example/libb.jdeps");
  }

  @Test
  public void testAndroidInferredPackage() throws Exception {
    scratch.file(
        "java/com/google/example/BUILD",
        "android_library(",
        "  name = 'l',",
        "  manifest = 'AndroidManifest.xml',",
        ")",
        "android_binary(",
        "  name = 'b',",
        "  srcs = ['Main.java'],",
        "  deps = [':l'],",
        "  manifest = 'AndroidManifest.xml',",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//java/com/google/example:b");
    TargetIdeInfo lTarget =
        getTargetIdeInfoAndVerifyLabel("//java/com/google/example:l", targetIdeInfos);
    TargetIdeInfo bTarget =
        getTargetIdeInfoAndVerifyLabel("//java/com/google/example:b", targetIdeInfos);

    assertThat(bTarget.getAndroidIdeInfo().getJavaPackage()).isEqualTo("com.google.example");
    assertThat(lTarget.getAndroidIdeInfo().getJavaPackage()).isEqualTo("com.google.example");
  }

  @Test
  public void testAndroidLibraryWithoutAidlHasNoIdlJars() throws Exception {
    scratch.file(
        "java/com/google/example/BUILD",
        "android_library(",
        "  name = 'no_idl',",
        "  srcs = ['Test.java'],",
        ")");
    String noIdlTarget = "//java/com/google/example:no_idl";
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo(noIdlTarget);
    TargetIdeInfo noIdlTargetIdeInfo = getTargetIdeInfoAndVerifyLabel(noIdlTarget, targetIdeInfos);

    assertThat(noIdlTargetIdeInfo.getAndroidIdeInfo().getHasIdlSources()).isFalse();
  }

  @Test
  public void testAndroidLibraryWithAidlHasIdlJars() throws Exception {
    scratch.file(
        "java/com/google/example/BUILD",
        "android_library(",
        "  name = 'has_idl',",
        "  idl_srcs = ['a.aidl'],",
        ")");
    String idlTarget = "//java/com/google/example:has_idl";
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo(idlTarget);
    TargetIdeInfo idlTargetIdeInfo = getTargetIdeInfoAndVerifyLabel(idlTarget, targetIdeInfos);

    assertThat(idlTargetIdeInfo.getAndroidIdeInfo().getHasIdlSources()).isTrue();
    assertThat(LIBRARY_ARTIFACT_TO_STRING.apply(idlTargetIdeInfo.getAndroidIdeInfo().getIdlJar()))
        .isEqualTo(
            jarString(
                "java/com/google/example", "libhas_idl-idl.jar", null, "libhas_idl-idl.srcjar"));
    assertThat(relativePathsForJavaSourcesOf(idlTargetIdeInfo)).isEmpty();
    assertThat(getIdeResolveFiles())
        .containsExactly(
            "java/com/google/example/libhas_idl.jar",
            "java/com/google/example/libhas_idl-hjar.jar",
            "java/com/google/example/libhas_idl-src.jar",
            "java/com/google/example/libhas_idl-idl.jar",
            "java/com/google/example/libhas_idl-idl.srcjar");
  }

  @Test
  public void testAndroidLibraryWithAidlWithoutImportRoot() throws Exception {
    scratch.file(
        "java/com/google/example/BUILD",
        "android_library(",
        "  name = 'no_idl_import_root',",
        "  idl_srcs = ['a.aidl'],",
        ")");
    String idlTarget = "//java/com/google/example:no_idl_import_root";
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo(idlTarget);
    TargetIdeInfo idlTargetIdeInfo = getTargetIdeInfoAndVerifyLabel(idlTarget, targetIdeInfos);
    assertThat(idlTargetIdeInfo.getAndroidIdeInfo().getIdlImportRoot()).isEmpty();
  }

  @Test
  public void testAndroidLibraryWithAidlWithImportRoot() throws Exception {
    scratch.file(
        "java/com/google/example/BUILD",
        "android_library(",
        "  name = 'has_idl_import_root',",
        "  idl_import_root = 'idl',",
        "  idl_srcs = ['idl/com/google/example/a.aidl'],",
        ")");
    String idlTarget = "//java/com/google/example:has_idl_import_root";
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo(idlTarget);
    TargetIdeInfo idlTargetIdeInfo = getTargetIdeInfoAndVerifyLabel(idlTarget, targetIdeInfos);
    assertThat(idlTargetIdeInfo.getAndroidIdeInfo().getIdlImportRoot()).isEqualTo("idl");
  }

  @Test
  public void testAndroidLibraryGeneratedManifestIsAddedToOutputGroup() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "android_library(",
        "  name = 'lib',",
        "  manifest = ':manifest',",
        "  custom_package = 'com.google.example',",
        ")",
        "genrule(",
        "  name = 'manifest',",
        "  outs = ['AndroidManifest.xml'],",
        "  cmd = '',",
        ")");
    buildIdeInfo("//com/google/example:lib");
    assertThat(getIdeResolveFiles())
        .containsExactly(
            "com/google/example/liblib.jar",
            "com/google/example/liblib-src.jar",
            "com/google/example/lib_resources.jar",
            "com/google/example/lib_resources-src.jar",
            "com/google/example/AndroidManifest.xml");
  }

  @Test
  public void testJavaLibraryWithoutGeneratedSourcesHasNoGenJars() throws Exception {
    scratch.file(
        "java/com/google/example/BUILD",
        "java_library(",
        "  name = 'no_plugin',",
        "  srcs = ['Test.java'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//java/com/google/example:no_plugin");
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//java/com/google/example:no_plugin", targetIdeInfos);

    assertThat(targetIdeInfo.getJavaIdeInfo().getGeneratedJarsList()).isEmpty();
  }

  @Test
  public void testJavaLibraryWithGeneratedSourcesHasGenJars() throws Exception {
    scratch.file(
        "java/com/google/example/BUILD",
        "java_library(",
        "  name = 'test',",
        "  srcs = ['Test.java'],",
        "  plugins = [':plugin']",
        ")",
        "java_plugin(",
        "  name = 'plugin',",
        "  processor_class = 'com.google.example.Plugin',",
        "  deps = ['plugin_lib'],",
        ")",
        "java_library(",
        "  name = 'plugin_lib',",
        "  srcs = ['Plugin.java'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//java/com/google/example:test");
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//java/com/google/example:test", targetIdeInfos);

    assertThat(
            transform(
                targetIdeInfo.getJavaIdeInfo().getGeneratedJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(
            jarString("java/com/google/example", "libtest-gen.jar", null, "libtest-gensrc.jar"));
    assertThat(getIdeResolveFiles())
        .containsExactly(
            "java/com/google/example/libtest.jar",
            "java/com/google/example/libtest-hjar.jar",
            "java/com/google/example/libtest-src.jar",
            "java/com/google/example/libtest-gen.jar",
            "java/com/google/example/libtest-gensrc.jar");
  }

  @Test
  public void testTags() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "    name = 'lib',",
        "    srcs = ['Test.java'],",
        "    tags = ['d', 'b', 'c', 'a'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:lib");
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:lib", targetIdeInfos);
    assertThat(targetIdeInfo.getTagsList()).containsExactly("a", "b", "c", "d");
  }

  @Test
  public void testAndroidLibraryWithoutSourcesExportsDependencies() throws Exception {
    scratch.file(
        "java/com/google/example/BUILD",
        "android_library(",
        "  name = 'lib',",
        "  srcs = ['Test.java']",
        ")",
        "android_library(",
        "  name = 'forward',",
        "  deps = [':lib'],",
        ")",
        "android_library(",
        "  name = 'super',",
        "  deps = [':forward'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//java/com/google/example:super");
    TargetIdeInfo target =
        getTargetIdeInfoAndVerifyLabel("//java/com/google/example:super", targetIdeInfos);

    assertThat(target.getDependenciesList())
        .containsAllOf("//java/com/google/example:forward", "//java/com/google/example:lib");
  }

  @Test
  public void testAndroidLibraryExportsDoNotOverReport() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "android_library(",
        "  name = 'lib',",
        "  deps = [':middle'],",
        ")",
        "android_library(",
        "  name = 'middle',",
        "  srcs = ['Middle.java'],",
        "  deps = [':exported'],",
        ")",
        "android_library(",
        "  name = 'exported',",
        "  srcs = ['Exported.java'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:lib");
    TargetIdeInfo target =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:lib", targetIdeInfos);
    TargetIdeInfo javaToolchain = Iterables.getOnlyElement(findJavaToolchain(targetIdeInfos));
    assertThat(target.getDependenciesList())
        .containsExactly(javaToolchain.getLabel(), "//com/google/example:middle");
  }

  @Test
  public void testSourceFilesAreCorrectlyMarkedAsSourceOrGenerated() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "genrule(",
        "   name = 'gen',",
        "   outs = ['gen.java'],",
        "   cmd = '',",
        ")",
        "java_library(",
        "    name = 'lib',",
        "    srcs = ['Test.java', ':gen'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:lib");
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:lib", targetIdeInfos);
    // todo(dslomov): Skylark aspect implementation does not yet return a correct root path.
    assertThat(targetIdeInfo.getJavaIdeInfo().getSourcesList())
        .containsExactly(
            ArtifactLocation.newBuilder()
                .setRootExecutionPathFragment(
                    targetConfig.getGenfilesDirectory(RepositoryName.MAIN).getExecPathString())
                .setRelativePath("com/google/example/gen.java")
                .setIsSource(false)
                .build(),
            ArtifactLocation.newBuilder()
                .setRelativePath("com/google/example/Test.java")
                .setIsSource(true)
                .build());
  }

  @Test
  public void testAspectIsPropagatedAcrossRuntimeDeps() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "   name = 'foobar',",
        "   srcs = ['FooBar.java'],",
        ")",
        "java_library(",
        "   name = 'lib',",
        "   srcs = ['Lib.java'],",
        "   runtime_deps = [':foobar'],",
        ")");

    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:lib");
    // Fails if aspect was not propagated
    getTargetIdeInfoAndVerifyLabel("//com/google/example:foobar", targetIdeInfos);

    getTargetIdeInfoAndVerifyLabel("//com/google/example:foobar", targetIdeInfos);
  }

  @Test
  public void testRuntimeDepsAddedToProto() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "   name = 'foobar',",
        "   srcs = ['FooBar.java'],",
        ")",
        "java_library(",
        "   name = 'foobar2',",
        "   srcs = ['FooBar2.java'],",
        ")",
        "java_library(",
        "   name = 'lib',",
        "   srcs = ['Lib.java'],",
        "   deps = [':lib2'],",
        "   runtime_deps = [':foobar'],",
        ")",
        "java_library(",
        "   name = 'lib2',",
        "   srcs = ['Lib2.java'],",
        "   runtime_deps = [':foobar2'],",
        ")");

    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:lib");
    // Fails if aspect was not propagated
    TargetIdeInfo lib = getTargetIdeInfoAndVerifyLabel("//com/google/example:lib", targetIdeInfos);
    TargetIdeInfo lib2 =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:lib2", targetIdeInfos);

    assertThat(lib.getRuntimeDepsList()).containsExactly("//com/google/example:foobar");
    assertThat(lib2.getRuntimeDepsList()).containsExactly("//com/google/example:foobar2");
  }

  @Test
  public void testAndroidLibraryGeneratesResourceClass() throws Exception {
    scratch.file(
        "java/com/google/example/BUILD",
        "android_library(",
        "   name = 'resource_files',",
        "   resource_files = ['res/drawable/a.png'],",
        "   manifest = 'AndroidManifest.xml',",
        ")",
        "android_library(",
        "   name = 'manifest',",
        "   manifest = 'AndroidManifest.xml',",
        ")",
        "android_library(",
        "   name = 'neither',",
        "   srcs = ['FooBar.java'],",
        "   deps = [':resource_files', ':manifest']",
        ")");

    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//java/com/google/example:neither");
    TargetIdeInfo neither =
        getTargetIdeInfoAndVerifyLabel("//java/com/google/example:neither", targetIdeInfos);
    TargetIdeInfo resourceFiles =
        getTargetIdeInfoAndVerifyLabel("//java/com/google/example:resource_files", targetIdeInfos);
    TargetIdeInfo manifest =
        getTargetIdeInfoAndVerifyLabel("//java/com/google/example:manifest", targetIdeInfos);

    assertThat(neither.getAndroidIdeInfo().getGenerateResourceClass()).isFalse();
    assertThat(resourceFiles.getAndroidIdeInfo().getGenerateResourceClass()).isTrue();
    assertThat(manifest.getAndroidIdeInfo().getGenerateResourceClass()).isTrue();
  }

  @Test
  public void testJavaPlugin() throws Exception {
    scratch.file(
        "java/com/google/example/BUILD",
        "java_plugin(",
        "  name = 'plugin',",
        "  srcs = ['Plugin.java'],",
        "  processor_class = 'com.google.example.Plugin',",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//java/com/google/example:plugin");
    TargetIdeInfo plugin =
        getTargetIdeInfoAndVerifyLabel("//java/com/google/example:plugin", targetIdeInfos);

    assertThat(plugin.getKindString()).isEqualTo("java_plugin");
    assertThat(transform(plugin.getJavaIdeInfo().getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(
            jarString(
                "java/com/google/example",
                "libplugin.jar",
                "libplugin-hjar.jar",
                "libplugin-src.jar"));
  }

  @Test
  public void testSimpleCCLibraryForCCToolchainExistence() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "cc_library(",
        "    name = 'simple',",
        "    srcs = ['simple/simple.cc'],",
        "    hdrs = ['simple/simple.h'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:simple");
    assertThat(targetIdeInfos).hasSize(2);
    TargetIdeInfo target =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);
    Entry<String, TargetIdeInfo> toolchainEntry =
        getCcToolchainRuleAndVerifyThereIsOnlyOne(targetIdeInfos);
    TargetIdeInfo toolchainInfo = toolchainEntry.getValue();
    ArtifactLocation location = target.getBuildFileArtifactLocation();
    assertThat(Paths.get(location.getRelativePath()).toString())
        .isEqualTo(Paths.get("com/google/example/BUILD").toString());

    assertThat(target.hasCIdeInfo()).isTrue();
    assertThat(target.getDependenciesList()).hasSize(1);
    assertThat(toolchainInfo.hasCToolchainIdeInfo()).isTrue();
  }

  @Test
  public void testSimpleCCLibrary() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "cc_library(",
        "    name = 'simple',",
        "    srcs = ['simple/simple.cc'],",
        "    hdrs = ['simple/simple.h'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:simple");
    assertThat(targetIdeInfos).hasSize(2);
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);
    ArtifactLocation location = targetIdeInfo.getBuildFileArtifactLocation();
    assertThat(Paths.get(location.getRelativePath()).toString())
        .isEqualTo(Paths.get("com/google/example/BUILD").toString());

    assertThat(targetIdeInfo.getKindString()).isEqualTo("cc_library");
    assertThat(targetIdeInfo.getDependenciesCount()).isEqualTo(1);

    assertThat(relativePathsForCSourcesOf(targetIdeInfo))
        .containsExactly("com/google/example/simple/simple.cc");

    assertThat(targetIdeInfo.hasCIdeInfo()).isTrue();
    assertThat(targetIdeInfo.hasJavaIdeInfo()).isFalse();
    assertThat(targetIdeInfo.hasAndroidIdeInfo()).isFalse();
    CIdeInfo cTargetIdeInfo = targetIdeInfo.getCIdeInfo();

    assertThat(cTargetIdeInfo.getTargetCoptList()).isEmpty();
    assertThat(cTargetIdeInfo.getTargetDefineList()).isEmpty();
    assertThat(cTargetIdeInfo.getTargetIncludeList()).isEmpty();

    ProtocolStringList transQuoteIncludeDirList =
        cTargetIdeInfo.getTransitiveQuoteIncludeDirectoryList();
    assertThat(transQuoteIncludeDirList).contains(".");

    assertThat(targetIdeInfo.getJavaIdeInfo().getJarsList()).isEmpty();

    assertThat(getIdeResolveFiles()).containsExactly("com/google/example/simple/simple.h");
  }

  @Test
  public void testSimpleCCLibraryWithIncludes() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "cc_library(",
        "    name = 'simple',",
        "    srcs = ['simple/simple.cc'],",
        "    hdrs = ['simple/simple.h'],",
        "    includes = ['foo/bar'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:simple");
    assertThat(targetIdeInfos).hasSize(2);
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);

    assertThat(targetIdeInfo.hasCIdeInfo()).isTrue();
    CIdeInfo cTargetIdeInfo = targetIdeInfo.getCIdeInfo();

    assertThat(cTargetIdeInfo.getTargetIncludeList()).containsExactly("foo/bar");

    // Make sure our understanding of where this attributes show up in other providers is correct.
    Entry<String, TargetIdeInfo> toolchainEntry =
        getCcToolchainRuleAndVerifyThereIsOnlyOne(targetIdeInfos);
    TargetIdeInfo toolchainInfo = toolchainEntry.getValue();
    assertThat(toolchainInfo.hasCToolchainIdeInfo()).isTrue();
    CToolchainIdeInfo cToolchainIdeInfo = toolchainInfo.getCToolchainIdeInfo();
    ProtocolStringList builtInIncludeDirectoryList =
        cToolchainIdeInfo.getBuiltInIncludeDirectoryList();
    assertThat(builtInIncludeDirectoryList).doesNotContain("foo/bar");
    assertThat(builtInIncludeDirectoryList).doesNotContain("com/google/example/foo/bar");

    ProtocolStringList transIncludeDirList = cTargetIdeInfo.getTransitiveIncludeDirectoryList();
    assertThat(transIncludeDirList).doesNotContain("foo/bar");
    assertThat(transIncludeDirList).doesNotContain("com/google/example/foo/bar");

    ProtocolStringList transQuoteIncludeDirList =
        cTargetIdeInfo.getTransitiveQuoteIncludeDirectoryList();
    assertThat(transQuoteIncludeDirList).doesNotContain("foo/bar");
    assertThat(transQuoteIncludeDirList).doesNotContain("com/google/example/foo/bar");

    ProtocolStringList transSysIncludeDirList =
        cTargetIdeInfo.getTransitiveSystemIncludeDirectoryList();
    assertThat(transSysIncludeDirList).doesNotContain("foo/bar");
    assertThat(transSysIncludeDirList).contains("com/google/example/foo/bar");
  }

  @Test
  public void testSimpleCCLibraryWithCompilerFlags() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "cc_library(",
        "    name = 'simple',",
        "    srcs = ['simple/simple.cc'],",
        "    hdrs = ['simple/simple.h'],",
        "    copts = ['-DGOPT', '-Ifoo/baz/'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:simple");
    assertThat(targetIdeInfos).hasSize(2);
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);

    assertThat(targetIdeInfo.hasCIdeInfo()).isTrue();
    CIdeInfo cTargetIdeInfo = targetIdeInfo.getCIdeInfo();

    assertThat(cTargetIdeInfo.getTargetCoptList()).containsExactly("-DGOPT", "-Ifoo/baz/");

    // Make sure our understanding of where this attributes show up in other providers is correct.
    Entry<String, TargetIdeInfo> toolchainEntry =
        getCcToolchainRuleAndVerifyThereIsOnlyOne(targetIdeInfos);
    TargetIdeInfo toolchainInfo = toolchainEntry.getValue();
    assertThat(toolchainInfo.hasCToolchainIdeInfo()).isTrue();
    CToolchainIdeInfo cToolchainIdeInfo = toolchainInfo.getCToolchainIdeInfo();
    ProtocolStringList baseCompilerOptionList = cToolchainIdeInfo.getBaseCompilerOptionList();
    assertThat(baseCompilerOptionList).doesNotContain("-DGOPT");
    assertThat(baseCompilerOptionList).doesNotContain("-Ifoo/baz/");

    ProtocolStringList cOptionList = cToolchainIdeInfo.getCOptionList();
    assertThat(cOptionList).doesNotContain("-DGOPT");
    assertThat(cOptionList).doesNotContain("-Ifoo/baz/");

    ProtocolStringList cppOptionList = cToolchainIdeInfo.getCppOptionList();
    assertThat(cppOptionList).doesNotContain("-DGOPT");
    assertThat(cppOptionList).doesNotContain("-Ifoo/baz/");
  }

  @Test
  public void testSimpleCCLibraryWithDefines() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "cc_library(",
        "    name = 'simple',",
        "    srcs = ['simple/simple.cc'],",
        "    hdrs = ['simple/simple.h'],",
        "    defines = ['VERSION2'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:simple");
    assertThat(targetIdeInfos).hasSize(2);
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);

    assertThat(targetIdeInfo.hasCIdeInfo()).isTrue();
    CIdeInfo cTargetIdeInfo = targetIdeInfo.getCIdeInfo();

    assertThat(cTargetIdeInfo.getTargetDefineList()).containsExactly("VERSION2");

    // Make sure our understanding of where this attributes show up in other providers is correct.
    ProtocolStringList transDefineList = cTargetIdeInfo.getTransitiveDefineList();
    assertThat(transDefineList).contains("VERSION2");
  }

  @Test
  public void testSimpleCCBinary() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "cc_binary(",
        "    name = 'simple',",
        "    srcs = ['simple/simple.cc'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:simple");
    assertThat(targetIdeInfos).hasSize(2);
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);
    ArtifactLocation location = targetIdeInfo.getBuildFileArtifactLocation();
    assertThat(Paths.get(location.getRelativePath()).toString())
        .isEqualTo(Paths.get("com/google/example/BUILD").toString());
    assertThat(targetIdeInfo.getKindString()).isEqualTo("cc_binary");
    assertThat(targetIdeInfo.getDependenciesCount()).isEqualTo(1);

    assertThat(relativePathsForCSourcesOf(targetIdeInfo))
        .containsExactly("com/google/example/simple/simple.cc");

    assertThat(targetIdeInfo.hasCIdeInfo()).isTrue();
    assertThat(targetIdeInfo.hasJavaIdeInfo()).isFalse();
    assertThat(targetIdeInfo.hasAndroidIdeInfo()).isFalse();
    CIdeInfo cTargetIdeInfo = targetIdeInfo.getCIdeInfo();

    assertThat(cTargetIdeInfo.getTargetCoptList()).isEmpty();
    assertThat(cTargetIdeInfo.getTargetDefineList()).isEmpty();
    assertThat(cTargetIdeInfo.getTargetIncludeList()).isEmpty();

    assertThat(targetIdeInfo.getJavaIdeInfo().getJarsList()).isEmpty();

    assertThat(getIdeResolveFiles()).isEmpty();
  }

  @Test
  public void testSimpleCCTest() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "cc_test(",
        "    name = 'simple',",
        "    srcs = ['simple/simple.cc'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:simple");
    assertThat(targetIdeInfos).hasSize(2);
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);
    ArtifactLocation location = targetIdeInfo.getBuildFileArtifactLocation();
    assertThat(Paths.get(location.getRelativePath()).toString())
        .isEqualTo(Paths.get("com/google/example/BUILD").toString());
    assertThat(targetIdeInfo.getKindString()).isEqualTo("cc_test");
    assertThat(targetIdeInfo.getDependenciesCount()).isEqualTo(1);

    assertThat(relativePathsForCSourcesOf(targetIdeInfo))
        .containsExactly("com/google/example/simple/simple.cc");

    assertThat(targetIdeInfo.hasCIdeInfo()).isTrue();
    assertThat(targetIdeInfo.hasJavaIdeInfo()).isFalse();
    assertThat(targetIdeInfo.hasAndroidIdeInfo()).isFalse();
    CIdeInfo cTargetIdeInfo = targetIdeInfo.getCIdeInfo();

    assertThat(cTargetIdeInfo.getTargetCoptList()).isEmpty();
    assertThat(cTargetIdeInfo.getTargetDefineList()).isEmpty();
    assertThat(cTargetIdeInfo.getTargetIncludeList()).isEmpty();

    assertThat(targetIdeInfo.getJavaIdeInfo().getJarsList()).isEmpty();

    assertThat(getIdeResolveFiles()).isEmpty();
  }

  @Test
  public void testSimpleCCLibraryWithDeps() throws Exception {
    // Specify '-fPIC' so that compilation output filenames are consistent for mac and linux.
    scratch.file(
        "com/google/example/BUILD",
        "cc_library(",
        "    name = 'lib',",
        "    srcs = ['lib/lib.cc'],",
        "    hdrs = ['lib/lib.h'],",
        ")",
        "cc_library(",
        "    name = 'simple',",
        "    srcs = ['simple/simple.cc'],",
        "    hdrs = ['simple/simple.h'],",
        "    deps = [':lib'],",
        "    nocopts = '-fPIC',",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:simple");
    assertThat(targetIdeInfos).hasSize(3);
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);

    assertThat(targetIdeInfo.getDependenciesList()).contains("//com/google/example:lib");
    assertThat(targetIdeInfo.getDependenciesList()).hasSize(2);

    assertThat(getIdeCompileFiles())
        .containsExactly("com/google/example/_objs/simple/com/google/example/simple/simple.o");
  }

  @Test
  public void testSimpleAndroidBinaryThatDependsOnCCLibrary() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "android_library(",
        "    name = 'androidlib',",
        "    srcs = ['Lib.java'],",
        "    deps = ['simple'],",
        ")",
        "cc_library(",
        "    name = 'simple',",
        "    srcs = ['simple/simple.cc'],",
        "    hdrs = ['simple/simple.h'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:androidlib");
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:androidlib", targetIdeInfos);

    assertThat(targetIdeInfo.getDependenciesList()).contains("//com/google/example:simple");
  }

  @Test
  public void testTransitiveCCLibraryWithIncludes() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "cc_library(",
        "    name = 'lib2',",
        "    srcs = ['lib2/lib2.cc'],",
        "    hdrs = ['lib2/lib2.h'],",
        "    includes = ['baz/lib'],",
        ")",
        "cc_library(",
        "    name = 'lib1',",
        "    srcs = ['lib1/lib1.cc'],",
        "    hdrs = ['lib1/lib1.h'],",
        "    includes = ['foo/bar'],",
        "    deps = [':lib2'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:lib1");
    assertThat(targetIdeInfos).hasSize(3);
    TargetIdeInfo lib1 =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:lib1", targetIdeInfos);

    assertThat(lib1.hasCIdeInfo()).isTrue();
    CIdeInfo cTargetIdeInfo = lib1.getCIdeInfo();

    assertThat(cTargetIdeInfo.getTargetIncludeList()).containsExactly("foo/bar");

    // Make sure our understanding of where this attributes show up in other providers is correct.
    Entry<String, TargetIdeInfo> toolchainEntry =
        getCcToolchainRuleAndVerifyThereIsOnlyOne(targetIdeInfos);
    TargetIdeInfo toolchainInfo = toolchainEntry.getValue();
    assertThat(toolchainInfo.hasCToolchainIdeInfo()).isTrue();
    CToolchainIdeInfo cToolchainIdeInfo = toolchainInfo.getCToolchainIdeInfo();
    ProtocolStringList builtInIncludeDirectoryList =
        cToolchainIdeInfo.getBuiltInIncludeDirectoryList();
    assertThat(builtInIncludeDirectoryList).doesNotContain("foo/bar");
    assertThat(builtInIncludeDirectoryList).doesNotContain("baz/lib");
    assertThat(builtInIncludeDirectoryList).doesNotContain("com/google/example/foo/bar");
    assertThat(builtInIncludeDirectoryList).doesNotContain("com/google/example/baz/lib");

    ProtocolStringList transIncludeDirList = cTargetIdeInfo.getTransitiveIncludeDirectoryList();
    assertThat(transIncludeDirList).doesNotContain("foo/bar");
    assertThat(transIncludeDirList).doesNotContain("baz/lib");
    assertThat(transIncludeDirList).doesNotContain("com/google/example/foo/bar");
    assertThat(transIncludeDirList).doesNotContain("com/google/example/baz/lib");

    ProtocolStringList transQuoteIncludeDirList =
        cTargetIdeInfo.getTransitiveQuoteIncludeDirectoryList();
    assertThat(transQuoteIncludeDirList).doesNotContain("foo/bar");
    assertThat(transQuoteIncludeDirList).doesNotContain("baz/lib");
    assertThat(transQuoteIncludeDirList).doesNotContain("com/google/example/foo/bar");
    assertThat(transQuoteIncludeDirList).doesNotContain("com/google/example/baz/lib");

    ProtocolStringList transSysIncludeDirList =
        cTargetIdeInfo.getTransitiveSystemIncludeDirectoryList();
    assertThat(transSysIncludeDirList).doesNotContain("foo/bar");
    assertThat(transSysIncludeDirList).doesNotContain("baz/lib");
    assertThat(transSysIncludeDirList).contains("com/google/example/foo/bar");
    assertThat(transSysIncludeDirList).contains("com/google/example/baz/lib");
  }

  @Test
  public void testTransitiveCLibraryWithCompilerFlags() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "cc_library(",
        "    name = 'lib2',",
        "    srcs = ['lib2/lib2.cc'],",
        "    hdrs = ['lib2/lib2.h'],",
        "    copts = ['-v23', '-DDEV'],",
        ")",
        "cc_library(",
        "    name = 'lib1',",
        "    srcs = ['lib1/lib1.cc'],",
        "    hdrs = ['lib1/lib1.h'],",
        "    copts = ['-DGOPT', '-Ifoo/baz/'],",
        "    deps = [':lib2'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:lib1");
    assertThat(targetIdeInfos).hasSize(3);
    TargetIdeInfo lib1 =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:lib1", targetIdeInfos);

    assertThat(lib1.hasCIdeInfo()).isTrue();
    CIdeInfo cTargetIdeInfo = lib1.getCIdeInfo();

    assertThat(cTargetIdeInfo.getTargetCoptList()).containsExactly("-DGOPT", "-Ifoo/baz/");

    // Make sure our understanding of where this attributes show up in other providers is correct.
    Entry<String, TargetIdeInfo> toolchainEntry =
        getCcToolchainRuleAndVerifyThereIsOnlyOne(targetIdeInfos);
    TargetIdeInfo toolchainInfo = toolchainEntry.getValue();
    assertThat(toolchainInfo.hasCToolchainIdeInfo()).isTrue();
    CToolchainIdeInfo cToolchainIdeInfo = toolchainInfo.getCToolchainIdeInfo();
    ProtocolStringList baseCompilerOptionList = cToolchainIdeInfo.getBaseCompilerOptionList();
    assertThat(baseCompilerOptionList).doesNotContain("-DGOPT");
    assertThat(baseCompilerOptionList).doesNotContain("-Ifoo/baz/");
    assertThat(baseCompilerOptionList).doesNotContain("-v23");
    assertThat(baseCompilerOptionList).doesNotContain("-DDEV");

    ProtocolStringList cOptionList = cToolchainIdeInfo.getCOptionList();
    assertThat(cOptionList).doesNotContain("-DGOPT");
    assertThat(cOptionList).doesNotContain("-Ifoo/baz/");
    assertThat(cOptionList).doesNotContain("-v23");
    assertThat(cOptionList).doesNotContain("-DDEV");

    ProtocolStringList cppOptionList = cToolchainIdeInfo.getCppOptionList();
    assertThat(cppOptionList).doesNotContain("-DGOPT");
    assertThat(cppOptionList).doesNotContain("-Ifoo/baz/");
    assertThat(cppOptionList).doesNotContain("-v23");
    assertThat(cppOptionList).doesNotContain("-DDEV");
  }

  @Test
  public void testTransitiveCCLibraryWithDefines() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "cc_library(",
        "    name = 'lib2',",
        "    srcs = ['lib2/lib2.cc'],",
        "    hdrs = ['lib2/lib2.h'],",
        "    defines = ['COMPLEX_IMPL'],",
        ")",
        "cc_library(",
        "    name = 'lib1',",
        "    srcs = ['lib1/lib1.cc'],",
        "    hdrs = ['lib1/lib1.h'],",
        "    defines = ['VERSION2'],",
        "    deps = [':lib2'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:lib1");
    assertThat(targetIdeInfos).hasSize(3);
    TargetIdeInfo lib1 =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:lib1", targetIdeInfos);

    assertThat(lib1.hasCIdeInfo()).isTrue();
    CIdeInfo cIdeInfo = lib1.getCIdeInfo();

    assertThat(cIdeInfo.getTargetDefineList()).containsExactly("VERSION2");

    // Make sure our understanding of where this attributes show up in other providers is correct.
    ProtocolStringList transDefineList = cIdeInfo.getTransitiveDefineList();
    assertThat(transDefineList).contains("VERSION2");
    assertThat(transDefineList).contains("COMPLEX_IMPL");
  }

  @Test
  public void testMacroDoesntAffectRuleClass() throws Exception {
    scratch.file(
        "java/com/google/example/build_defs.bzl",
        "def my_macro(name):",
        "  native.android_binary(",
        "    name = name,",
        "    srcs = ['simple/Simple.java'],",
        "    manifest = 'AndroidManifest.xml',",
        ")");
    scratch.file(
        "java/com/google/example/BUILD",
        "load('//java/com/google/example:build_defs.bzl', 'my_macro')",
        "my_macro(",
        "    name = 'simple',",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//java/com/google/example:simple");
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//java/com/google/example:simple", targetIdeInfos);
    assertThat(targetIdeInfo.getKindString()).isEqualTo("android_binary");
  }

  @Test
  public void testAndroidBinaryIsSerialized() throws Exception {
    TargetIdeInfo.Builder builder = TargetIdeInfo.newBuilder();
    builder.setKindString("android_binary");
    ByteString byteString = builder.build().toByteString();
    TargetIdeInfo result = TargetIdeInfo.parseFrom(byteString);
    assertThat(result.getKindString()).isEqualTo("android_binary");
  }

  @Test
  public void testCcToolchainInfoIsOnlyPresentForToolchainRules() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "cc_library(",
        "    name = 'simple',",
        "    srcs = ['simple/simple.cc'],",
        "    hdrs = ['simple/simple.h'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:simple");
    assertThat(targetIdeInfos).hasSize(2);
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);
    Entry<String, TargetIdeInfo> toolchainEntry =
        getCcToolchainRuleAndVerifyThereIsOnlyOne(targetIdeInfos);
    TargetIdeInfo toolchainInfo = toolchainEntry.getValue();
    ArtifactLocation location = targetIdeInfo.getBuildFileArtifactLocation();
    assertThat(Paths.get(location.getRelativePath()).toString())
        .isEqualTo(Paths.get("com/google/example/BUILD").toString());

    assertThat(targetIdeInfo.hasCToolchainIdeInfo()).isFalse();
    assertThat(toolchainInfo.hasCToolchainIdeInfo()).isTrue();
  }

  @Test
  public void testJavaLibraryDoesNotHaveCInfo() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "    name = 'simple',",
        "    srcs = ['simple/Simple.java']",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:simple");
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);
    assertThat(targetIdeInfo.hasCIdeInfo()).isFalse();
  }

  @Test
  public void testSimplePyBinary() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "py_binary(",
        "    name = 'simple',",
        "    srcs = ['simple/simple.py'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:simple");
    assertThat(targetIdeInfos).hasSize(2);
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);
    ArtifactLocation location = targetIdeInfo.getBuildFileArtifactLocation();
    assertThat(Paths.get(location.getRelativePath()).toString())
        .isEqualTo(Paths.get("com/google/example/BUILD").toString());
    assertThat(targetIdeInfo.getKindString()).isEqualTo("py_binary");
    assertThat(targetIdeInfo.getDependenciesCount()).isEqualTo(1);

    assertThat(relativePathsForPySourcesOf(targetIdeInfo))
        .containsExactly("com/google/example/simple/simple.py");

    assertThat(targetIdeInfo.hasPyIdeInfo()).isTrue();
    assertThat(targetIdeInfo.hasJavaIdeInfo()).isFalse();
    assertThat(targetIdeInfo.hasCIdeInfo()).isFalse();
    assertThat(targetIdeInfo.hasAndroidIdeInfo()).isFalse();

    assertThat(getIdeResolveFiles()).containsExactly("com/google/example/simple/simple.py");
  }

  @Test
  public void testSimplePyLibrary() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "py_library(",
        "    name = 'simple',",
        "    srcs = ['simple/simple.py'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:simple");
    assertThat(targetIdeInfos).hasSize(1);
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);
    ArtifactLocation location = targetIdeInfo.getBuildFileArtifactLocation();
    assertThat(Paths.get(location.getRelativePath()).toString())
        .isEqualTo(Paths.get("com/google/example/BUILD").toString());
    assertThat(targetIdeInfo.getKindString()).isEqualTo("py_library");
    assertThat(targetIdeInfo.getDependenciesCount()).isEqualTo(0);

    assertThat(relativePathsForPySourcesOf(targetIdeInfo))
        .containsExactly("com/google/example/simple/simple.py");

    assertThat(targetIdeInfo.hasPyIdeInfo()).isTrue();
    assertThat(targetIdeInfo.hasJavaIdeInfo()).isFalse();
    assertThat(targetIdeInfo.hasCIdeInfo()).isFalse();
    assertThat(targetIdeInfo.hasAndroidIdeInfo()).isFalse();

    assertThat(getIdeResolveFiles()).containsExactly("com/google/example/simple/simple.py");
  }

  @Test
  public void testSimplePyTest() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "py_test(",
        "    name = 'simple',",
        "    srcs = ['simple/simple.py'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:simple");
    assertThat(targetIdeInfos).hasSize(2);
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:simple", targetIdeInfos);
    ArtifactLocation location = targetIdeInfo.getBuildFileArtifactLocation();
    assertThat(Paths.get(location.getRelativePath()).toString())
        .isEqualTo(Paths.get("com/google/example/BUILD").toString());
    assertThat(targetIdeInfo.getKindString()).isEqualTo("py_test");
    assertThat(targetIdeInfo.getDependenciesCount()).isEqualTo(1);

    assertThat(relativePathsForPySourcesOf(targetIdeInfo))
        .containsExactly("com/google/example/simple/simple.py");

    assertThat(targetIdeInfo.hasPyIdeInfo()).isTrue();
    assertThat(targetIdeInfo.hasJavaIdeInfo()).isFalse();
    assertThat(targetIdeInfo.hasCIdeInfo()).isFalse();
    assertThat(targetIdeInfo.hasAndroidIdeInfo()).isFalse();

    assertThat(getIdeResolveFiles()).containsExactly("com/google/example/simple/simple.py");
  }

  @Test
  public void testPyTestWithDeps() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "py_library(",
        "    name = 'lib',",
        "    srcs = ['lib.py'],",
        ")",
        "py_test(",
        "    name = 'test',",
        "    srcs = ['test.py'],",
        "    deps = [':lib'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:test");
    assertThat(targetIdeInfos).hasSize(3);
    TargetIdeInfo targetIdeInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:test", targetIdeInfos);
    ArtifactLocation location = targetIdeInfo.getBuildFileArtifactLocation();
    assertThat(Paths.get(location.getRelativePath()).toString())
        .isEqualTo(Paths.get("com/google/example/BUILD").toString());
    assertThat(targetIdeInfo.getKindString()).isEqualTo("py_test");

    assertThat(targetIdeInfo.getDependenciesList()).contains("//com/google/example:lib");
    assertThat(targetIdeInfo.getDependenciesCount()).isEqualTo(2);

    assertThat(relativePathsForPySourcesOf(targetIdeInfo))
        .containsExactly("com/google/example/test.py");

    assertThat(targetIdeInfo.hasPyIdeInfo()).isTrue();
    assertThat(targetIdeInfo.hasJavaIdeInfo()).isFalse();
    assertThat(targetIdeInfo.hasCIdeInfo()).isFalse();
    assertThat(targetIdeInfo.hasAndroidIdeInfo()).isFalse();

    assertThat(getIdeResolveFiles())
        .containsExactly("com/google/example/test.py", "com/google/example/lib.py");
  }

  @Test
  public void testAlias() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "    name = 'test',",
        "    srcs = ['Test.java'],",
        "    deps = [':alias']",
        ")",
        "alias(",
        "    name = 'alias',",
        "    actual = ':alias2',",
        ")",
        "alias(",
        "    name = 'alias2',",
        "    actual = ':real',",
        ")",
        "java_library(",
        "    name = 'real',",
        "    srcs = ['Real.java'],",
        ")");
    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:test");
    TargetIdeInfo testInfo =
        getTargetIdeInfoAndVerifyLabel("//com/google/example:test", targetIdeInfos);
    assertThat(testInfo.getDependenciesList()).contains("//com/google/example:real");
    assertThat(getTargetIdeInfoAndVerifyLabel("//com/google/example:real", targetIdeInfos))
        .isNotNull();
  }

  @Test
  public void testDataModeDepsAttributeDoesNotCrashAspect() throws Exception {
    scratch.file(
        "com/google/example/foo.bzl",
        "def impl(ctx):",
        "  return struct()",
        "",
        "foo = rule(",
        "  implementation=impl,",
        "  attrs={'deps': attr.label_list(cfg='data')},",
        ")");
    scratch.file(
        "com/google/example/BUILD",
        "load('//com/google/example:foo.bzl', 'foo')",
        "foo(",
        "  name='foo',",
        ")");
    buildIdeInfo("//com/google/example:foo");
  }

  @Test
  public void testExternalRootCorrectlyIdentified() throws Exception {
    ArtifactLocation location =
        AndroidStudioInfoAspect.makeArtifactLocation(
            Root.asSourceRoot(outputBase, false), new PathFragment("external/foo/bar.jar"), true);
    assertThat(location.getIsExternal()).isTrue();
  }

  @Test
  public void testNonExternalRootCorrectlyIdentified() throws Exception {
    ArtifactLocation location =
        AndroidStudioInfoAspect.makeArtifactLocation(
            Root.asSourceRoot(rootDirectory, true), new PathFragment("foo/bar.jar"), false);
    assertThat(location.getIsExternal()).isFalse();
  }

  @Test
  public void testExternalTarget() throws Exception {
    scratch.file(
        "/r/BUILD", "java_import(", "    name = 'junit',", "    jars = ['junit.jar'],", ")");
    scratch.file("/r/junit.jar");

    // AnalysisMock adds required toolchains, etc. to WORKSPACE, so retain the previous contents.
    String oldContents = scratch.readFile("WORKSPACE");
    scratch.overwriteFile("WORKSPACE", oldContents + "\nlocal_repository(name='r', path='/r')");
    invalidatePackages();

    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "    name = 'junit',",
        "    exports = ['@r//:junit'],",
        ")");

    Map<String, TargetIdeInfo> targetIdeInfos = buildIdeInfo("//com/google/example:junit");
    assertThat(
            getTargetIdeInfoAndVerifyLabel("//com/google/example:junit", targetIdeInfos)
                .getBuildFileArtifactLocation()
                .getIsExternal())
        .isFalse();

    TargetIdeInfo targetInfo = getTargetIdeInfoAndVerifyLabel("@r//:junit", targetIdeInfos);
    assertThat(targetInfo.getBuildFileArtifactLocation().getIsExternal()).isTrue();
    assertThat(targetInfo.getBuildFileArtifactLocation().getRelativePath()).startsWith("external");

    JavaIdeInfo javaInfo = targetInfo.getJavaIdeInfo();
    assertThat(javaInfo.getJarsList()).hasSize(1);
    ArtifactLocation jar = javaInfo.getJars(0).getJar();
    assertThat(jar.getIsSource()).isTrue();
    assertThat(jar.getIsExternal()).isTrue();
    assertThat(jar.getRelativePath()).isEqualTo("external/r/junit.jar");
  }
}
