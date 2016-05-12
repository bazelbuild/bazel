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

import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.ArtifactLocation;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.CRuleIdeInfo;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.CToolchainIdeInfo;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.JavaRuleIdeInfo;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.RuleIdeInfo;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.RuleIdeInfo.Kind;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import com.google.protobuf.ProtocolStringList;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.nio.file.Paths;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Tests for {@link AndroidStudioInfoAspect} validating proto's contents.
 */
@RunWith(JUnit4.class)
public class AndroidStudioInfoAspectTest extends AndroidStudioInfoAspectTestBase {

  @Test
  public void testSimpleJavaLibrary() throws Exception {
    Path buildFilePath =
        scratch.file(
            "com/google/example/BUILD",
            "java_library(",
            "    name = 'simple',",
            "    srcs = ['simple/Simple.java']",
            ")");
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:simple");
    assertThat(ruleIdeInfos.size()).isEqualTo(1);
    RuleIdeInfo ruleIdeInfo = getRuleInfoAndVerifyLabel(
        "//com/google/example:simple", ruleIdeInfos);
    ArtifactLocation location = ruleIdeInfo.getBuildFileArtifactLocation();
    assertThat(location.getRelativePath()).isEqualTo("com/google/example/BUILD");
    assertThat(location.getIsSource()).isTrue();
    if (isNativeTest()) {  // These will not be implemented in Skylark aspect.
      assertThat(ruleIdeInfo.getBuildFile()).isEqualTo(buildFilePath.toString());
      assertThat(Paths.get(location.getRootPath(), location.getRelativePath()).toString())
          .isEqualTo(buildFilePath.toString());
    }
    assertThat(ruleIdeInfo.getKind()).isEqualTo(Kind.JAVA_LIBRARY);
    assertThat(ruleIdeInfo.getKindString()).isEqualTo("java_library");
    assertThat(ruleIdeInfo.getDependenciesCount()).isEqualTo(0);
    assertThat(relativePathsForJavaSourcesOf(ruleIdeInfo))
        .containsExactly("com/google/example/simple/Simple.java");
    assertThat(
            transform(ruleIdeInfo.getJavaRuleIdeInfo().getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(jarString("com/google/example",
                "libsimple.jar", "libsimple-ijar.jar", "libsimple-src.jar"));

    assertThat(getIdeResolveFiles()).containsExactly(
        "com/google/example/libsimple.jar",
        "com/google/example/libsimple-ijar.jar",
        "com/google/example/libsimple-src.jar"
    );
    assertThat(ruleIdeInfo.getJavaRuleIdeInfo().getJdeps().getRelativePath())
        .isEqualTo("com/google/example/libsimple.jdeps");
  }

  @Test
  public void testPackageManifestCreated() throws Exception {
    if (!isNativeTest()) {
      return;
    }

    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "    name = 'simple',",
        "    srcs = ['simple/Simple.java']",
        ")");
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:simple");
    assertThat(ruleIdeInfos.size()).isEqualTo(1);
    RuleIdeInfo ruleIdeInfo = getRuleInfoAndVerifyLabel(
        "//com/google/example:simple", ruleIdeInfos);
    
    ArtifactLocation packageManifest = ruleIdeInfo.getJavaRuleIdeInfo().getPackageManifest();
    assertNotNull(packageManifest);
    assertEquals(packageManifest.getRelativePath(), "com/google/example/simple.manifest");
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

    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:complex");
    assertThat(ruleIdeInfos.size()).isEqualTo(2);

    getRuleInfoAndVerifyLabel("//com/google/example:simple", ruleIdeInfos);
    RuleIdeInfo complexRuleIdeInfo = getRuleInfoAndVerifyLabel(
        "//com/google/example:complex", ruleIdeInfos);

    assertThat(relativePathsForJavaSourcesOf(complexRuleIdeInfo))
        .containsExactly("com/google/example/complex/Complex.java");
    assertThat(complexRuleIdeInfo.getDependenciesList())
        .containsExactly("//com/google/example:simple");
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
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:extracomplex");
    assertThat(ruleIdeInfos.size()).isEqualTo(3);

    getRuleInfoAndVerifyLabel("//com/google/example:simple", ruleIdeInfos);
    getRuleInfoAndVerifyLabel("//com/google/example:complex", ruleIdeInfos);

    RuleIdeInfo extraComplexRuleIdeInfo = getRuleInfoAndVerifyLabel(
        "//com/google/example:extracomplex", ruleIdeInfos);

    assertThat(relativePathsForJavaSourcesOf(extraComplexRuleIdeInfo))
        .containsExactly("com/google/example/extracomplex/ExtraComplex.java");
    assertThat(extraComplexRuleIdeInfo.getDependenciesList())
        .containsExactly("//com/google/example:complex");

    assertThat(getIdeResolveFiles()).containsExactly(
        "com/google/example/libextracomplex.jar",
        "com/google/example/libextracomplex-ijar.jar",
        "com/google/example/libextracomplex-src.jar",
        "com/google/example/libcomplex.jar",
        "com/google/example/libcomplex-ijar.jar",
        "com/google/example/libcomplex-src.jar",
        "com/google/example/libsimple.jar",
        "com/google/example/libsimple-ijar.jar",
        "com/google/example/libsimple-src.jar"
    );
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
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:extracomplex");
    assertThat(ruleIdeInfos.size()).isEqualTo(4);

    getRuleInfoAndVerifyLabel("//com/google/example:simple", ruleIdeInfos);
    getRuleInfoAndVerifyLabel("//com/google/example:complex", ruleIdeInfos);
    getRuleInfoAndVerifyLabel("//com/google/example:complex1", ruleIdeInfos);

    RuleIdeInfo extraComplexRuleIdeInfo = getRuleInfoAndVerifyLabel(
        "//com/google/example:extracomplex", ruleIdeInfos);

    assertThat(relativePathsForJavaSourcesOf(extraComplexRuleIdeInfo))
        .containsExactly("com/google/example/extracomplex/ExtraComplex.java");
    assertThat(extraComplexRuleIdeInfo.getDependenciesList())
        .containsExactly("//com/google/example:complex", "//com/google/example:complex1");
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
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:extracomplex");
    assertThat(ruleIdeInfos.size()).isEqualTo(3);

    getRuleInfoAndVerifyLabel("//com/google/example:simple", ruleIdeInfos);
    getRuleInfoAndVerifyLabel("//com/google/example:complex", ruleIdeInfos);

    RuleIdeInfo complexRuleIdeInfo = getRuleInfoAndVerifyLabel("//com/google/example:complex",
        ruleIdeInfos);
    RuleIdeInfo extraComplexRuleIdeInfo = getRuleInfoAndVerifyLabel(
        "//com/google/example:extracomplex", ruleIdeInfos);

    assertThat(complexRuleIdeInfo.getDependenciesList())
        .containsExactly("//com/google/example:simple");

    assertThat(extraComplexRuleIdeInfo.getDependenciesList())
        .containsExactly("//com/google/example:simple", "//com/google/example:complex")
        .inOrder();
    assertThat(getIdeResolveFiles()).containsExactly(
        "com/google/example/libextracomplex.jar",
        "com/google/example/libextracomplex-ijar.jar",
        "com/google/example/libextracomplex-src.jar",
        "com/google/example/libcomplex.jar",
        "com/google/example/libcomplex-ijar.jar",
        "com/google/example/libcomplex-src.jar",
        "com/google/example/libsimple.jar",
        "com/google/example/libsimple-ijar.jar",
        "com/google/example/libsimple-src.jar"
    );
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
        ")"
    );
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:megacomplex");
    assertThat(ruleIdeInfos.size()).isEqualTo(4);

    getRuleInfoAndVerifyLabel("//com/google/example:simple", ruleIdeInfos);
    getRuleInfoAndVerifyLabel("//com/google/example:complex", ruleIdeInfos);
    getRuleInfoAndVerifyLabel("//com/google/example:extracomplex", ruleIdeInfos);

    RuleIdeInfo megaComplexRuleIdeInfo = getRuleInfoAndVerifyLabel(
        "//com/google/example:megacomplex", ruleIdeInfos);

    assertThat(relativePathsForJavaSourcesOf(megaComplexRuleIdeInfo))
        .containsExactly("com/google/example/megacomplex/MegaComplex.java");
    assertThat(megaComplexRuleIdeInfo.getDependenciesList())
        .containsExactly(
            "//com/google/example:simple",
            "//com/google/example:complex",
            "//com/google/example:extracomplex")
        .inOrder();
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

    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:lib");
    final RuleIdeInfo libInfo = getRuleInfoAndVerifyLabel("//com/google/example:lib", ruleIdeInfos);
    RuleIdeInfo impInfo = getRuleInfoAndVerifyLabel("//com/google/example:imp", ruleIdeInfos);
    assertThat(impInfo.getKind()).isEqualTo(Kind.JAVA_IMPORT);
    assertThat(impInfo.getKindString()).isEqualTo("java_import");
    assertThat(libInfo.getDependenciesList()).containsExactly("//com/google/example:imp");

    JavaRuleIdeInfo javaRuleIdeInfo = impInfo.getJavaRuleIdeInfo();
    assertThat(javaRuleIdeInfo).isNotNull();
    assertThat(transform(javaRuleIdeInfo.getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(
            jarString("com/google/example",
                "a.jar", "_ijar/imp/com/google/example/a-ijar.jar", "impsrc.jar"),
            jarString("com/google/example",
                "b.jar", "_ijar/imp/com/google/example/b-ijar.jar", "impsrc.jar"))
        .inOrder();

    assertThat(getIdeResolveFiles()).containsExactly(
        "com/google/example/_ijar/imp/com/google/example/a-ijar.jar",
        "com/google/example/_ijar/imp/com/google/example/b-ijar.jar",
        "com/google/example/liblib.jar",
        "com/google/example/liblib-ijar.jar",
        "com/google/example/liblib-src.jar"
    );
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

    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:lib");
    RuleIdeInfo libInfo = getRuleInfoAndVerifyLabel("//com/google/example:lib", ruleIdeInfos);
    RuleIdeInfo impInfo = getRuleInfoAndVerifyLabel("//com/google/example:imp", ruleIdeInfos);

    assertThat(impInfo.getKind()).isEqualTo(Kind.JAVA_IMPORT);
    assertThat(impInfo.getKindString()).isEqualTo("java_import");
    assertThat(impInfo.getDependenciesList()).containsExactly("//com/google/example:foobar");
    assertThat(libInfo.getDependenciesList())
        .containsExactly("//com/google/example:foobar", "//com/google/example:imp")
        .inOrder();
  }

  @Test
  public void testNoPackageManifestForExports() throws Exception {
    if (!isNativeTest()) {
      return;
    }

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
    
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:lib");
    RuleIdeInfo libInfo = getRuleInfoAndVerifyLabel("//com/google/example:lib", ruleIdeInfos);
    RuleIdeInfo impInfo = getRuleInfoAndVerifyLabel("//com/google/example:imp", ruleIdeInfos);
   
    assertThat(!impInfo.getJavaRuleIdeInfo().hasPackageManifest()).isTrue();
    assertThat(libInfo.getJavaRuleIdeInfo().hasPackageManifest()).isTrue();
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
    buildTarget("//com/google/example:imp");
    assertThat(getIdeResolveFiles()).containsExactly(
        "com/google/example/_ijar/imp/com/google/example/gen_jar-ijar.jar",
        "com/google/example/gen_jar.jar",
        "com/google/example/gen_srcjar.jar"
    );
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

    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:lib");
    getRuleInfoAndVerifyLabel("//com/google/example:foobar", ruleIdeInfos);
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
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo(
        "//java/com/google/example:FooBarTest");
    RuleIdeInfo testInfo = getRuleInfoAndVerifyLabel(
        "//java/com/google/example:FooBarTest", ruleIdeInfos);
    assertThat(testInfo.getKind()).isEqualTo(Kind.JAVA_TEST);
    assertThat(testInfo.getKindString()).isEqualTo("java_test");
    assertThat(relativePathsForJavaSourcesOf(testInfo))
        .containsExactly("java/com/google/example/FooBarTest.java");
    assertThat(testInfo.getDependenciesList()).contains("//java/com/google/example:foobar");
    assertThat(testInfo.getDependenciesList()).hasSize(2);
    assertThat(transform(testInfo.getJavaRuleIdeInfo().getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(jarString("java/com/google/example",
            "FooBarTest.jar", null, "FooBarTest-src.jar"));

    assertThat(getIdeResolveFiles()).containsExactly(
        "java/com/google/example/libfoobar.jar",
        "java/com/google/example/libfoobar-ijar.jar",
        "java/com/google/example/libfoobar-src.jar",
        "java/com/google/example/FooBarTest.jar",
        "java/com/google/example/FooBarTest-src.jar"
    );
    assertThat(testInfo.getJavaRuleIdeInfo().getJdeps().getRelativePath())
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
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:foobar-exe");
    RuleIdeInfo binaryInfo = getRuleInfoAndVerifyLabel(
        "//com/google/example:foobar-exe", ruleIdeInfos);
    assertThat(binaryInfo.getKind()).isEqualTo(Kind.JAVA_BINARY);
    assertThat(binaryInfo.getKindString()).isEqualTo("java_binary");
    assertThat(relativePathsForJavaSourcesOf(binaryInfo))
        .containsExactly("com/google/example/FooBarMain.java");
    assertThat(binaryInfo.getDependenciesList()).contains("//com/google/example:foobar");
    assertThat(binaryInfo.getDependenciesList()).hasSize(2);
    assertThat(transform(binaryInfo.getJavaRuleIdeInfo().getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(jarString("com/google/example",
            "foobar-exe.jar", null, "foobar-exe-src.jar"));

    assertThat(getIdeResolveFiles()).containsExactly(
        "com/google/example/libfoobar.jar",
        "com/google/example/libfoobar-ijar.jar",
        "com/google/example/libfoobar-src.jar",
        "com/google/example/foobar-exe.jar",
        "com/google/example/foobar-exe-src.jar"
    );
    assertThat(binaryInfo.getJavaRuleIdeInfo().getJdeps().getRelativePath())
        .isEqualTo("com/google/example/foobar-exe.jdeps");
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
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:l");
    RuleIdeInfo ruleInfo = getRuleInfoAndVerifyLabel("//com/google/example:l", ruleIdeInfos);
    assertThat(ruleInfo.getKind()).isEqualTo(Kind.ANDROID_LIBRARY);
    assertThat(ruleInfo.getKindString()).isEqualTo("android_library");
    assertThat(relativePathsForJavaSourcesOf(ruleInfo)).containsExactly("com/google/example/Main.java");
    assertThat(transform(ruleInfo.getJavaRuleIdeInfo().getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(
            jarString("com/google/example",
                "libl.jar", "libl-ijar.jar", "libl-src.jar"),
            jarString("com/google/example",
                "l_resources.jar", "l_resources-ijar.jar", "l_resources-src.jar"));
    assertThat(
            transform(
                ruleInfo.getAndroidRuleIdeInfo().getResourcesList(), ARTIFACT_TO_RELATIVE_PATH))
        .containsExactly("com/google/example/res");
    assertThat(ruleInfo.getAndroidRuleIdeInfo().getManifest().getRelativePath())
        .isEqualTo("com/google/example/AndroidManifest.xml");
    assertThat(ruleInfo.getAndroidRuleIdeInfo().getJavaPackage()).isEqualTo("com.google.example");

    assertThat(ruleInfo.getDependenciesList()).containsExactly("//com/google/example:l1");
    assertThat(getIdeResolveFiles()).containsExactly(
        "com/google/example/libl.jar",
        "com/google/example/libl-ijar.jar",
        "com/google/example/libl-src.jar",
        "com/google/example/l_resources.jar",
        "com/google/example/l_resources-ijar.jar",
        "com/google/example/l_resources-src.jar",
        "com/google/example/libl1.jar",
        "com/google/example/libl1-src.jar",
        "com/google/example/l1_resources.jar",
        "com/google/example/l1_resources-ijar.jar",
        "com/google/example/l1_resources-src.jar"
    );
    assertThat(ruleInfo.getJavaRuleIdeInfo().getJdeps().getRelativePath())
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
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:b");
    RuleIdeInfo ruleInfo = getRuleInfoAndVerifyLabel("//com/google/example:b", ruleIdeInfos);
    assertThat(ruleInfo.getKind()).isEqualTo(Kind.ANDROID_BINARY);
    assertThat(ruleInfo.getKindString()).isEqualTo("android_binary");
    assertThat(relativePathsForJavaSourcesOf(ruleInfo)).containsExactly("com/google/example/Main.java");
    assertThat(transform(ruleInfo.getJavaRuleIdeInfo().getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(
            jarString("com/google/example",
                "libb.jar", "libb-ijar.jar", "libb-src.jar"),
            jarString("com/google/example",
                "b_resources.jar", "b_resources-ijar.jar", "b_resources-src.jar"));

    assertThat(
            transform(
                ruleInfo.getAndroidRuleIdeInfo().getResourcesList(), ARTIFACT_TO_RELATIVE_PATH))
        .containsExactly("com/google/example/res");
    assertThat(ruleInfo.getAndroidRuleIdeInfo().getManifest().getRelativePath())
        .isEqualTo("com/google/example/AndroidManifest.xml");
    assertThat(ruleInfo.getAndroidRuleIdeInfo().getJavaPackage()).isEqualTo("com.google.example");
    assertThat(ruleInfo.getAndroidRuleIdeInfo().getApk().getRelativePath())
        .isEqualTo("com/google/example/b.apk");


    assertThat(ruleInfo.getDependenciesList()).contains("//com/google/example:l1");
    assertThat(ruleInfo.getDependenciesList()).hasSize(2);

    assertThat(getIdeResolveFiles()).containsExactly(
        "com/google/example/libb.jar",
        "com/google/example/libb-ijar.jar",
        "com/google/example/libb-src.jar",
        "com/google/example/b_resources.jar",
        "com/google/example/b_resources-ijar.jar",
        "com/google/example/b_resources-src.jar",
        "com/google/example/libl1.jar",
        "com/google/example/libl1-src.jar",
        "com/google/example/l1_resources.jar",
        "com/google/example/l1_resources-ijar.jar",
        "com/google/example/l1_resources-src.jar"
    );
    assertThat(ruleInfo.getJavaRuleIdeInfo().getJdeps().getRelativePath())
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
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//java/com/google/example:b");
    RuleIdeInfo lRuleInfo = getRuleInfoAndVerifyLabel("//java/com/google/example:l", ruleIdeInfos);
    RuleIdeInfo bRuleInfo = getRuleInfoAndVerifyLabel("//java/com/google/example:b", ruleIdeInfos);

    assertThat(bRuleInfo.getAndroidRuleIdeInfo().getJavaPackage()).isEqualTo("com.google.example");
    assertThat(lRuleInfo.getAndroidRuleIdeInfo().getJavaPackage()).isEqualTo("com.google.example");
  }

  @Test
  public void testAndroidLibraryWithoutAidlHasNoIdlJars() throws Exception {
    scratch.file(
        "java/com/google/example/BUILD",
        "android_library(",
        "  name = 'no_idl',",
        "  srcs = ['Test.java'],",
        ")"
    );
    String noIdlTarget = "//java/com/google/example:no_idl";
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo(noIdlTarget);
    RuleIdeInfo noIdlRuleInfo = getRuleInfoAndVerifyLabel(noIdlTarget, ruleIdeInfos);

    assertThat(noIdlRuleInfo.getAndroidRuleIdeInfo().getHasIdlSources()).isFalse();
  }

  @Test
  public void testAndroidLibraryWithAidlHasIdlJars() throws Exception {
    scratch.file(
        "java/com/google/example/BUILD",
        "android_library(",
        "  name = 'has_idl',",
        "  idl_srcs = ['a.aidl'],",
        ")"
    );
    String idlTarget = "//java/com/google/example:has_idl";
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo(idlTarget);
    RuleIdeInfo idlRuleInfo = getRuleInfoAndVerifyLabel(idlTarget, ruleIdeInfos);

    assertThat(idlRuleInfo.getAndroidRuleIdeInfo().getHasIdlSources()).isTrue();
    assertThat(LIBRARY_ARTIFACT_TO_STRING.apply(idlRuleInfo.getAndroidRuleIdeInfo().getIdlJar()))
        .isEqualTo(jarString("java/com/google/example",
            "libhas_idl-idl.jar", null, "libhas_idl-idl.srcjar"));
    assertThat(relativePathsForJavaSourcesOf(idlRuleInfo))
        .isEmpty();
    assertThat(getIdeResolveFiles()).containsExactly(
        "java/com/google/example/libhas_idl.jar",
        "java/com/google/example/libhas_idl-ijar.jar",
        "java/com/google/example/libhas_idl-src.jar",
        "java/com/google/example/libhas_idl-idl.jar",
        "java/com/google/example/libhas_idl-idl.srcjar"
    );
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
    buildTarget("//com/google/example:lib");
    assertThat(getIdeResolveFiles()).containsExactly(
        "com/google/example/liblib.jar",
        "com/google/example/liblib-src.jar",
        "com/google/example/lib_resources.jar",
        "com/google/example/lib_resources-ijar.jar",
        "com/google/example/lib_resources-src.jar",
        "com/google/example/AndroidManifest.xml"
    );
  }

  @Test
  public void testJavaLibraryWithoutGeneratedSourcesHasNoGenJars() throws Exception {
    scratch.file(
        "java/com/google/example/BUILD",
        "java_library(",
        "  name = 'no_plugin',",
        "  srcs = ['Test.java'],",
        ")"
    );
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//java/com/google/example:no_plugin");
    RuleIdeInfo ruleIdeInfo = getRuleInfoAndVerifyLabel(
        "//java/com/google/example:no_plugin", ruleIdeInfos);

    assertThat(ruleIdeInfo.getJavaRuleIdeInfo().getGeneratedJarsList())
        .isEmpty();
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
        ")"
    );
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//java/com/google/example:test");
    RuleIdeInfo ruleIdeInfo = getRuleInfoAndVerifyLabel(
        "//java/com/google/example:test", ruleIdeInfos);

    assertThat(transform(
        ruleIdeInfo.getJavaRuleIdeInfo().getGeneratedJarsList(),
        LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(jarString("java/com/google/example",
            "libtest-gen.jar", null, "libtest-gensrc.jar"));
    assertThat(getIdeResolveFiles()).containsExactly(
        "java/com/google/example/libtest.jar",
        "java/com/google/example/libtest-ijar.jar",
        "java/com/google/example/libtest-src.jar",
        "java/com/google/example/libtest-gen.jar",
        "java/com/google/example/libtest-gensrc.jar"
    );
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
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:lib");
    RuleIdeInfo ruleIdeInfo = getRuleInfoAndVerifyLabel("//com/google/example:lib", ruleIdeInfos);
    assertThat(ruleIdeInfo.getTagsList())
        .containsExactly("a", "b", "c", "d");
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
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//java/com/google/example:super");
    RuleIdeInfo ruleInfo = getRuleInfoAndVerifyLabel(
        "//java/com/google/example:super", ruleIdeInfos);

    assertThat(ruleInfo.getDependenciesList()).containsExactly(
        "//java/com/google/example:forward",
        "//java/com/google/example:lib");
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
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:lib");
    RuleIdeInfo ruleIdeInfo = getRuleInfoAndVerifyLabel("//com/google/example:lib", ruleIdeInfos);
    // todo(dslomov): Skylark aspect implementation does not yet return a correct root path.
    assertThat(ruleIdeInfo.getJavaRuleIdeInfo().getSourcesList()).containsExactly(
        ArtifactLocation.newBuilder()
            .setRootPath(
                isNativeTest() ? targetConfig.getGenfilesDirectory().getPath().getPathString() : "")
            .setRootExecutionPathFragment(
                targetConfig.getGenfilesDirectory().getExecPathString())
            .setRelativePath("com/google/example/gen.java")
            .setIsSource(false)
            .build(),
        ArtifactLocation.newBuilder()
            .setRootPath(isNativeTest() ? directories.getWorkspace().getPathString() : "")
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

    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:lib");
    // Fails if aspect was not propagated
    getRuleInfoAndVerifyLabel("//com/google/example:foobar", ruleIdeInfos);

    RuleIdeInfo libInfo = getRuleInfoAndVerifyLabel("//com/google/example:foobar", ruleIdeInfos);
    assertThat(libInfo.getDependenciesList()).isEmpty();
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

    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:lib");
    // Fails if aspect was not propagated
    RuleIdeInfo lib = getRuleInfoAndVerifyLabel("//com/google/example:lib", ruleIdeInfos);
    RuleIdeInfo lib2 = getRuleInfoAndVerifyLabel("//com/google/example:lib2", ruleIdeInfos);

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

    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//java/com/google/example:neither");
    RuleIdeInfo neither = getRuleInfoAndVerifyLabel(
        "//java/com/google/example:neither", ruleIdeInfos);
    RuleIdeInfo resourceFiles = getRuleInfoAndVerifyLabel(
        "//java/com/google/example:resource_files", ruleIdeInfos);
    RuleIdeInfo manifest = getRuleInfoAndVerifyLabel(
        "//java/com/google/example:manifest", ruleIdeInfos);

    assertThat(neither.getAndroidRuleIdeInfo().getGenerateResourceClass()).isFalse();
    assertThat(resourceFiles.getAndroidRuleIdeInfo().getGenerateResourceClass()).isTrue();
    assertThat(manifest.getAndroidRuleIdeInfo().getGenerateResourceClass()).isTrue();
  }

  @Test
  public void testJavaPlugin() throws Exception {
    scratch.file(
        "java/com/google/example/BUILD",
        "java_plugin(",
        "  name = 'plugin',",
        "  srcs = ['Plugin.java'],",
        "  processor_class = 'com.google.example.Plugin',",
        ")"
    );
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//java/com/google/example:plugin");
    RuleIdeInfo plugin = getRuleInfoAndVerifyLabel(
        "//java/com/google/example:plugin", ruleIdeInfos);

    assertThat(plugin.getKind()).isEqualTo(Kind.JAVA_PLUGIN);
    assertThat(plugin.getKindString()).isEqualTo("java_plugin");
    assertThat(transform(
        plugin.getJavaRuleIdeInfo().getJarsList(),
        LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(jarString("java/com/google/example",
            "libplugin.jar", "libplugin-ijar.jar", "libplugin-src.jar"));
  }

  @Test
  public void testSimpleCCLibraryForCCToolchainExistence() throws Exception {
    Path buildFilePath =
        scratch.file(
            "com/google/example/BUILD",
            "cc_library(",
            "    name = 'simple',",
            "    srcs = ['simple/simple.cc'],",
            "    hdrs = ['simple/simple.h'],",
            ")");
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:simple");
    assertThat(ruleIdeInfos.size()).isEqualTo(2);
    RuleIdeInfo ruleInfo = getRuleInfoAndVerifyLabel(
        "//com/google/example:simple",
        ruleIdeInfos
    );
    Entry<String, RuleIdeInfo> toolchainEntry = getCcToolchainRuleAndVerifyThereIsOnlyOne(
        ruleIdeInfos);
    RuleIdeInfo toolchainInfo = toolchainEntry.getValue();
    if (isNativeTest()) {
      assertThat(ruleInfo.getBuildFile()).isEqualTo(buildFilePath.toString());
      ArtifactLocation location = ruleInfo.getBuildFileArtifactLocation();
      assertThat(Paths.get(location.getRootPath(), location.getRelativePath()).toString())
          .isEqualTo(buildFilePath.toString());
      assertThat(location.getRelativePath()).isEqualTo("com/google/example/BUILD");
    }

    assertThat(ruleInfo.hasCRuleIdeInfo()).isTrue();
    assertThat(ruleInfo.getDependenciesList()).hasSize(1);
    assertThat(toolchainInfo.hasCToolchainIdeInfo()).isTrue();
  }

  @Test
  public void testSimpleCCLibrary() throws Exception {
    Path buildFilePath =
        scratch.file(
            "com/google/example/BUILD",
            "cc_library(",
            "    name = 'simple',",
            "    srcs = ['simple/simple.cc'],",
            "    hdrs = ['simple/simple.h'],",
            ")");
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:simple");
    assertThat(ruleIdeInfos.size()).isEqualTo(2);
    RuleIdeInfo ruleIdeInfo = getRuleInfoAndVerifyLabel(
        "//com/google/example:simple", ruleIdeInfos);
    if (isNativeTest()) {
      assertThat(ruleIdeInfo.getBuildFile()).isEqualTo(buildFilePath.toString());
      ArtifactLocation location = ruleIdeInfo.getBuildFileArtifactLocation();
      assertThat(Paths.get(location.getRootPath(), location.getRelativePath()).toString())
          .isEqualTo(buildFilePath.toString());
      assertThat(location.getRelativePath()).isEqualTo("com/google/example/BUILD");
    }
    assertThat(ruleIdeInfo.getKind()).isEqualTo(Kind.CC_LIBRARY);
    assertThat(ruleIdeInfo.getKindString()).isEqualTo("cc_library");
    assertThat(ruleIdeInfo.getDependenciesCount()).isEqualTo(1);

    assertThat(relativePathsForCSourcesOf(ruleIdeInfo))
        .containsExactly("com/google/example/simple/simple.cc");
    assertThat(relativePathsForExportedCHeadersOf(ruleIdeInfo))
        .containsExactly("com/google/example/simple/simple.h");

    assertThat(ruleIdeInfo.hasCRuleIdeInfo()).isTrue();
    assertThat(ruleIdeInfo.hasJavaRuleIdeInfo()).isFalse();
    assertThat(ruleIdeInfo.hasAndroidRuleIdeInfo()).isFalse();
    CRuleIdeInfo cRuleIdeInfo = ruleIdeInfo.getCRuleIdeInfo();

    assertThat(cRuleIdeInfo.getRuleCoptList()).isEmpty();
    assertThat(cRuleIdeInfo.getRuleDefineList()).isEmpty();
    assertThat(cRuleIdeInfo.getRuleIncludeList()).isEmpty();

    ProtocolStringList transQuoteIncludeDirList =
        cRuleIdeInfo.getTransitiveQuoteIncludeDirectoryList();
    assertThat(transQuoteIncludeDirList).contains(".");

    assertThat(ruleIdeInfo.getJavaRuleIdeInfo().getJarsList()).isEmpty();

    assertThat(getIdeResolveFiles()).isEmpty();
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
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:simple");
    assertThat(ruleIdeInfos.size()).isEqualTo(2);
    RuleIdeInfo ruleIdeInfo = getRuleInfoAndVerifyLabel(
        "//com/google/example:simple", ruleIdeInfos);

    assertThat(ruleIdeInfo.hasCRuleIdeInfo()).isTrue();
    CRuleIdeInfo cRuleIdeInfo = ruleIdeInfo.getCRuleIdeInfo();

    assertThat(cRuleIdeInfo.getRuleIncludeList()).containsExactly("foo/bar");

    // Make sure our understanding of where this attributes show up in other providers is correct.
    Entry<String, RuleIdeInfo> toolchainEntry = getCcToolchainRuleAndVerifyThereIsOnlyOne(
        ruleIdeInfos);
    RuleIdeInfo toolchainInfo = toolchainEntry.getValue();
    assertThat(toolchainInfo.hasCToolchainIdeInfo()).isTrue();
    CToolchainIdeInfo cToolchainIdeInfo = toolchainInfo.getCToolchainIdeInfo();
    ProtocolStringList builtInIncludeDirectoryList =
        cToolchainIdeInfo.getBuiltInIncludeDirectoryList();
    assertThat(builtInIncludeDirectoryList).doesNotContain("foo/bar");
    assertThat(builtInIncludeDirectoryList).doesNotContain("com/google/example/foo/bar");

    ProtocolStringList transIncludeDirList = cRuleIdeInfo.getTransitiveIncludeDirectoryList();
    assertThat(transIncludeDirList).doesNotContain("foo/bar");
    assertThat(transIncludeDirList).doesNotContain("com/google/example/foo/bar");

    ProtocolStringList transQuoteIncludeDirList =
        cRuleIdeInfo.getTransitiveQuoteIncludeDirectoryList();
    assertThat(transQuoteIncludeDirList).doesNotContain("foo/bar");
    assertThat(transQuoteIncludeDirList).doesNotContain("com/google/example/foo/bar");

    ProtocolStringList transSysIncludeDirList =
        cRuleIdeInfo.getTransitiveSystemIncludeDirectoryList();
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
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:simple");
    assertThat(ruleIdeInfos.size()).isEqualTo(2);
    RuleIdeInfo ruleIdeInfo = getRuleInfoAndVerifyLabel(
        "//com/google/example:simple", ruleIdeInfos);

    assertThat(ruleIdeInfo.hasCRuleIdeInfo()).isTrue();
    CRuleIdeInfo cRuleIdeInfo = ruleIdeInfo.getCRuleIdeInfo();

    assertThat(cRuleIdeInfo.getRuleCoptList()).containsExactly("-DGOPT", "-Ifoo/baz/");

    // Make sure our understanding of where this attributes show up in other providers is correct.
    Entry<String, RuleIdeInfo> toolchainEntry = getCcToolchainRuleAndVerifyThereIsOnlyOne(
        ruleIdeInfos);
    RuleIdeInfo toolchainInfo = toolchainEntry.getValue();
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
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:simple");
    assertThat(ruleIdeInfos.size()).isEqualTo(2);
    RuleIdeInfo ruleIdeInfo = getRuleInfoAndVerifyLabel(
        "//com/google/example:simple", ruleIdeInfos);

    assertThat(ruleIdeInfo.hasCRuleIdeInfo()).isTrue();
    CRuleIdeInfo cRuleIdeInfo = ruleIdeInfo.getCRuleIdeInfo();

    assertThat(cRuleIdeInfo.getRuleDefineList()).containsExactly("VERSION2");

    // Make sure our understanding of where this attributes show up in other providers is correct.
    ProtocolStringList transDefineList = cRuleIdeInfo.getTransitiveDefineList();
    assertThat(transDefineList).contains("VERSION2");
  }

  @Test
  public void testSimpleCCBinary() throws Exception {
    Path buildFilePath =
        scratch.file(
            "com/google/example/BUILD",
            "cc_binary(",
            "    name = 'simple',",
            "    srcs = ['simple/simple.cc'],",
            ")");
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:simple");
    assertThat(ruleIdeInfos.size()).isEqualTo(2);
    RuleIdeInfo ruleIdeInfo = getRuleInfoAndVerifyLabel(
        "//com/google/example:simple", ruleIdeInfos);
    if (isNativeTest()) {
      assertThat(ruleIdeInfo.getBuildFile()).isEqualTo(buildFilePath.toString());
      ArtifactLocation location = ruleIdeInfo.getBuildFileArtifactLocation();
      assertThat(Paths.get(location.getRootPath(), location.getRelativePath()).toString())
          .isEqualTo(buildFilePath.toString());
      assertThat(location.getRelativePath()).isEqualTo("com/google/example/BUILD");
    }
    assertThat(ruleIdeInfo.getKind()).isEqualTo(Kind.CC_BINARY);
    assertThat(ruleIdeInfo.getKindString()).isEqualTo("cc_binary");
    assertThat(ruleIdeInfo.getDependenciesCount()).isEqualTo(1);

    assertThat(relativePathsForCSourcesOf(ruleIdeInfo))
        .containsExactly("com/google/example/simple/simple.cc");

    assertThat(ruleIdeInfo.hasCRuleIdeInfo()).isTrue();
    assertThat(ruleIdeInfo.hasJavaRuleIdeInfo()).isFalse();
    assertThat(ruleIdeInfo.hasAndroidRuleIdeInfo()).isFalse();
    CRuleIdeInfo cRuleIdeInfo = ruleIdeInfo.getCRuleIdeInfo();

    assertThat(cRuleIdeInfo.getRuleCoptList()).isEmpty();
    assertThat(cRuleIdeInfo.getRuleDefineList()).isEmpty();
    assertThat(cRuleIdeInfo.getRuleIncludeList()).isEmpty();

    assertThat(ruleIdeInfo.getJavaRuleIdeInfo().getJarsList()).isEmpty();

    assertThat(getIdeResolveFiles()).isEmpty();
  }

  @Test
  public void testSimpleCCTest() throws Exception {
    Path buildFilePath =
        scratch.file(
            "com/google/example/BUILD",
            "cc_test(",
            "    name = 'simple',",
            "    srcs = ['simple/simple.cc'],",
            ")");
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:simple");
    assertThat(ruleIdeInfos.size()).isEqualTo(2);
    RuleIdeInfo ruleIdeInfo = getRuleInfoAndVerifyLabel(
        "//com/google/example:simple", ruleIdeInfos);
    if (isNativeTest()) {
      assertThat(ruleIdeInfo.getBuildFile()).isEqualTo(buildFilePath.toString());
      ArtifactLocation location = ruleIdeInfo.getBuildFileArtifactLocation();
      assertThat(Paths.get(location.getRootPath(), location.getRelativePath()).toString())
          .isEqualTo(buildFilePath.toString());
      assertThat(location.getRelativePath()).isEqualTo("com/google/example/BUILD");
    }
    assertThat(ruleIdeInfo.getKind()).isEqualTo(Kind.CC_TEST);
    assertThat(ruleIdeInfo.getKindString()).isEqualTo("cc_test");
    assertThat(ruleIdeInfo.getDependenciesCount()).isEqualTo(1);

    assertThat(relativePathsForCSourcesOf(ruleIdeInfo))
        .containsExactly("com/google/example/simple/simple.cc");

    assertThat(ruleIdeInfo.hasCRuleIdeInfo()).isTrue();
    assertThat(ruleIdeInfo.hasJavaRuleIdeInfo()).isFalse();
    assertThat(ruleIdeInfo.hasAndroidRuleIdeInfo()).isFalse();
    CRuleIdeInfo cRuleIdeInfo = ruleIdeInfo.getCRuleIdeInfo();

    assertThat(cRuleIdeInfo.getRuleCoptList()).isEmpty();
    assertThat(cRuleIdeInfo.getRuleDefineList()).isEmpty();
    assertThat(cRuleIdeInfo.getRuleIncludeList()).isEmpty();

    assertThat(ruleIdeInfo.getJavaRuleIdeInfo().getJarsList()).isEmpty();

    assertThat(getIdeResolveFiles()).isEmpty();
  }

  @Test
  public void testSimpleCCLibraryWithDeps() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "cc_library(",
        "   name = 'lib',",
        "   srcs = ['lib/lib.cc'],",
        "   hdrs = ['lib/lib.h'],",
        ")",
        "cc_library(",
        "    name = 'simple',",
        "    srcs = ['simple/simple.cc'],",
        "    hdrs = ['simple/simple.h'],",
        "    deps = [':lib'],",
        ")");
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:simple");
    assertThat(ruleIdeInfos.size()).isEqualTo(3);
    RuleIdeInfo ruleIdeInfo = getRuleInfoAndVerifyLabel(
        "//com/google/example:simple", ruleIdeInfos);

    assertThat(ruleIdeInfo.getDependenciesList()).contains("//com/google/example:lib");
    assertThat(ruleIdeInfo.getDependenciesList()).hasSize(2);
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
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:androidlib");
    assertThat(ruleIdeInfos.size()).isEqualTo(3);
    RuleIdeInfo ruleIdeInfo = getRuleInfoAndVerifyLabel(
        "//com/google/example:androidlib", ruleIdeInfos);

    assertThat(ruleIdeInfo.getDependenciesList()).containsExactly("//com/google/example:simple");
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
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:lib1");
    assertThat(ruleIdeInfos.size()).isEqualTo(3);
    RuleIdeInfo lib1 = getRuleInfoAndVerifyLabel(
        "//com/google/example:lib1", ruleIdeInfos);

    assertThat(lib1.hasCRuleIdeInfo()).isTrue();
    CRuleIdeInfo cRuleIdeInfo = lib1.getCRuleIdeInfo();

    assertThat(cRuleIdeInfo.getRuleIncludeList()).containsExactly("foo/bar");

    // Make sure our understanding of where this attributes show up in other providers is correct.
    Entry<String, RuleIdeInfo> toolchainEntry = getCcToolchainRuleAndVerifyThereIsOnlyOne(
        ruleIdeInfos);
    RuleIdeInfo toolchainInfo = toolchainEntry.getValue();
    assertThat(toolchainInfo.hasCToolchainIdeInfo()).isTrue();
    CToolchainIdeInfo cToolchainIdeInfo = toolchainInfo.getCToolchainIdeInfo();
    ProtocolStringList builtInIncludeDirectoryList =
        cToolchainIdeInfo.getBuiltInIncludeDirectoryList();
    assertThat(builtInIncludeDirectoryList).doesNotContain("foo/bar");
    assertThat(builtInIncludeDirectoryList).doesNotContain("baz/lib");
    assertThat(builtInIncludeDirectoryList).doesNotContain("com/google/example/foo/bar");
    assertThat(builtInIncludeDirectoryList).doesNotContain("com/google/example/baz/lib");

    ProtocolStringList transIncludeDirList = cRuleIdeInfo.getTransitiveIncludeDirectoryList();
    assertThat(transIncludeDirList).doesNotContain("foo/bar");
    assertThat(transIncludeDirList).doesNotContain("baz/lib");
    assertThat(transIncludeDirList).doesNotContain("com/google/example/foo/bar");
    assertThat(transIncludeDirList).doesNotContain("com/google/example/baz/lib");

    ProtocolStringList transQuoteIncludeDirList =
        cRuleIdeInfo.getTransitiveQuoteIncludeDirectoryList();
    assertThat(transQuoteIncludeDirList).doesNotContain("foo/bar");
    assertThat(transQuoteIncludeDirList).doesNotContain("baz/lib");
    assertThat(transQuoteIncludeDirList).doesNotContain("com/google/example/foo/bar");
    assertThat(transQuoteIncludeDirList).doesNotContain("com/google/example/baz/lib");

    ProtocolStringList transSysIncludeDirList =
        cRuleIdeInfo.getTransitiveSystemIncludeDirectoryList();
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
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:lib1");
    assertThat(ruleIdeInfos.size()).isEqualTo(3);
    RuleIdeInfo lib1 = getRuleInfoAndVerifyLabel(
        "//com/google/example:lib1", ruleIdeInfos);

    assertThat(lib1.hasCRuleIdeInfo()).isTrue();
    CRuleIdeInfo cRuleIdeInfo = lib1.getCRuleIdeInfo();

    assertThat(cRuleIdeInfo.getRuleCoptList()).containsExactly("-DGOPT", "-Ifoo/baz/");

    // Make sure our understanding of where this attributes show up in other providers is correct.
    Entry<String, RuleIdeInfo> toolchainEntry = getCcToolchainRuleAndVerifyThereIsOnlyOne(
        ruleIdeInfos);
    RuleIdeInfo toolchainInfo = toolchainEntry.getValue();
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
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:lib1");
    assertThat(ruleIdeInfos.size()).isEqualTo(3);
    RuleIdeInfo lib1 = getRuleInfoAndVerifyLabel(
        "//com/google/example:lib1", ruleIdeInfos);

    assertThat(lib1.hasCRuleIdeInfo()).isTrue();
    CRuleIdeInfo cRuleIdeInfo = lib1.getCRuleIdeInfo();

    assertThat(cRuleIdeInfo.getRuleDefineList()).containsExactly("VERSION2");

    // Make sure our understanding of where this attributes show up in other providers is correct.
    ProtocolStringList transDefineList = cRuleIdeInfo.getTransitiveDefineList();
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
        "),");
    scratch.file(
        "java/com/google/example/BUILD",
        "load('//java/com/google/example:build_defs.bzl', 'my_macro')",
        "my_macro(",
        "    name = 'simple',",
        ")");
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//java/com/google/example:simple");
    RuleIdeInfo ruleIdeInfo = getRuleInfoAndVerifyLabel(
        "//java/com/google/example:simple", ruleIdeInfos);
    assertThat(ruleIdeInfo.getKind()).isEqualTo(Kind.ANDROID_BINARY);
    assertThat(ruleIdeInfo.getKindString()).isEqualTo("android_binary");
  }

  @Test
  public void testAndroidBinaryIsSerialized() throws Exception {
    RuleIdeInfo.Builder builder = RuleIdeInfo.newBuilder();
    builder.setKind(Kind.ANDROID_BINARY);
    builder.setKindString("android_binary");
    ByteString byteString = builder.build().toByteString();
    RuleIdeInfo result = RuleIdeInfo.parseFrom(byteString);
    assertThat(result.getKind()).isEqualTo(Kind.ANDROID_BINARY);
    assertThat(result.getKindString()).isEqualTo("android_binary");
  }

  @Test
  public void testCcToolchainInfoIsOnlyPresentForToolchainRules() throws Exception {
    Path buildFilePath =
        scratch.file(
            "com/google/example/BUILD",
            "cc_library(",
            "    name = 'simple',",
            "    srcs = ['simple/simple.cc'],",
            "    hdrs = ['simple/simple.h'],",
            ")");
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//com/google/example:simple");
    assertThat(ruleIdeInfos.size()).isEqualTo(2);
    RuleIdeInfo ruleInfo = getRuleInfoAndVerifyLabel(
        "//com/google/example:simple",
        ruleIdeInfos
    );
    Entry<String, RuleIdeInfo> toolchainEntry = getCcToolchainRuleAndVerifyThereIsOnlyOne(
        ruleIdeInfos);
    RuleIdeInfo toolchainInfo = toolchainEntry.getValue();
    if (isNativeTest()) {
      assertThat(ruleInfo.getBuildFile()).isEqualTo(buildFilePath.toString());
      ArtifactLocation location = ruleInfo.getBuildFileArtifactLocation();
      assertThat(Paths.get(location.getRootPath(), location.getRelativePath()).toString())
          .isEqualTo(buildFilePath.toString());
      assertThat(location.getRelativePath()).isEqualTo("com/google/example/BUILD");
    }

    assertThat(ruleInfo.hasCToolchainIdeInfo()).isFalse();
    assertThat(toolchainInfo.hasCToolchainIdeInfo()).isTrue();
  }

  /**
   * Returns true if we are testing the native aspect, not the Skylark one.
   * Eventually Skylark aspect will be equivalent to a native one, and this method
   * will be removed.
   */
  @Override
  protected boolean isNativeTest() {
    return true;
  }

  /**
   * Test for Skylark version of the aspect.
   */
  @RunWith(JUnit4.class)
  public static class IntelliJSkylarkAspectTest extends AndroidStudioInfoAspectTest {

    @Override
    public boolean isNativeTest() {
      return false;
    }
  }
}
