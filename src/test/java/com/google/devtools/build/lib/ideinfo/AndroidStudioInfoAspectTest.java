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

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.BuildView;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.ArtifactLocation;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.JavaRuleIdeInfo;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.RuleIdeInfo;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.RuleIdeInfo.Kind;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.TextFormat;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

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
    if (isNativeTest()) {
      assertThat(ruleIdeInfo.getBuildFile()).isEqualTo(buildFilePath.toString());
    }
    assertThat(ruleIdeInfo.getKind()).isEqualTo(Kind.JAVA_LIBRARY);
    assertThat(ruleIdeInfo.getDependenciesCount()).isEqualTo(0);
    assertThat(relativePathsForSourcesOf(ruleIdeInfo))
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

    assertThat(relativePathsForSourcesOf(complexRuleIdeInfo))
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

    assertThat(relativePathsForSourcesOf(extraComplexRuleIdeInfo))
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

    assertThat(relativePathsForSourcesOf(extraComplexRuleIdeInfo))
        .containsExactly("com/google/example/extracomplex/ExtraComplex.java");
    assertThat(extraComplexRuleIdeInfo.getDependenciesList())
        .containsExactly("//com/google/example:complex", "//com/google/example:complex1");
  }

  @Test
  public void testJavaLibraryWithExports() throws Exception {
    if (!isNativeTest()) {
      return;
    }

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
    if (!isNativeTest()) {
      return;
    }

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

    assertThat(relativePathsForSourcesOf(megaComplexRuleIdeInfo))
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

    assertThat(impInfo.getKind()).isEqualTo(Kind.JAVA_IMPORT);
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
    if (!isNativeTest()) {
      return;
    }

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
        "   deps = [':foobar'],",
        ")");
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo(
        "//java/com/google/example:FooBarTest");
    RuleIdeInfo testInfo = getRuleInfoAndVerifyLabel(
        "//java/com/google/example:FooBarTest", ruleIdeInfos);
    assertThat(testInfo.getKind()).isEqualTo(Kind.JAVA_TEST);
    assertThat(relativePathsForSourcesOf(testInfo))
        .containsExactly("java/com/google/example/FooBarTest.java");
    assertThat(testInfo.getDependenciesList())
        .containsExactly("//java/com/google/example:foobar");
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
    assertThat(relativePathsForSourcesOf(binaryInfo))
        .containsExactly("com/google/example/FooBarMain.java");
    assertThat(binaryInfo.getDependenciesList()).containsExactly("//com/google/example:foobar");
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
    assertThat(relativePathsForSourcesOf(ruleInfo)).containsExactly("com/google/example/Main.java");
    assertThat(transform(ruleInfo.getJavaRuleIdeInfo().getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(
            jarString("com/google/example",
                "libl.jar", "libl-ijar.jar", "libl-src.jar"),
            jarString("com/google/example",
                "l_resources.jar", "l_resources-ijar.jar", "l_resources-src.jar"));
    if (isNativeTest()) {
      assertThat(
          transform(
              ruleInfo.getAndroidRuleIdeInfo().getResourcesList(), ARTIFACT_TO_RELATIVE_PATH))
          .containsExactly("com/google/example/res");
      assertThat(ruleInfo.getAndroidRuleIdeInfo().getManifest().getRelativePath())
          .isEqualTo("com/google/example/AndroidManifest.xml");
      assertThat(ruleInfo.getAndroidRuleIdeInfo().getJavaPackage()).isEqualTo("com.google.example");
    }

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
    assertThat(relativePathsForSourcesOf(ruleInfo)).containsExactly("com/google/example/Main.java");
    assertThat(transform(ruleInfo.getJavaRuleIdeInfo().getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(
            jarString("com/google/example",
                "libb.jar", "libb-ijar.jar", "libb-src.jar"),
            jarString("com/google/example",
                "b_resources.jar", "b_resources-ijar.jar", "b_resources-src.jar"));

    if (isNativeTest()) {
      assertThat(
          transform(
              ruleInfo.getAndroidRuleIdeInfo().getResourcesList(), ARTIFACT_TO_RELATIVE_PATH))
          .containsExactly("com/google/example/res");
      assertThat(ruleInfo.getAndroidRuleIdeInfo().getManifest().getRelativePath())
          .isEqualTo("com/google/example/AndroidManifest.xml");
      assertThat(ruleInfo.getAndroidRuleIdeInfo().getJavaPackage()).isEqualTo("com.google.example");
      assertThat(ruleInfo.getAndroidRuleIdeInfo().getApk().getRelativePath())
          .isEqualTo("com/google/example/b.apk");
    }

    assertThat(ruleInfo.getDependenciesList()).containsExactly("//com/google/example:l1");

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
    if (!isNativeTest()) {
      return;
    }

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
    if (!isNativeTest()) {
      return;
    }

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
    if (!isNativeTest()) {
      return;
    }

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
    assertThat(relativePathsForSourcesOf(idlRuleInfo))
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
    if (!isNativeTest()) {
      return;
    }

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
  public void testNonConformingPackageName() throws Exception {
    if (!isNativeTest()) {
      return;
    }

    scratch.file(
        "bad/package/google/example/BUILD",
        "android_library(",
        "  name = 'test',",
        "  srcs = ['Test.java'],",
        ")"
    );
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo("//bad/package/google/example:test");
    RuleIdeInfo ruleInfo = getRuleInfoAndVerifyLabel(
        "//bad/package/google/example:test", ruleIdeInfos);

    assertThat(ruleInfo.getAndroidRuleIdeInfo().getJavaPackage())
        .isEqualTo("bad.package.google.example");
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
    if (!isNativeTest()) {
      return;
    }

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
    assertThat(ruleIdeInfo.getJavaRuleIdeInfo().getSourcesList())
        .containsExactly(
            expectedArtifactLocationWithRootPath(
                    targetConfig.getGenfilesDirectory().getPath().getPathString())
                .setRelativePath("com/google/example/gen.java")
                .setIsSource(false)
                .build(),
            expectedArtifactLocationWithRootPath(directories.getWorkspace().getPathString())
                .setRelativePath("com/google/example/Test.java")
                .setIsSource(true)
                .build());
  }

  private ArtifactLocation.Builder expectedArtifactLocationWithRootPath(String pathString) {
    if (isNativeTest()) {
      return ArtifactLocation.newBuilder().setRootPath(pathString);
    } else {
      // todo(dslomov): Skylark aspect implementation does not yet return a correct root path.
      return ArtifactLocation.newBuilder();
    }
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
    if (!isNativeTest()) {
      return;
    }

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
    if (!isNativeTest()) {
      return;
    }

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
    if (!isNativeTest()) {
      return;
    }

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
    assertThat(transform(
        plugin.getJavaRuleIdeInfo().getJarsList(),
        LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(jarString("java/com/google/example",
            "libplugin.jar", "libplugin-ijar.jar", "libplugin-src.jar"));
  }

  /**
   * Returns true if we are testing the native aspect, not the Skylark one.
   * Eventually Skylark aspect will be equivalent to a native one, and this method
   * will be removed.
   */
  public boolean isNativeTest() {
    return true;
  }

  /**
   * Test for Skylark version of the aspect.
   */
  @RunWith(JUnit4.class)
  public static class IntelliJSkylarkAspectTest extends AndroidStudioInfoAspectTest {
    @Before
    public void setupBzl() throws Exception {
      InputStream stream = IntelliJSkylarkAspectTest.class
          .getResourceAsStream("intellij_info.bzl");
      BufferedReader reader =
          new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8));
      String line;
      ArrayList<String> contents = new ArrayList<>();
      while ((line = reader.readLine()) != null) {
        contents.add(line);
      }

      scratch.file("intellij_tools/BUILD", "# empty");
      scratch.file("intellij_tools/intellij_info.bzl", contents.toArray(new String[0]));
    }

    @Override
    protected Map<String, RuleIdeInfo> buildRuleIdeInfo(String target) throws Exception {
      BuildView.AnalysisResult analysisResult = update(
          ImmutableList.of(target),
          ImmutableList.of("intellij_tools/intellij_info.bzl%intellij_info_aspect"),
          false,
          LOADING_PHASE_THREADS,
          true,
          new EventBus()
      );
      Collection<AspectValue> aspects = analysisResult.getAspects();
      assertThat(aspects).hasSize(1);
      AspectValue aspectValue = aspects.iterator().next();
      this.configuredAspect = aspectValue.getConfiguredAspect();
      OutputGroupProvider provider = configuredAspect.getProvider(OutputGroupProvider.class);
      NestedSet<Artifact> outputGroup = provider.getOutputGroup("ide-info-text");
      Map<String, RuleIdeInfo> ruleIdeInfos = new HashMap<>();
      for (Artifact artifact : outputGroup) {
        Action generatingAction = getGeneratingAction(artifact);
        assertThat(generatingAction).isInstanceOf(FileWriteAction.class);
        String fileContents = ((FileWriteAction) generatingAction).getFileContents();
        RuleIdeInfo.Builder builder = RuleIdeInfo.newBuilder();
        TextFormat.getParser().merge(fileContents, builder);
        RuleIdeInfo ruleIdeInfo = builder.build();
        ruleIdeInfos.put(ruleIdeInfo.getLabel(), ruleIdeInfo);
      }
      return ruleIdeInfos;
    }

    @Override
    public boolean isNativeTest() {
      return false;
    }
  }
}
