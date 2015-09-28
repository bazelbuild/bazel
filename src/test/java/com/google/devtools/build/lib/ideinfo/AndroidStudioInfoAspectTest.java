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

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.BuildView.AnalysisResult;
import com.google.devtools.build.lib.analysis.actions.BinaryFileWriteAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.ArtifactLocation;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.JavaRuleIdeInfo;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.LibraryArtifact;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.RuleIdeInfo;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.RuleIdeInfo.Kind;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.lib.vfs.Path;

import java.util.Collection;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * Tests for {@link AndroidStudioInfoAspect} validating proto's contents.
 */
public class AndroidStudioInfoAspectTest extends BuildViewTestCase {

  public static final Function<ArtifactLocation, String> ARTIFACT_TO_RELATIVE_PATH =
      new Function<ArtifactLocation, String>() {
        @Nullable
        @Override
        public String apply(ArtifactLocation artifactLocation) {
          return artifactLocation.getRelativePath();
        }
      };
  public static final Function<LibraryArtifact, String> LIBRARY_ARTIFACT_TO_STRING =
      new Function<LibraryArtifact, String>() {
        @Override
        public String apply(LibraryArtifact libraryArtifact) {
          StringBuilder stringBuilder = new StringBuilder();
          if (libraryArtifact.hasJar()) {
            stringBuilder.append("<jar:");
            stringBuilder.append(libraryArtifact.getJar().getRelativePath());
            stringBuilder.append(">");
          }
          if (libraryArtifact.hasInterfaceJar()) {
            stringBuilder.append("<ijar:");
            stringBuilder.append(libraryArtifact.getInterfaceJar().getRelativePath());
            stringBuilder.append(">");
          }
          if (libraryArtifact.hasSourceJar()) {
            stringBuilder.append("<source:");
            stringBuilder.append(libraryArtifact.getSourceJar().getRelativePath());
            stringBuilder.append(">");
          }

          return stringBuilder.toString();
        }
      };

  public void testSimpleJavaLibrary() throws Exception {
    Path buildFilePath =
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
    String target = "//com/google/example:simple";
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo(target);
    assertThat(ruleIdeInfos.size()).isEqualTo(1);
    RuleIdeInfo ruleIdeInfo = getRuleInfoAndVerifyLabel(target, ruleIdeInfos);
    assertThat(ruleIdeInfo.getBuildFile()).isEqualTo(buildFilePath.toString());
    assertThat(ruleIdeInfo.getKind()).isEqualTo(Kind.JAVA_LIBRARY);
    assertThat(ruleIdeInfo.getDependenciesCount()).isEqualTo(0);
    assertThat(relativePathsForSourcesOf(ruleIdeInfo))
        .containsExactly("com/google/example/simple/Simple.java");
    assertThat(
            transform(ruleIdeInfo.getJavaRuleIdeInfo().getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(
            "<jar:com/google/example/libsimple.jar><source:com/google/example/libsimple-src.jar>");
  }

  private static Iterable<String> relativePathsForSourcesOf(RuleIdeInfo ruleIdeInfo) {
    return transform(ruleIdeInfo.getJavaRuleIdeInfo().getSourcesList(), ARTIFACT_TO_RELATIVE_PATH);
  }

  private RuleIdeInfo getRuleInfoAndVerifyLabel(
      String target, Map<String, RuleIdeInfo> ruleIdeInfos) {
    RuleIdeInfo ruleIdeInfo = ruleIdeInfos.get(target);
    assertThat(ruleIdeInfo.getLabel()).isEqualTo(target);
    return ruleIdeInfo;
  }

  public void testJavaLibraryProtoWithDependencies() throws Exception {
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
    String target = "//com/google/example:complex";
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo(target);
    assertThat(ruleIdeInfos.size()).isEqualTo(2);

    getRuleInfoAndVerifyLabel("//com/google/example:simple", ruleIdeInfos);

    RuleIdeInfo complexRuleIdeInfo = getRuleInfoAndVerifyLabel(target, ruleIdeInfos);

    assertThat(relativePathsForSourcesOf(complexRuleIdeInfo))
        .containsExactly("com/google/example/complex/Complex.java");
    assertThat(complexRuleIdeInfo.getDependenciesList())
        .containsExactly("//com/google/example:simple");

    assertThat(complexRuleIdeInfo.getTransitiveDependenciesList())
        .containsExactly("//com/google/example:simple");
  }

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
    String target = "//com/google/example:extracomplex";
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo(target);
    assertThat(ruleIdeInfos.size()).isEqualTo(3);

    getRuleInfoAndVerifyLabel("//com/google/example:simple", ruleIdeInfos);
    getRuleInfoAndVerifyLabel("//com/google/example:complex", ruleIdeInfos);

    RuleIdeInfo extraComplexRuleIdeInfo = getRuleInfoAndVerifyLabel(target, ruleIdeInfos);

    assertThat(relativePathsForSourcesOf(extraComplexRuleIdeInfo))
        .containsExactly("com/google/example/extracomplex/ExtraComplex.java");
    assertThat(extraComplexRuleIdeInfo.getDependenciesList())
        .containsExactly("//com/google/example:complex");

    assertThat(extraComplexRuleIdeInfo.getTransitiveDependenciesList())
        .containsExactly("//com/google/example:complex", "//com/google/example:simple");
  }

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
    String target = "//com/google/example:extracomplex";
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo(target);
    assertThat(ruleIdeInfos.size()).isEqualTo(4);

    getRuleInfoAndVerifyLabel("//com/google/example:simple", ruleIdeInfos);
    getRuleInfoAndVerifyLabel("//com/google/example:complex", ruleIdeInfos);
    getRuleInfoAndVerifyLabel("//com/google/example:complex1", ruleIdeInfos);

    RuleIdeInfo extraComplexRuleIdeInfo = getRuleInfoAndVerifyLabel(target, ruleIdeInfos);

    assertThat(relativePathsForSourcesOf(extraComplexRuleIdeInfo))
        .containsExactly("com/google/example/extracomplex/ExtraComplex.java");
    assertThat(extraComplexRuleIdeInfo.getDependenciesList())
        .containsExactly("//com/google/example:complex", "//com/google/example:complex1");

    assertThat(extraComplexRuleIdeInfo.getTransitiveDependenciesList())
        .containsExactly(
            "//com/google/example:complex",
            "//com/google/example:complex1",
            "//com/google/example:simple");
  }

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
        "    deps = [':simple'],",
        "    exports = [':simple'],",
        ")",
        "java_library(",
        "    name = 'extracomplex',",
        "    srcs = ['extracomplex/ExtraComplex.java'],",
        "    deps = [':complex']",
        ")");
    String target = "//com/google/example:extracomplex";
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo(target);
    assertThat(ruleIdeInfos.size()).isEqualTo(3);

    getRuleInfoAndVerifyLabel("//com/google/example:simple", ruleIdeInfos);
    getRuleInfoAndVerifyLabel("//com/google/example:complex", ruleIdeInfos);

    RuleIdeInfo extraComplexRuleIdeInfo = getRuleInfoAndVerifyLabel(target, ruleIdeInfos);

    assertThat(relativePathsForSourcesOf(extraComplexRuleIdeInfo))
        .containsExactly("com/google/example/extracomplex/ExtraComplex.java");
    assertThat(extraComplexRuleIdeInfo.getDependenciesList())
        .containsExactly("//com/google/example:complex", "//com/google/example:simple");

    assertThat(extraComplexRuleIdeInfo.getTransitiveDependenciesList())
        .containsExactly(
            "//com/google/example:complex",
            "//com/google/example:simple");
  }

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
        "    deps = [':simple'],",
        "    exports = [':simple'],",
        ")",
        "java_library(",
        "    name = 'extracomplex',",
        "    srcs = ['extracomplex/ExtraComplex.java'],",
        "    deps = [':complex'],",
        "    exports = [':complex'],",
        ")",
        "java_library(",
        "    name = 'megacomplex',",
        "    srcs = ['megacomplex/MegaComplex.java'],",
        "    deps = [':extracomplex'],",
        ")"
    );
    String target = "//com/google/example:megacomplex";
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo(target);
    assertThat(ruleIdeInfos.size()).isEqualTo(4);

    getRuleInfoAndVerifyLabel("//com/google/example:simple", ruleIdeInfos);
    getRuleInfoAndVerifyLabel("//com/google/example:complex", ruleIdeInfos);
    getRuleInfoAndVerifyLabel("//com/google/example:extracomplex", ruleIdeInfos);

    RuleIdeInfo megaComplexRuleIdeInfo = getRuleInfoAndVerifyLabel(target, ruleIdeInfos);

    assertThat(relativePathsForSourcesOf(megaComplexRuleIdeInfo))
        .containsExactly("com/google/example/megacomplex/MegaComplex.java");
    assertThat(megaComplexRuleIdeInfo.getDependenciesList())
        .containsExactly(
            "//com/google/example:extracomplex",
            "//com/google/example:complex",
            "//com/google/example:simple");

    assertThat(megaComplexRuleIdeInfo.getTransitiveDependenciesList())
        .containsExactly(
            "//com/google/example:extracomplex",
            "//com/google/example:complex",
            "//com/google/example:simple");
  }

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
            "<jar:com/google/example/a.jar><source:com/google/example/impsrc.jar>",
            "<jar:com/google/example/b.jar><source:com/google/example/impsrc.jar>");
  }

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
    assertThat(impInfo.getDependenciesList()).containsExactly("//com/google/example:foobar");
    assertThat(libInfo.getDependenciesList())
        .containsExactly("//com/google/example:imp", "//com/google/example:foobar");
  }

  public void testJavaTest() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "java_library(",
        "   name = 'foobar',",
        "   srcs = ['FooBar.java'],",
        ")",
        "java_test(",
        "   name = 'foobar-test',",
        "   test_class = 'MyTestClass',",
        "   srcs = ['FooBarTest.java'],",
        "   deps = [':foobar'],",
        ")");
    String target = "//com/google/example:foobar-test";
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo(target);
    RuleIdeInfo testInfo = getRuleInfoAndVerifyLabel(target, ruleIdeInfos);
    assertThat(testInfo.getKind()).isEqualTo(Kind.JAVA_TEST);
    assertThat(relativePathsForSourcesOf(testInfo))
        .containsExactly("com/google/example/FooBarTest.java");
    assertThat(testInfo.getDependenciesList()).containsExactly("//com/google/example:foobar");
    assertThat(transform(testInfo.getJavaRuleIdeInfo().getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(
            "<jar:com/google/example/foobar-test.jar><source:com/google/example/foobar-test-src.jar>");
  }

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
    String target = "//com/google/example:foobar-exe";
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo(target);
    RuleIdeInfo binaryInfo = getRuleInfoAndVerifyLabel(target, ruleIdeInfos);
    assertThat(binaryInfo.getKind()).isEqualTo(Kind.JAVA_BINARY);
    assertThat(relativePathsForSourcesOf(binaryInfo))
        .containsExactly("com/google/example/FooBarMain.java");
    assertThat(binaryInfo.getDependenciesList()).containsExactly("//com/google/example:foobar");
    assertThat(transform(binaryInfo.getJavaRuleIdeInfo().getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(
            "<jar:com/google/example/foobar-exe.jar><source:com/google/example/foobar-exe-src.jar>");
  }

  public void testAndroidLibrary() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "android_library(",
        "  name = 'l1',",
        "  manifest = 'Manifesto.xml',",
        "  custom_package = 'com.google.example',",
        "  resource_files = ['r1/values/a.xml'],",
        ")",
        "android_library(",
        "  name = 'l',",
        "  srcs = ['Main.java'],",
        "  deps = [':l1'],",
        "  manifest = 'Abracadabra.xml',",
        "  custom_package = 'com.google.example',",
        "  resource_files = ['res/drawable/a.png', 'res/drawable/b.png'],",
        ")");
    String target = "//com/google/example:l";
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo(target);
    RuleIdeInfo ruleInfo = getRuleInfoAndVerifyLabel(target, ruleIdeInfos);
    assertThat(ruleInfo.getKind()).isEqualTo(Kind.ANDROID_LIBRARY);
    assertThat(relativePathsForSourcesOf(ruleInfo)).containsExactly("com/google/example/Main.java");
    assertThat(transform(ruleInfo.getJavaRuleIdeInfo().getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(
            "<jar:com/google/example/libl.jar><source:com/google/example/libl-src.jar>");
    assertThat(
            transform(
                ruleInfo.getAndroidRuleIdeInfo().getResourcesList(), ARTIFACT_TO_RELATIVE_PATH))
        .containsExactly("com/google/example/res");
    assertThat(ruleInfo.getAndroidRuleIdeInfo().getManifest().getRelativePath())
        .isEqualTo("com/google/example/Abracadabra.xml");
    assertThat(ruleInfo.getAndroidRuleIdeInfo().getJavaPackage()).isEqualTo("com.google.example");

    assertThat(ruleInfo.getDependenciesList()).containsExactly("//com/google/example:l1");
    assertThat(
            transform(
                ruleInfo.getAndroidRuleIdeInfo().getTransitiveResourcesList(),
                ARTIFACT_TO_RELATIVE_PATH))
        .containsExactly("com/google/example/res", "com/google/example/r1");
  }

  public void testAndroidBinary() throws Exception {
    scratch.file(
        "com/google/example/BUILD",
        "android_library(",
        "  name = 'l1',",
        "  manifest = 'Manifesto.xml',",
        "  custom_package = 'com.google.example',",
        "  resource_files = ['r1/values/a.xml'],",
        ")",
        "android_binary(",
        "  name = 'b',",
        "  srcs = ['Main.java'],",
        "  deps = [':l1'],",
        "  manifest = 'Abracadabra.xml',",
        "  custom_package = 'com.google.example',",
        "  resource_files = ['res/drawable/a.png', 'res/drawable/b.png'],",
        ")");
    String target = "//com/google/example:b";
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo(target);
    RuleIdeInfo ruleInfo = getRuleInfoAndVerifyLabel(target, ruleIdeInfos);
    assertThat(ruleInfo.getKind()).isEqualTo(Kind.ANDROID_BINARY);
    assertThat(relativePathsForSourcesOf(ruleInfo)).containsExactly("com/google/example/Main.java");
    assertThat(transform(ruleInfo.getJavaRuleIdeInfo().getJarsList(), LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(
            "<jar:com/google/example/libb.jar><source:com/google/example/libb-src.jar>");
    assertThat(
            transform(
                ruleInfo.getAndroidRuleIdeInfo().getResourcesList(), ARTIFACT_TO_RELATIVE_PATH))
        .containsExactly("com/google/example/res");
    assertThat(ruleInfo.getAndroidRuleIdeInfo().getManifest().getRelativePath())
        .isEqualTo("com/google/example/Abracadabra.xml");
    assertThat(ruleInfo.getAndroidRuleIdeInfo().getJavaPackage()).isEqualTo("com.google.example");

    assertThat(ruleInfo.getDependenciesList()).containsExactly("//com/google/example:l1");
    assertThat(
            transform(
                ruleInfo.getAndroidRuleIdeInfo().getTransitiveResourcesList(),
                ARTIFACT_TO_RELATIVE_PATH))
        .containsExactly("com/google/example/res", "com/google/example/r1");
  }

  public void testAndroidInferredPackage() throws Exception {
    scratch.file(
        "java/com/google/example/BUILD",
        "android_library(",
        "  name = 'l',",
        "  manifest = 'Manifesto.xml',",
        ")",
        "android_binary(",
        "  name = 'b',",
        "  srcs = ['Main.java'],",
        "  deps = [':l'],",
        "  manifest = 'Abracadabra.xml',",
        ")");
    String target = "//java/com/google/example:b";
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo(target);
    RuleIdeInfo lRuleInfo = getRuleInfoAndVerifyLabel("//java/com/google/example:l", ruleIdeInfos);
    RuleIdeInfo bRuleInfo = getRuleInfoAndVerifyLabel(target, ruleIdeInfos);

    assertThat(bRuleInfo.getAndroidRuleIdeInfo().getJavaPackage()).isEqualTo("com.google.example");
    assertThat(lRuleInfo.getAndroidRuleIdeInfo().getJavaPackage()).isEqualTo("com.google.example");
  }

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
        .isEqualTo(
                "<jar:java/com/google/example/libhas_idl-idl.jar>"
                + "<source:java/com/google/example/libhas_idl-idl.srcjar>");
  }
  
  public void testJavaLibraryWithoutGeneratedSourcesHasNoGenJars() throws Exception {
    scratch.file(
        "java/com/google/example/BUILD",
        "java_library(",
        "  name = 'no_plugin',",
        "  srcs = ['Test.java'],",
        ")"
    );
    String target = "//java/com/google/example:no_plugin";
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo(target);
    RuleIdeInfo ruleIdeInfo = getRuleInfoAndVerifyLabel(target, ruleIdeInfos);

    assertThat(ruleIdeInfo.getJavaRuleIdeInfo().getGeneratedJarsList())
        .isEmpty();
  }

  public void testJavaLibraryWithGeneratedSourcesHasGenJars() throws Exception {
    scratch.file(
        "java/com/google/example/BUILD",
        "java_library(",
        "  name = 'test',",
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
    String target = "//java/com/google/example:test";
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo(target);
    RuleIdeInfo ruleIdeInfo = getRuleInfoAndVerifyLabel(target, ruleIdeInfos);

    assertThat(
            transform(ruleIdeInfo.getJavaRuleIdeInfo().getGeneratedJarsList(), 
                LIBRARY_ARTIFACT_TO_STRING))
        .containsExactly(
            "<jar:java/com/google/example/libtest-gen.jar>"
            + "<source:java/com/google/example/libtest-gensrc.jar>");
  }
  
  public void testNonConformingPackageName() throws Exception {
    scratch.file(
        "bad/package/google/example/BUILD",
        "android_library(",
        "  name = 'test',",
        "  srcs = ['Test.java'],",
        ")"
    );
    String target = "//bad/package/google/example:test";
    Map<String, RuleIdeInfo> ruleIdeInfos = buildRuleIdeInfo(target);
    RuleIdeInfo ruleInfo = getRuleInfoAndVerifyLabel(target, ruleIdeInfos);

    assertThat(ruleInfo.getAndroidRuleIdeInfo().getJavaPackage())
        .isEqualTo("bad.package.google.example");
  }

  private Map<String, RuleIdeInfo> buildRuleIdeInfo(String target) throws Exception {
    AnalysisResult analysisResult =
        update(
            ImmutableList.of(target),
            ImmutableList.of(AndroidStudioInfoAspect.NAME),
            false,
            LOADING_PHASE_THREADS,
            true,
            new EventBus());
    Collection<AspectValue> aspects = analysisResult.getAspects();
    assertThat(aspects.size()).isEqualTo(1);
    AspectValue value = aspects.iterator().next();
    assertThat(value.getAspect().getName()).isEqualTo(AndroidStudioInfoAspect.NAME);
    AndroidStudioInfoFilesProvider provider =
        value.getAspect().getProvider(AndroidStudioInfoFilesProvider.class);
    Iterable<Artifact> artifacts = provider.getIdeBuildFiles();
    ImmutableMap.Builder<String, RuleIdeInfo> builder = ImmutableMap.builder();
    for (Artifact artifact : artifacts) {
      BinaryFileWriteAction generatingAction =
          (BinaryFileWriteAction) getGeneratingAction(artifact);
      RuleIdeInfo ruleIdeInfo = RuleIdeInfo.parseFrom(generatingAction.getSource().openStream());
      builder.put(ruleIdeInfo.getLabel(), ruleIdeInfo);
    }
    return builder.build();
  }
}
