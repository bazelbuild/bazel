// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.view.java;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyArtifactNames;
import static com.google.devtools.build.lib.rules.java.JavaCompileActionTestHelper.getDirectJars;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaCompileAction;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.ProguardSpecProvider;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for java_import. */
@RunWith(JUnit4.class)
public class JavaImportConfiguredTargetTest extends BuildViewTestCase {

  @Before
  public void setCommandLineFlags() throws Exception {
    setBuildLanguageOptions("--experimental_google_legacy_api");
  }

  @Before
  public final void writeBuildFile() throws Exception {
    scratch.file(
        "java/jarlib/BUILD",
        "java_import(name = 'libraryjar',",
        "            jars = ['library.jar'])",
        "java_import(name = 'libraryjar_with_srcjar',",
        "            jars = ['library.jar'],",
        "            srcjar = 'library.srcjar')");

    scratch.overwriteFile(
        "tools/allowlists/java_import_exports/BUILD",
        "package_group(",
        "    name = 'java_import_exports',",
        "    packages = ['//...'],",
        ")");
    scratch.overwriteFile(
        "tools/allowlists/java_import_empty_jars/BUILD",
        "package_group(",
        "    name = 'java_import_empty_jars',",
        "    packages = [],",
        ")");
  }

  @Test
  public void testSimple() throws Exception {
    ConfiguredTarget jarLib = getConfiguredTarget("//java/jarlib:libraryjar");
    assertThat(prettyArtifactNames(getFilesToBuild(jarLib)))
        .containsExactly("java/jarlib/library.jar");
  }

  // Regression test for b/262751943.
  @Test
  public void testCommandLineContainsTargetLabel() throws Exception {
    scratch.file("java/BUILD", "java_import(name = 'java_imp', jars = ['import.jar'])");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//java:java_imp");
    Artifact compiledArtifact =
        JavaInfo.getProvider(JavaCompilationArgsProvider.class, configuredTarget)
            .getDirectCompileTimeJars()
            .toList()
            .get(0);
    SpawnAction action = (SpawnAction) getGeneratingAction(compiledArtifact);
    ImmutableList<String> args = action.getCommandLines().allArguments();

    MoreAsserts.assertContainsSublist(args, "--target_label", "//java:java_imp");
  }

  // Regression test for b/5868388.
  @Test
  public void testJavaLibraryAllowsImportInDeps() throws Exception {
    scratchConfiguredTarget(
        "java",
        "javalib",
        "java_library(name = 'javalib',",
        "             srcs = ['Other.java'],",
        "             exports = ['//java/jarlib:libraryjar'])");
    assertNoEvents(); // Make sure that no warnings were emitted.
  }

  private static void validateRuntimeClassPath(
      ConfiguredTarget binary, String... expectedRuntimeClasspath) throws Exception {
    assertThat(
            prettyArtifactNames(
                JavaInfo.getProvider(JavaCompilationInfoProvider.class, binary)
                    .getRuntimeClasspath()
                    .getSet(Artifact.class)))
        .containsExactlyElementsIn(expectedRuntimeClasspath)
        .inOrder();
  }

  private static void validateCompilationClassPath(
      ConfiguredTarget binary, String... expectedCompilationClasspath) throws Exception {
    assertThat(
            prettyArtifactNames(
                JavaInfo.getProvider(JavaCompilationInfoProvider.class, binary)
                    .getCompilationClasspath()
                    .getSet(Artifact.class)))
        .containsExactlyElementsIn(expectedCompilationClasspath)
        .inOrder();
  }

  @Test
  public void testWithJavaLibrary() throws Exception {
    scratch.file(
        "java/somelib/BUILD",
        "java_library(name  = 'javalib',",
        "             srcs = ['Other.java'],",
        "             deps = ['//java/jarlib:libraryjar'])");

    ConfiguredTarget javaLib = getConfiguredTarget("//java/somelib:javalib");

    validateCompilationClassPath(
        javaLib, "java/jarlib/_ijar/libraryjar/java/jarlib/library-ijar.jar");

    validateRuntimeClassPath(javaLib, "java/somelib/libjavalib.jar", "java/jarlib/library.jar");
  }


  @Test
  public void testDeps() throws Exception {
    scratch.file(
        "java/jarlib2/BUILD",
        "java_library(name  = 'lib',",
        "             srcs = ['Main.java'],",
        "             deps = [':import-jar'])",
        "java_import(name  = 'import-jar',",
        "            jars = ['import.jar'],",
        "            deps = ['//java/jarlib2:depjar'],",
        "            exports = ['//java/jarlib2:exportjar'],",
        ")",
        "java_import(name  = 'depjar',",
        "            jars = ['depjar.jar'])",
        "java_import(name  = 'exportjar',",
        "            jars = ['exportjar.jar'])");

    ConfiguredTarget importJar = getConfiguredTarget("//java/jarlib2:import-jar");

    assertThat(prettyArtifactNames(getFilesToBuild(importJar)))
        .containsExactly("java/jarlib2/import.jar");

    // JavaCompilationArgs should hold classpaths of the transitive closure.
    JavaCompilationArgsProvider recursiveCompilationArgs =
        JavaInfo.getProvider(JavaCompilationArgsProvider.class, importJar);
    assertThat(prettyArtifactNames(recursiveCompilationArgs.getTransitiveCompileTimeJars()))
        .containsExactly(
            "java/jarlib2/_ijar/import-jar/java/jarlib2/import-ijar.jar",
            "java/jarlib2/_ijar/exportjar/java/jarlib2/exportjar-ijar.jar",
            "java/jarlib2/_ijar/depjar/java/jarlib2/depjar-ijar.jar")
        .inOrder();
    assertThat(prettyArtifactNames(recursiveCompilationArgs.getRuntimeJars()))
        .containsExactly(
            "java/jarlib2/import.jar", "java/jarlib2/exportjar.jar", "java/jarlib2/depjar.jar")
        .inOrder();

    // Recursive deps work the same as with java_library.
    JavaCompilationArgsProvider compilationArgsProvider =
        JavaInfo.getProvider(JavaCompilationArgsProvider.class, importJar);
    assertThat(prettyArtifactNames(compilationArgsProvider.getDirectCompileTimeJars()))
        .containsExactly(
            "java/jarlib2/_ijar/import-jar/java/jarlib2/import-ijar.jar",
            "java/jarlib2/_ijar/exportjar/java/jarlib2/exportjar-ijar.jar")
        .inOrder();
    assertThat(prettyArtifactNames(compilationArgsProvider.getRuntimeJars()))
        .containsExactly(
            "java/jarlib2/import.jar", "java/jarlib2/exportjar.jar", "java/jarlib2/depjar.jar")
        .inOrder();

    // Check that parameters propagate to Java libraries properly.
    ConfiguredTarget lib = getConfiguredTarget("//java/jarlib2:lib");
    validateCompilationClassPath(
        lib,
        "java/jarlib2/_ijar/import-jar/java/jarlib2/import-ijar.jar",
        "java/jarlib2/_ijar/exportjar/java/jarlib2/exportjar-ijar.jar",
        "java/jarlib2/_ijar/depjar/java/jarlib2/depjar-ijar.jar");

    validateRuntimeClassPath(
        lib,
        "java/jarlib2/liblib.jar",
        "java/jarlib2/import.jar",
        "java/jarlib2/exportjar.jar",
        "java/jarlib2/depjar.jar");
  }

  @Test
  public void testSrcJars() throws Exception {
    ConfiguredTarget jarLibWithSources =
        getConfiguredTarget("//java/jarlib:libraryjar_with_srcjar");

    assertThat(
            Iterables.getOnlyElement(
                    JavaInfo.getProvider(JavaRuleOutputJarsProvider.class, jarLibWithSources)
                        .getAllSrcOutputJars())
                .prettyPrint())
        .isEqualTo("java/jarlib/library.srcjar");
  }

  @Test
  public void testFromGenrule() throws Exception {
    scratch.file(
        "java/genrules/BUILD",
        "genrule(name  = 'generated_jar',",
        "        outs = ['generated.jar'],",
        "        cmd = '')",
        "genrule(name  = 'generated_src_jar',",
        "        outs = ['generated.srcjar'],",
        "        cmd = '')",
        "java_import(name  = 'library-jar',",
        "            jars = [':generated_jar'],",
        "            srcjar = ':generated_src_jar',",
        "            exports = ['//java/jarlib:libraryjar'])");
    ConfiguredTarget jarLib = getConfiguredTarget("//java/genrules:library-jar");

    JavaCompilationArgsProvider compilationArgs =
        JavaInfo.getProvider(JavaCompilationArgsProvider.class, jarLib);
    assertThat(prettyArtifactNames(compilationArgs.getTransitiveCompileTimeJars()))
        .containsExactly(
            "java/genrules/_ijar/library-jar/java/genrules/generated-ijar.jar",
            "java/jarlib/_ijar/libraryjar/java/jarlib/library-ijar.jar")
        .inOrder();
    assertThat(prettyArtifactNames(compilationArgs.getRuntimeJars()))
        .containsExactly("java/genrules/generated.jar", "java/jarlib/library.jar")
        .inOrder();

    Artifact jar = compilationArgs.getRuntimeJars().toList().get(0);
    assertThat(getGeneratingAction(jar).prettyPrint())
        .isEqualTo("action 'Executing genrule //java/genrules:generated_jar'");
  }

  @Test
  public void testAllowsJarInSrcjars() throws Exception {
    scratch.file(
        "java/srcjarlib/BUILD",
        "java_import(name  = 'library-jar',",
        "            jars = ['somelib.jar'],",
        "            srcjar = 'somelib-src.jar')");
    ConfiguredTarget jarLib = getConfiguredTarget("//java/srcjarlib:library-jar");
    assertThat(
            Iterables.getOnlyElement(
                    JavaInfo.getProvider(JavaRuleOutputJarsProvider.class, jarLib)
                        .getAllSrcOutputJars())
                .prettyPrint())
        .isEqualTo("java/srcjarlib/somelib-src.jar");
  }

  @Test
  public void testRequiresJars() throws Exception {
    checkError("pkg", "rule", "mandatory attribute 'jars'", "java_import(name = 'rule')");
  }

  @Test
  public void testPermitsEmptyJars() throws Exception {
    useConfiguration("--incompatible_disallow_java_import_empty_jars=0");
    scratchConfiguredTarget("pkg", "rule", "java_import(name = 'rule', jars = [])");
    assertNoEvents();
  }

  @Test
  public void testDisallowsFilesInExports() throws Exception {
    scratch.file("pkg/bad.jar", "");
    checkError(
        "pkg",
        "rule",
        "expected no files",
        "java_import(name = 'rule', jars = ['good.jar'], exports = ['bad.jar'])");
  }

  @Test
  public void testDisallowsArbitraryFiles() throws Exception {
    scratch.file("badlib/not-a-jar.txt", "foo");
    checkError(
        "badlib",
        "library-jar",
        getErrorMsgMisplacedFiles(
            "jars", "java_import", "//badlib:library-jar", "//badlib:not-a-jar.txt"),
        "java_import(name = 'library-jar',",
        "            jars = ['not-a-jar.txt'])");
  }

  @Test
  public void testDisallowsArbitraryFilesFromGenrule() throws Exception {
    checkError(
        "badlib",
        "library-jar",
        getErrorMsgNoGoodFiles("jars", "java_import", "//badlib:library-jar", "//badlib:gen"),
        "genrule(name = 'gen', outs = ['not-a-jar.txt'], cmd = '')",
        "java_import(name  = 'library-jar',",
        "            jars = [':gen'])");
  }

  @Test
  public void testDisallowsJavaRulesInSrcs() throws Exception {
    checkError(
        "badlib",
        "library-jar",
        "'jars' attribute cannot contain labels of Java targets",
        "java_library(name = 'javalib',",
        "             srcs = ['Javalib.java'])",
        "java_import(name  = 'library-jar',",
        "            jars = [':javalib'])");
  }

  @Test
  public void testJavaImportExportsTransitiveProguardSpecs() throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "java_import(name = 'export',",
        "            jars = ['Export.jar'],",
        "            proguard_specs = ['export.pro'],",
        "            constraints = ['android'])",
        "java_import(name = 'runtime_dep',",
        "            jars = ['RuntimeDep.jar'],",
        "            proguard_specs = ['runtime_dep.pro'],",
        "            constraints = ['android'])",
        "java_import(name = 'lib',",
        "            jars = ['Lib.jar'],",
        "            proguard_specs = ['lib.pro'],",
        "            constraints = ['android'],",
        "            exports = [':export'],",
        "            runtime_deps = [':runtime_dep'])");
    NestedSet<Artifact> providedSpecs =
        getConfiguredTarget("//java/com/google/android/hello:lib")
            .get(ProguardSpecProvider.PROVIDER)
            .getTransitiveProguardSpecs();
    assertThat(ActionsTestUtil.baseArtifactNames(providedSpecs))
        .containsAtLeast("lib.pro_valid", "export.pro_valid", "runtime_dep.pro_valid");
  }

  @Test
  public void testJavaImportValidatesProguardSpecs() throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "java_import(name = 'lib',",
        "            jars = ['Lib.jar'],",
        "            proguard_specs = ['lib.pro'],",
        "            constraints = ['android'])");
    SpawnAction action =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(
                    getOutputGroup(
                        getConfiguredTarget("//java/com/google/android/hello:lib"),
                        OutputGroupInfo.HIDDEN_TOP_LEVEL),
                    "lib.pro_valid");
    assertWithMessage("Proguard validate action").that(action).isNotNull();
    assertWithMessage("Proguard validate action input")
        .that(prettyArtifactNames(action.getInputs()))
        .contains("java/com/google/android/hello/lib.pro");
  }

  @Test
  public void testJavaImportValidatesTransitiveProguardSpecs() throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "java_import(name = 'transitive',",
        "            jars = ['Transitive.jar'],",
        "            proguard_specs = ['transitive.pro'],",
        "            constraints = ['android'])",
        "java_import(name = 'lib',",
        "            jars = ['Lib.jar'],",
        "            constraints = ['android'],",
        "            exports = [':transitive'])");
    SpawnAction action =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(
                    getOutputGroup(
                        getConfiguredTarget("//java/com/google/android/hello:lib"),
                        OutputGroupInfo.HIDDEN_TOP_LEVEL),
                    "transitive.pro_valid");
    assertWithMessage("Proguard validate action").that(action).isNotNull();
    assertWithMessage("Proguard validate action input")
        .that(prettyArtifactNames(action.getInputs()))
        .contains("java/com/google/android/hello/transitive.pro");
  }

  @Test
  public void testNeverlinkIsPopulated() throws Exception {
    scratch.file(
        "java/com/google/test/BUILD",
        "java_library(name = 'lib')",
        "java_import(name = 'jar',",
        "    neverlink = 1,",
        "    jars = ['dummy.jar'],",
        "    exports = [':lib'])");
    ConfiguredTarget processorTarget = getConfiguredTarget("//java/com/google/test:jar");
    JavaInfo javaInfo = processorTarget.get(JavaInfo.PROVIDER);
    assertThat(javaInfo.isNeverlink()).isTrue();
  }

  @Test
  public void testTransitiveSourceJars() throws Exception {
    ConfiguredTarget aTarget =
        scratchConfiguredTarget(
            "java/my",
            "a",
            "java_import(name = 'a',",
            "    jars = ['dummy.jar'],",
            "    srcjar = 'dummy-src.jar',",
            "    exports = [':b'])",
            "java_library(name = 'b',",
            "    srcs = ['B.java'])");
    getConfiguredTarget("//java/my:a");
    Set<String> inputs =
        artifactsToStrings(
            JavaInfo.getProvider(JavaSourceJarsProvider.class, aTarget).getTransitiveSourceJars());
    assertThat(inputs)
        .isEqualTo(Sets.newHashSet("src java/my/dummy-src.jar", "bin java/my/libb-src.jar"));
  }

  @Test
  public void testExportsRunfilesCollection() throws Exception {
    scratch.file(
        "java/com/google/exports/BUILD",
        "java_import(name = 'other_lib',",
        "  data = ['foo.txt'],",
        "  jars = ['other.jar'])",
        "java_import(name = 'lib',",
        "  jars = ['lib.jar'],",
        "  exports = [':other_lib'])",
        "java_binary(name = 'tool',",
        "  data = [':lib'],",
        "  main_class = 'com.google.exports.Launcher')");

    ConfiguredTarget testTarget = getConfiguredTarget("//java/com/google/exports:tool");
    Runfiles runfiles = getDefaultRunfiles(testTarget);
    assertThat(prettyArtifactNames(runfiles.getArtifacts()))
        .containsAtLeast(
            "java/com/google/exports/lib.jar",
            "java/com/google/exports/other.jar",
            "java/com/google/exports/foo.txt");
  }

  // Regression test for b/13936397: don't flatten transitive dependencies into direct deps.
  @Test
  public void testTransitiveDependencies() throws Exception {
    scratch.file(
        "java/jarlib2/BUILD",
        "java_library(name = 'lib',",
        "             srcs = ['Lib.java'],",
        "             deps = ['//java/jarlib:libraryjar'])",
        "java_import(name  = 'library2-jar',",
        "            jars = ['library2.jar'],",
        "            exports = [':lib'])",
        "java_library(name  = 'javalib2',",
        "             srcs = ['Other.java'],",
        "             deps = [':library2-jar'])");

    JavaCompileAction javacAction =
        (JavaCompileAction) getGeneratingActionForLabel("//java/jarlib2:libjavalib2.jar");
    // Direct jars should NOT include java/jarlib/libraryjar-ijar.jar
    assertThat(prettyArtifactNames(getInputs(javacAction, getDirectJars(javacAction))))
        .isEqualTo(
            Arrays.asList(
                "java/jarlib2/_ijar/library2-jar/java/jarlib2/library2-ijar.jar",
                "java/jarlib2/liblib-hjar.jar"));
  }

  @Test
  public void testRuntimeDepsAreNotOnClasspath() throws Exception {
    scratch.file(
        "java/com/google/runtimetest/BUILD",
        "java_import(",
        "    name = 'import_dep',",
        "    jars = ['import_compile.jar'],",
        "    runtime_deps = ['import_runtime.jar'],",
        ")",
        "java_library(",
        "    name = 'library_dep',",
        "    srcs = ['library_compile.java'],",
        ")",
        "java_library(",
        "    name = 'depends_on_runtimedep',",
        "    srcs = ['dummy.java'],",
        "    deps = [",
        "        ':import_dep',",
        "        ':library_dep',",
        "    ],",
        ")");

    OutputFileConfiguredTarget dependsOnRuntimeDep =
        (OutputFileConfiguredTarget)
            getFileConfiguredTarget("//java/com/google/runtimetest:libdepends_on_runtimedep.jar");

    JavaCompileAction javacAction =
        (JavaCompileAction) getGeneratingAction(dependsOnRuntimeDep.getArtifact());
    // Direct jars should NOT include import_runtime.jar
    assertThat(prettyArtifactNames(getInputs(javacAction, getDirectJars(javacAction))))
        .containsExactly(
            "java/com/google/runtimetest/_ijar/import_dep/java/com/google/runtimetest/import_compile-ijar.jar",
            "java/com/google/runtimetest/liblibrary_dep-hjar.jar");
  }

  @Test
  public void testDuplicateJars() throws Exception {
    checkError(
        "ji",
        "ji-with-dupe",
        // error:
        "Label '//ji:a.jar' is duplicated in the 'jars' attribute of rule 'ji-with-dupe'",
        // build file
        "filegroup(name='jars', srcs=['a.jar'])",
        "java_import(name = 'ji-with-dupe', jars = ['a.jar', 'a.jar'])");
  }

  @Test
  public void testDuplicateJarsThroughFilegroup() throws Exception {
    checkError(
        "ji",
        "ji-with-dupe-through-fg",
        // error:
        "in jars attribute of java_import rule //ji:ji-with-dupe-through-fg: a.jar is a duplicate",
        // build file
        "filegroup(name='jars', srcs=['a.jar'])",
        "java_import(name = 'ji-with-dupe-through-fg', jars = ['a.jar', ':jars'])");
  }

  @Test
  public void testExposesJavaProvider() throws Exception {
    ConfiguredTarget jarLib = getConfiguredTarget("//java/jarlib:libraryjar");
    JavaCompilationArgsProvider compilationArgsProvider =
        JavaInfo.getProvider(JavaCompilationArgsProvider.class, jarLib);
    assertThat(prettyArtifactNames(compilationArgsProvider.getRuntimeJars()))
        .containsExactly("java/jarlib/library.jar");
  }

  @Test
  public void testIjarCanBeDisabled() throws Exception {
    useConfiguration("--nouse_ijars");
    ConfiguredTarget lib =
        scratchConfiguredTarget(
            "java/a",
            "a",
            "java_library(name='a', srcs=['A.java'], deps=[':b'])",
            "java_import(name='b', jars=['b.jar'])");
    List<String> jars =
        ActionsTestUtil.baseArtifactNames(
            JavaInfo.getProvider(JavaCompilationArgsProvider.class, lib)
                .getTransitiveCompileTimeJars());
    assertThat(jars).doesNotContain("b-ijar.jar");
    assertThat(jars).contains("b.jar");
  }

  @Test
  public void testExports() throws Exception {
    useConfiguration("--incompatible_disallow_java_import_exports");
    checkError(
        "ugly",
        "jar",
        "java_import.exports is no longer supported; use java_import.deps instead",
        "java_library(name = 'dep', srcs = ['dep.java'])",
        "java_import(name = 'jar',",
        "    jars = ['dummy.jar'],",
        "    exports = [':dep'])");
  }
}
