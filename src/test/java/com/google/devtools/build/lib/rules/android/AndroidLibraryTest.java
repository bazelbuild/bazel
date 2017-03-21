// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.truth.Truth;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.android.AndroidIdeInfoProvider.SourceDirectory;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompileAction;
import com.google.devtools.build.lib.rules.java.JavaExportsProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link AndroidLibrary}.
 */
@RunWith(JUnit4.class)
public class AndroidLibraryTest extends AndroidBuildViewTestCase {

  @Test
  public void testSimpleLibrary() throws Exception {
    scratch.file("java/android/BUILD",
        "android_library(name = 'a',",
        "                srcs = ['A.java'],",
        "               )");
    getConfiguredTarget("//java/android:a");
  }

  @Test
  public void testBaselineCoverageArtifacts() throws Exception {
    useConfiguration("--collect_code_coverage");
    ConfiguredTarget target = scratchConfiguredTarget("java/a", "a",
        "android_library(name='a', srcs=['A.java'], deps=[':b'])",
        "android_library(name='b', srcs=['B.java'])");

    assertThat(baselineCoverageArtifactBasenames(target)).containsExactly("A.java", "B.java");
  }

  // regression test for #3169099
  @Test
  public void testLibrarySrcs() throws Exception {
    scratch.file("java/srcs/a.foo", "foo");
    scratch.file("java/srcs/BUILD",
        "android_library(name = 'valid', srcs = ['a.java', 'b.srcjar', ':gvalid', ':gmix'])",
        "android_library(name = 'invalid', srcs = ['a.foo', ':ginvalid'])",
        "android_library(name = 'mix', srcs = ['a.java', 'a.foo'])",
        "genrule(name = 'gvalid', srcs = ['a.java'], outs = ['b.java'], cmd = '')",
        "genrule(name = 'ginvalid', srcs = ['a.java'], outs = ['b.foo'], cmd = '')",
        "genrule(name = 'gmix', srcs = ['a.java'], outs = ['c.java', 'c.foo'], cmd = '')"
    );
    assertSrcsValidityForRuleType("//java/srcs", "android_library",
        ".java or .srcjar");
  }

  // regression test for #3169095
  @Test
  public void testXmbInSrcsDoesNotThrow() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratchConfiguredTarget("java/xmb", "a", "android_library(name = 'a', srcs = ['a.xmb'])");
  }

  @Test
  public void testExportsJackProvider() throws Exception {
    scratch.file(
        "java/com/google/android/BUILD",
        "android_library(",
        "  name = 'dep',",
        "  srcs = ['dep.java']",
        ")",
        "android_library(",
        "  name = 'neverlink',",
        "  srcs = ['neverlink.java'],",
        "  neverlink = 1",
        ")",
        "java_plugin(",
        "  name = 'plugin',",
        "  srcs = ['plugin.java'],",
        "  processor_class = 'com.google.android.Plugin'",
        ")",
        "android_library(",
        "  name = 'top',",
        "  srcs = ['foo.java', 'bar.srcjar'],",
        "  plugins = [':plugin'],",
        "  deps = [':dep', ':neverlink'],",
        ")");

    ConfiguredTarget topTarget = getConfiguredTarget("//java/com/google/android:top");
    JackLibraryProvider jackProvider = topTarget.getProvider(JackLibraryProvider.class);
    Iterable<String> classpathNames =
        ActionsTestUtil.baseArtifactNames(jackProvider.getTransitiveJackClasspathLibraries());
    Iterable<String> runtimeNames =
        ActionsTestUtil.baseArtifactNames(jackProvider.getTransitiveJackLibrariesToLink());
    assertThat(classpathNames).containsAllOf("libtop.jack", "libdep.jack", "libneverlink.jack");
    assertThat(runtimeNames).containsAllOf("libtop.jack", "libdep.jack");
    assertThat(runtimeNames).doesNotContain("libneverlink.jack");
    Artifact jackLibrary = getBinArtifact("libtop.jack", topTarget);
    assertThat(ActionsTestUtil.baseArtifactNames(actionsTestUtil().artifactClosureOf(jackLibrary)))
        .containsAllOf("foo.java", "bar.srcjar", "libplugin.jar");
  }

  @Test
  public void testSlashInIdlImportRoot() throws Exception {
    scratchConfiguredTarget("java/com/google/android", "avocado",
        "android_library(name='avocado',",
        "                idl_parcelables=['tropical/fruit/Avocado.aidl'],",
        "                idl_import_root='tropical/fruit')");
  }

  @Test
  public void testAndroidLibraryWithIdlImportAndNoIdls() throws Exception {
    checkError("java/com/google/android", "lib",
        "Neither idl_srcs nor idl_parcelables were specified, "
            + "but 'idl_import_root' attribute was set",
        "android_library(name = 'lib',",
        "    srcs = ['Dummy.java'],",
        "    idl_import_root = 'src')");
  }

  @Test
  public void testAndroidLibraryWithIdlImportAndIdlSrcs() throws Exception {
    scratchConfiguredTarget("java/com/google/android", "lib",
        "android_library(name = 'lib',",
        "    idl_srcs = ['Dummy.aidl'],",
        "    idl_import_root = 'src')");
  }

  @Test
  public void testAndroidLibraryWithIdlImportAndIdlParcelables() throws Exception {
    scratchConfiguredTarget("java/com/google/android", "lib",
        "android_library(name = 'lib',",
        "    idl_parcelables = ['src/android/DummyParcelable.aidl'],",
        "    idl_import_root = 'src')");
  }

  @Test
  public void testAndroidLibraryWithIdlImportAndBothIdlTypes() throws Exception {
    scratchConfiguredTarget("java/com/google/android", "lib",
        "android_library(name  = 'lib',",
        "    idl_srcs = ['src/android/Dummy.aidl'],",
        "    idl_parcelables = ['src/android/DummyParcelable.aidl'],",
        "    idl_import_root = 'src')");
  }

  @Test
  public void testAndroidLibraryWithIdlImportAndEmptyLists() throws Exception {
    scratchConfiguredTarget("java/com/google/android", "lib",
        "android_library(name  = 'lib',",
        "    idl_srcs = [],",
        "    idl_parcelables = [],",
        "    idl_import_root = 'src')");
  }

  @Test
  public void testCommandLineContainsTargetLabelAndRuleKind() throws Exception {
    scratch.file("java/android/BUILD",
        "android_library(name = 'a', srcs = ['A.java'])");
    JavaCompileAction javacAction =
        (JavaCompileAction) getGeneratingActionForLabel("//java/android:liba.jar");

    String commandLine = Iterables.toString(javacAction.buildCommandLine());
    assertThat(commandLine).contains("--rule_kind, android_library");
    assertThat(commandLine).contains("--target_label, //java/android:a");
  }

  @Test
  public void testStrictAndroidDepsOff() throws Exception {
    useConfiguration("--strict_java_deps=OFF");
    scratch.file("java/android/strict/BUILD",
        "android_library(name = 'b', srcs = ['B.java'])");
    Artifact artifact = getFileConfiguredTarget("//java/android/strict:libb.jar").getArtifact();
    JavaCompileAction compileAction = (JavaCompileAction) getGeneratingAction(artifact);
    assertEquals(BuildConfiguration.StrictDepsMode.OFF, compileAction.getStrictJavaDepsMode());
  }

  @Test
  public void testStrictAndroidDepsOn() throws Exception {
    scratch.file("java/android/strict/BUILD",
        "android_library(name = 'b', srcs = ['B.java'])");
    Artifact artifact = getFileConfiguredTarget("//java/android/strict:libb.jar").getArtifact();
    JavaCompileAction compileAction = (JavaCompileAction) getGeneratingAction(artifact);
    assertEquals(BuildConfiguration.StrictDepsMode.ERROR, compileAction.getStrictJavaDepsMode());
  }

  @Test
  public void testStrictAndroidDepsWarn() throws Exception {
    useConfiguration("--strict_android_deps=WARN");
    scratch.file("java/android/strict/BUILD",
        "android_library(name = 'b', srcs = ['B.java'])");
    Artifact artifact = getFileConfiguredTarget("//java/android/strict:libb.jar").getArtifact();
    JavaCompileAction compileAction = (JavaCompileAction) getGeneratingAction(artifact);
    assertEquals(BuildConfiguration.StrictDepsMode.WARN, compileAction.getStrictJavaDepsMode());
  }

  @Test
  public void testJavaPluginProcessorPath() throws Exception {
    scratch.file("java/test/BUILD",
        "java_library(name = 'plugin_dep',",
        "    srcs = [ 'ProcessorDep.java'])",
        "java_plugin(name = 'plugin',",
        "    srcs = ['AnnotationProcessor.java'],",
        "    processor_class = 'com.google.process.stuff',",
        "    deps = [ ':plugin_dep' ])",
        "android_library(name = 'to_be_processed',",
        "    plugins = [':plugin'],",
        "    srcs = ['ToBeProcessed.java'])");
    ConfiguredTarget target = getConfiguredTarget("//java/test:to_be_processed");

    OutputFileConfiguredTarget output = (OutputFileConfiguredTarget)
        getFileConfiguredTarget("//java/test:libto_be_processed.jar");
    JavaCompileAction javacAction =
        (JavaCompileAction) getGeneratingAction(output.getArtifact());

    assertThat(javacAction.getProcessorNames()).contains("com.google.process.stuff");
    assertThat(javacAction.getProcessorNames()).hasSize(1);

    assertEquals("libplugin.jar libplugin_dep.jar", ActionsTestUtil.baseNamesOf(
        javacAction.getProcessorpath()));
    assertEquals("ToBeProcessed.java AnnotationProcessor.java ProcessorDep.java",
        actionsTestUtil().predecessorClosureOf(getFilesToBuild(target),
            JavaSemantics.JAVA_SOURCE));
  }

  // Same test as above, enabling the plugin through the command line.
  @Test
  public void testPluginCommandLine() throws Exception {
    scratch.file("java/test/BUILD",
        "java_library(name = 'plugin_dep',",
        "    srcs = [ 'ProcessorDep.java'])",
        "java_plugin(name = 'plugin',",
        "    srcs = ['AnnotationProcessor.java'],",
        "    processor_class = 'com.google.process.stuff',",
        "    deps = [ ':plugin_dep' ])",
        "android_library(name = 'to_be_processed',",
        "    srcs = ['ToBeProcessed.java'])");

    useConfiguration("--plugin=//java/test:plugin");
    ConfiguredTarget target = getConfiguredTarget("//java/test:to_be_processed");
    OutputFileConfiguredTarget output =
        (OutputFileConfiguredTarget) getFileConfiguredTarget("//java/test:libto_be_processed.jar");
    JavaCompileAction javacAction = (JavaCompileAction) getGeneratingAction(output.getArtifact());

    assertThat(javacAction.getProcessorNames()).contains("com.google.process.stuff");
    assertThat(javacAction.getProcessorNames()).hasSize(1);
    assertEquals("libplugin.jar libplugin_dep.jar",
        ActionsTestUtil.baseNamesOf(javacAction.getProcessorpath()));
    assertEquals("ToBeProcessed.java AnnotationProcessor.java ProcessorDep.java",
        actionsTestUtil().predecessorClosureOf(getFilesToBuild(target),
            JavaSemantics.JAVA_SOURCE));
  }

  @Test
  public void testInvalidPlugin() throws Exception {
    checkError("java/test", "lib",
        // error:
        getErrorMsgMisplacedRules("plugins", "android_library",
            "//java/test:lib", "java_library", "//java/test:not_a_plugin"),
        // BUILD file:
        "java_library(name = 'not_a_plugin',",
        "    srcs = [ 'NotAPlugin.java'])",
        "android_library(name = 'lib',",
        "    plugins = [':not_a_plugin'],",
        "    srcs = ['Lib.java'])");
  }

  @Test
  public void testDisallowDepsWithoutSrcs() throws Exception {
    useConfiguration("--experimental_allow_android_library_deps_without_srcs=false");
    checkError("android/deps", "b",
        // message:
        "deps not allowed without srcs; move to exports?",
        // build file
        "android_library(name = 'a', srcs = ['a.java'])",
        "android_library(name = 'b', deps = [':a'])");
  }

  @Test
  public void testAlwaysAllowDepsWithoutSrcsIfLocalResources() throws Exception {
    scratch.file("java/android/BUILD",
        "android_library(name = 'a', srcs = ['a.java'])",
        "android_library(name = 'r',",
        "    manifest = 'AndroidManifest.xml',",
        "    resource_files = glob(['res/**']),",
        "    deps = [':a'])");

    scratch.file("java/android/res/values/strings.xml",
        "<resources><string name = 'hello'>Hello Android!</string></resources>");

    useConfiguration("--experimental_allow_android_library_deps_without_srcs=false");

    getConfiguredTarget("//java/android:r");
    assertNoEvents();
  }

  @Test
  public void testTransitiveDependencyThroughExports() throws Exception {
    scratch.file("java/test/BUILD",
        "android_library(name = 'somelib',",
        "    srcs = ['Lib.java'],",
        "    deps = [':somealias'])",
        "android_library(name = 'somealias',",
        "    exports = [':somedep'])",
        "android_library(name = 'somedep',",
        "    srcs = ['Dependency.java'],",
        "    deps = [ ':otherdep' ])",
        "android_library(name = 'otherdep',",
        "    srcs = ['OtherDependency.java'])");
    ConfiguredTarget libTarget = getConfiguredTarget("//java/test:somelib");
    assertThat(actionsTestUtil().predecessorClosureAsCollection(getFilesToBuild(libTarget),
        JavaSemantics.JAVA_SOURCE)).containsExactly(
        "Lib.java", "Dependency.java", "OtherDependency.java");
    assertNoEvents();
  }

  @Test
  public void testTransitiveStrictDeps() throws Exception {
    scratch.file("java/peach/BUILD",
        "android_library(name='a', exports=[':b'])",
        "android_library(name='b', srcs=['B.java'], deps=[':c'])",
        "android_library(name='c', srcs=['C.java'])");

    useConfiguration("--strict_java_deps=ERROR");

    ConfiguredTarget a = getConfiguredTarget("//java/peach:a");
    Iterable<String> compileTimeJars = ActionsTestUtil.baseArtifactNames(
        a.getProvider(JavaCompilationArgsProvider.class)
            .getJavaCompilationArgs().getCompileTimeJars());
    assertThat(compileTimeJars).contains("libb-hjar.jar");
    assertThat(compileTimeJars).doesNotContain("libc-hjar.jar");
    assertNoEvents();
  }

  @Test
  public void testEmitOutputDeps() throws Exception {
    scratch.file("java/deps/BUILD",
        "android_library(name = 'a', exports = [':b'])",
        "android_library(name = 'b', srcs = ['B.java'])");

    useConfiguration("--java_deps");

    JavaCompileAction aAction = (JavaCompileAction) getGeneratingActionForLabel(
        "//java/deps:liba.jar");
    List<String> aOutputs = ActionsTestUtil.prettyArtifactNames(aAction.getOutputs());
    assertThat(aOutputs).doesNotContain("java/deps/liba.jdeps");

    JavaCompileAction bAction = (JavaCompileAction) getGeneratingActionForLabel(
        "//java/deps:libb.jar");
    List<String> bOutputs = ActionsTestUtil.prettyArtifactNames(bAction.getOutputs());
    assertThat(bOutputs).contains("java/deps/libb.jdeps");
    assertNoEvents();
  }

  @Test
  public void testDependencyArtifactsWithExports() throws Exception {
    scratch.file("java/classpath/BUILD",
        "android_library(name = 'a', srcs = ['A.java'], deps = [':b', ':c'])",
        "android_library(name = 'b', exports = [':d'])",
        "android_library(name = 'c', srcs = ['C.java'], exports = [':e'])",
        "android_library(name = 'd', srcs = ['D.java'])",
        "android_library(name = 'e', srcs = ['E.java'])");

    JavaCompileAction aAction = (JavaCompileAction) getGeneratingActionForLabel(
        "//java/classpath:liba.jar");
    List<String> deps =
        ActionsTestUtil.prettyArtifactNames(aAction.getCompileTimeDependencyArtifacts());
    assertThat(deps)
        .containsExactly(
            "java/classpath/libc-hjar.jdeps",
            "java/classpath/libd-hjar.jdeps",
            "java/classpath/libe-hjar.jdeps");
    assertNoEvents();
  }

  @Test
  public void testSrcsLessExportsAreStillDirect() throws Exception {
    scratch.file("java/exports/BUILD",
        "android_library(name = 'a', srcs = ['a.java'])",
        "android_library(name = 'b', deps = ['a'])",
        "android_library(name = 'c', srcs = ['c.java'], deps = [':b'])");

    ConfiguredTarget bTarget = getConfiguredTarget("//java/exports:b");
    ConfiguredTarget cTarget = getConfiguredTarget("//java/exports:c");

    assertEquals(Arrays.asList("liba.jar"),
        ActionsTestUtil.baseArtifactNames(getDefaultRunfiles(bTarget).getArtifacts()));

    assertEquals(Arrays.asList("liba.jar", "libc.jar"),
        ActionsTestUtil.baseArtifactNames(getDefaultRunfiles(cTarget).getArtifacts()));

    assertNoEvents();
  }

  @Test
  public void testExportsWithStrictJavaDepsFlag() throws Exception {
    scratch.file("java/exports/BUILD",
        "android_library(name = 'a', srcs = ['a.java'])",
        "android_library(name = 'b', srcs = ['b.java'], exports = ['a'])",
        "android_library(name = 'c', srcs = ['c.java'], deps = [':b'])");

    useConfiguration("--strict_java_deps=WARN");

    JavaCompileAction javacAction =
        (JavaCompileAction) getGeneratingActionForLabel("//java/exports:libc.jar");

    assertThat(ActionsTestUtil.prettyArtifactNames(javacAction.getDirectJars()))
        .containsExactly("java/exports/libb-hjar.jar", "java/exports/liba-hjar.jar");
    assertNoEvents();
  }

  @Test
  public void testExportsRunfiles() throws Exception {
    scratch.file("java/exports/BUILD",
        "android_library(name = 'a', srcs = ['a.java'], data = ['data.txt'])",
        "android_library(name = 'b', srcs = ['b.java'], exports = [':a'])");

    ConfiguredTarget bTarget = getConfiguredTarget("//java/exports:b");

    assertEquals(
        ActionsTestUtil.baseArtifactNames(getDefaultRunfiles(bTarget).getArtifacts()),
        Arrays.asList("data.txt", "liba.jar", "libb.jar"));
    assertNoEvents();
  }

  @Test
  public void testTransitiveExports() throws Exception {
    scratch.file("java/com/google/exports/BUILD",
        "android_library(name = 'dummy',",
        "    srcs = ['dummy.java'],",
        "    exports = [':dummy2'])",
        "android_library(name = 'dummy2',",
        "    srcs = ['dummy2.java'],",
        "    exports = [':dummy3'])",
        "android_library(name = 'dummy3',",
        "    srcs = ['dummy3.java'],",
        "    exports = [':dummy4'])",
        "android_library(name = 'dummy4',",
        "    srcs = ['dummy4.java'])");

    ConfiguredTarget target = getConfiguredTarget("//java/com/google/exports:dummy");
    List<Label> exports = ImmutableList.copyOf(
        target.getProvider(JavaExportsProvider.class).getTransitiveExports());
    assertThat(exports).containsExactly(Label.parseAbsolute("//java/com/google/exports:dummy2"),
        Label.parseAbsolute("//java/com/google/exports:dummy3"),
        Label.parseAbsolute("//java/com/google/exports:dummy4"));
    assertNoEvents();
  }

  @Test
  public void testSimpleIdl() throws Exception {
    scratch.file("java/android/BUILD",
        "android_library(name = 'idl',",
        "                idl_srcs = ['a.aidl'])");
    getConfiguredTarget("//java/android:idl");
    assertNoEvents();
  }

  @Test
  public void testIdlSrcsFromAnotherPackageFails() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("java/android/a/BUILD",
        "exports_files(['A.aidl'])");
    scratch.file("java/android/BUILD",
        "android_library(name = 'idl',",
        "                idl_srcs = ['//java/android/a:A.aidl'])");
    getConfiguredTarget("//java/android:idl");
    assertContainsEvent("do not import '//java/android/a:A.aidl' directly. You should either"
        + " move the file to this package or depend on an appropriate rule there");
  }

  @Test
  public void testIdlClassJarAction() throws Exception {
    scratch.file("java/android/BUILD",
        "android_library(name = 'idl',",
        "                idl_srcs = ['a.aidl', 'b.aidl', 'c.aidl'])");
    ConfiguredTarget idlTarget =
        getConfiguredTarget("//java/android:idl");
    NestedSet<Artifact> outputGroup =
        getOutputGroup(idlTarget, AndroidSemantics.IDL_JARS_OUTPUT_GROUP);

    SpawnAction classJarAction = (SpawnAction) actionsTestUtil().getActionForArtifactEndingWith(
        actionsTestUtil().artifactClosureOf(outputGroup), "libidl-idl.jar");
    SpawnAction sourceJarAction = (SpawnAction) actionsTestUtil().getActionForArtifactEndingWith(
        actionsTestUtil().artifactClosureOf(outputGroup), "libidl-idl.srcjar");

    assertSame(classJarAction, sourceJarAction);

    PathFragment genfilesPath =
        getTargetConfiguration()
            .getOutputDirectory(RepositoryName.MAIN)
            .getExecPath()
            .getRelative("genfiles/java/android/idl_aidl/java/android");
    assertThat(classJarAction.getArguments()).containsAllOf(
        genfilesPath.getRelative("a.java").getPathString(),
        genfilesPath.getRelative("b.java").getPathString(),
        genfilesPath.getRelative("c.java").getPathString());
  }

  @Test
  public void testIdlOutputGroupTransitivity() throws Exception {
    scratch.file("java/android/BUILD",
        "android_library(name = 'lib',",
        "                idl_srcs = ['a.aidl'],",
        "                deps = [':dep'])",
        "android_library(name = 'dep',",
        "                idl_srcs = ['b.aidl'])");
    ConfiguredTarget idlTarget =
        getConfiguredTarget("//java/android:lib");
    NestedSet<Artifact> outputGroup =
        getOutputGroup(idlTarget, AndroidSemantics.IDL_JARS_OUTPUT_GROUP);
    List<String> asString = Lists.newArrayList();
    for (Artifact artifact : outputGroup) {
      asString.add(artifact.getRootRelativePathString());
    }
    assertThat(asString).containsAllOf(
        "java/android/libdep-idl.jar",
        "java/android/libdep-idl.srcjar",
        "java/android/liblib-idl.jar",
        "java/android/liblib-idl.srcjar"
    );
  }

  @Test
  public void testNoJavaDir() throws Exception {
    checkError("android/hello", "idl",
        // message:
        "Cannot determine java/javatests root for import android/hello/Import.aidl",
        // build file:
        "android_library(name = 'idl',",
        "                srcs = ['Import.java'],",
        "                idl_parcelables = ['Import.aidl'])");
  }

  @Test
  public void testExportedPluginsAreInherited() throws Exception {
    scratch.file(
        "java/test/BUILD",
        "java_plugin(name = 'plugin',",
        "    srcs = [ 'Plugin.java' ],",
        "    processor_class = 'com.google.process.stuff')",
        "android_library(name = 'exporting_lib',",
        "    srcs = [ 'ExportingLib.java' ],",
        "    exported_plugins = [ ':plugin' ])",
        "android_library(name = 'consuming_lib',",
        "    srcs = [ 'ConsumingLib.java' ],",
        "    deps = [ ':exporting_lib' ])",
        "android_library(name = 'leaf_lib',",
        "    srcs = [ 'LeafLib.java' ],",
        "    deps = [ ':consuming_lib' ])");

    getConfiguredTarget("//java/test:consuming_lib");
    getConfiguredTarget("//java/test:leaf_lib");
    // libconsuming_lib should include the plugin, since it directly depends on exporting_lib
    assertThat(getProcessorNames("//java/test:libconsuming_lib.jar"))
        .containsExactly("com.google.process.stuff");
    // but libleaf_lib should not, because its dependency is transitive.
    assertThat(getProcessorNames("//java/test:libleaf_lib.jar")).isEmpty();
  }

  @Test
  public void testAidlLibAddsProguardSpecs() throws Exception {
    scratch.file("sdk/BUILD",
        "android_sdk(name = 'sdk',",
        "            aapt = 'aapt',",
        "            adb = 'adb',",
        "            aidl = 'aidl',",
        "            aidl_lib = ':aidl_lib',",
        "            android_jar = 'android.jar',",
        "            annotations_jar = 'annotations_jar',",
        "            apkbuilder = 'apkbuilder',",
        "            apksigner = 'apksigner',",
        "            dx = 'dx',",
        "            framework_aidl = 'framework_aidl',",
        "            main_dex_classes = 'main_dex_classes',",
        "            main_dex_list_creator = 'main_dex_list_creator',",
        "            proguard = 'proguard',",
        "            shrinked_android_jar = 'shrinked_android_jar',",
        "            zipalign = 'zipalign',",
        "            jack = 'jack',",
        "            jill = 'jill',",
        "            resource_extractor = 'resource_extractor')",
        "java_library(name = 'aidl_lib',",
        "             srcs = ['AidlLib.java'],",
        "             proguard_specs = ['aidl_lib.cfg'])");

    scratch.file("java/com/google/android/hello/BUILD",
        "android_library(name = 'library',",
        "                srcs = ['MainActivity.java'],",
        "                idl_srcs = ['IMyInterface.aidl'])",
        "android_library(name = 'library_no_idl',",
        "                srcs = ['MainActivity.java'])",
        "android_binary(name = 'binary',",
        "               deps = [':library'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_specs = ['proguard-spec.pro'])",
        "android_binary(name = 'binary_no_idl',",
        "               deps = [':library_no_idl'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_specs = ['proguard-spec.pro'])");
    useConfiguration("--android_sdk=//sdk:sdk");

    // Targets with AIDL-generated sources also get AIDL support lib Proguard specs
    ConfiguredTarget binary = getConfiguredTarget("//java/com/google/android/hello:binary");
    Action action = actionsTestUtil().getActionForArtifactEndingWith(
        getFilesToBuild(binary), "_proguard.jar");
    assertNotNull(ActionsTestUtil.getFirstArtifactEndingWith(action.getInputs(),
        "sdk/aidl_lib.cfg_valid"));

    // Targets without AIDL-generated sources don't care
    ConfiguredTarget binaryNoIdl =
        getConfiguredTarget("//java/com/google/android/hello:binary_no_idl");
    Action actionNoIdl = actionsTestUtil().getActionForArtifactEndingWith(
        getFilesToBuild(binaryNoIdl), "_proguard.jar");
    assertNull(ActionsTestUtil.getFirstArtifactEndingWith(actionNoIdl.getInputs(),
        "sdk/aidl_lib.cfg_valid"));
  }

  private List<String> getTransitiveDependentResourceDirs(
      ConfiguredTarget target, List<String> actualArgs) {
    assertThat(actualArgs).contains("--data");
    String actualFlagValue = actualArgs.get(actualArgs.indexOf("--data") + 1);
    return getDependentResourceDirs(target, actualFlagValue);
  }

  private List<String> getDirectDependentResourceDirs(
      ConfiguredTarget target, List<String> actualArgs) {
    assertThat(actualArgs).contains("--directData");
    String actualFlagValue = actualArgs.get(actualArgs.indexOf("--directData") + 1);
    return getDependentResourceDirs(target, actualFlagValue);
  }

  private List<String> getDependentResourceDirs(
      ConfiguredTarget target, String actualFlagValue) {
    ImmutableList.Builder<String> actualPaths = ImmutableList.builder();
    for (String resourceDependency : actualFlagValue.split(",")) {
      if (target
          .getConfiguration()
          .getFragment(AndroidConfiguration.class)
          .useParallelResourceProcessing()) {
        assertThat(actualFlagValue).matches("[^;]*;[^;]*;[^;]*;.*");
        actualPaths.add(resourceDependency.split(";")[0].split("#"));
      } else {
        assertThat(actualFlagValue).matches("[^:]*:[^:]*:[^:]*:.*");
        actualPaths.add(resourceDependency.split(":")[0].split("#"));
      }
    }
    return actualPaths.build();
  }

  private List<String> getDependentAssetDirs(
      ConfiguredTarget target, String flag, List<String> actualArgs) {
    assertThat(actualArgs).contains(flag);
    String actualFlagValue = actualArgs.get(actualArgs.indexOf(flag) + 1);
    ImmutableList.Builder<String> actualPaths = ImmutableList.builder();
    for (String resourceDependency : actualFlagValue.split(",")) {
      if (target
          .getConfiguration()
          .getFragment(AndroidConfiguration.class)
          .useParallelResourceProcessing()) {
        assertThat(actualFlagValue).matches("[^;]*;[^;]*;[^;]*;.*");
        actualPaths.add(resourceDependency.split(";")[1].split("#"));
      } else {
        assertThat(actualFlagValue).matches("[^:]*:[^:]*:[^:]*:.*");
        actualPaths.add(resourceDependency.split(":")[1].split("#"));
      }
    }
    return actualPaths.build();
  }

  @Test
  public void testResourcesMultipleDirectoriesFromPackage() throws Exception {
    scratch.file("c/b/m/a/BUILD",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                custom_package = 'com.google.android.apps.a',",
        "                resource_files = [",
        "                  'b_/res/values/strings.xml',",
        "                ]",
        "                )");
    scratch.file("c/b/m/a/b_/res",
        "<resources><string name = 'hello'>Hello Android!</string></resources>");
    ConfiguredTarget resource = getConfiguredTarget("//c/b/m/a:r");

    List<String> args = ((SpawnAction) getGeneratingAction(getResourceArtifact(resource)))
        .getArguments();
    assertPrimaryResourceDirs(resource, ImmutableList.of("c/b/m/a/b_/res"), args);
  }

  @Test
  public void testSimpleResources() throws Exception {
    scratch.file("java/android/BUILD",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = glob(['res/**']),",
        "                )");
    scratch.file("java/android/res/values/strings.xml",
        "<resources><string name = 'hello'>Hello Android!</string></resources>");
    ConfiguredTarget resource = getConfiguredTarget("//java/android:r");

    List<String> args = ((SpawnAction) getGeneratingAction(getResourceArtifact(resource)))
        .getArguments();
    assertPrimaryResourceDirs(resource, ImmutableList.of("java/android/res"), args);
  }

  @Test
  public void testResourcesWithConfigurationQualifier() throws Exception {
    scratch.file("java/android/BUILD",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = glob(['res/**']),",
        "                )");
    scratch.file("java/android/res/values-en/strings.xml",
        "<resources><string name = 'hello'>Hello Android!</string></resources>");
    scratch.file("java/android/res/values/strings.xml",
        "<resources><string name = 'hello'>Hello Android!</string></resources>");
    ConfiguredTarget resource = getConfiguredTarget("//java/android:r");

    List<String> args = ((SpawnAction) getGeneratingAction(getResourceArtifact(resource)))
        .getArguments();
    assertPrimaryResourceDirs(resource, ImmutableList.of("java/android/res"), args);
  }

  @Test
  public void testResourcesInOtherPackage_exported() throws Exception {
    scratch.file("java/android/BUILD",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['//java/other:res/values/strings.xml'],",
        "                )");
    scratch.file("java/other/BUILD",
        "exports_files(['res/values/strings.xml'])");
    ConfiguredTarget resource = getConfiguredTarget("//java/android:r");

    List<String> args = ((SpawnAction) getGeneratingAction(getResourceArtifact(resource)))
        .getArguments();
    assertPrimaryResourceDirs(resource, ImmutableList.of("java/other/res"), args);
    assertNoEvents();
  }

  @Test
  public void testResourcesInOtherPackage_filegroup() throws Exception {
    scratch.file("java/android/BUILD",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['//java/other:fg'],",
        "                )");
    scratch.file("java/other/BUILD",
        "filegroup(name = 'fg',",
        "          srcs = ['res/values/strings.xml'],",
        ")");
    ConfiguredTarget resource = getConfiguredTarget("//java/android:r");

    List<String> args = ((SpawnAction) getGeneratingAction(getResourceArtifact(resource)))
        .getArguments();
    assertPrimaryResourceDirs(resource, ImmutableList.of("java/other/res"), args);
    assertNoEvents();
  }

  // Regression test for b/11924769
  @Test
  public void testResourcesInOtherPackage_filegroupWithExternalSources() throws Exception {
    scratch.file("java/android/BUILD",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = [':fg'],",
        "                )",
        "filegroup(name = 'fg',",
        "          srcs = ['//java/other:res/values/strings.xml'])");
    scratch.file("java/other/BUILD",
        "exports_files(['res/values/strings.xml'])");
    ConfiguredTarget resource = getConfiguredTarget("//java/android:r");

    List<String> args = ((SpawnAction) getGeneratingAction(getResourceArtifact(resource)))
        .getArguments();
    assertPrimaryResourceDirs(resource, ImmutableList.of("java/other/res"), args);
    assertNoEvents();
  }

  // Regression test for b/11924769
  @Test
  public void testResourcesInOtherPackage_doubleFilegroup() throws Exception {
    scratch.file("java/android/BUILD",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = [':fg'],",
        "                )",
        "filegroup(name = 'fg',",
        "          srcs = ['//java/other:fg'])");
    scratch.file("java/other/BUILD",
        "filegroup(name = 'fg',",
        "          srcs = ['res/values/strings.xml'],",
        ")");
    ConfiguredTarget resource = getConfiguredTarget("//java/android:r");

    List<String> args = ((SpawnAction) getGeneratingAction(getResourceArtifact(resource)))
        .getArguments();
    assertPrimaryResourceDirs(resource, ImmutableList.of("java/other/res"), args);
    assertNoEvents();
  }

  @Test
  public void testManifestMissingFails() throws Exception {
    checkError("java/android", "r",
        "is required when resource_files or assets are defined.",
        "filegroup(name = 'b')",
        "android_library(name = 'r',",
        "                resource_files = [':b'],",
        "                )");
  }

  @Test
  public void testResourcesDoesNotMatchDirectoryLayout_BadFile() throws Exception {
    checkError("java/android", "r",
        "'java/android/res/somefile.xml' is not in the expected resource directory structure of"
            + " <resource directory>/{"
            + Joiner.on(',').join(LocalResourceContainer.Builder.RESOURCE_DIRECTORY_TYPES) + "}",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['res/somefile.xml', 'r/t/f/m/raw/fold']",
        "                )");
  }

  @Test
  public void testResourcesDoesNotMatchDirectoryLayout_BadDirectory() throws Exception {
    checkError("java/android", "r",
        "'java/android/res/other/somefile.xml' is not in the expected resource directory structure"
            + " of <resource directory>/{"
            + Joiner.on(',').join(LocalResourceContainer.Builder.RESOURCE_DIRECTORY_TYPES) + "}",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['res/other/somefile.xml', 'r/t/f/m/raw/fold']",
        "                )");
  }

  @Test
  public void testResourcesNotUnderCommonDirectoryFails() throws Exception {
    checkError("java/android", "r",
        "'java/android/r/t/f/m/raw/fold' (generated by '//java/android:r/t/f/m/raw/fold') is not"
            + " in the same directory 'res' (derived from java/android/res/raw/speed). All"
            + " resources must share a common directory.",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['res/raw/speed', 'r/t/f/m/raw/fold']",
        "                )");
  }

  @Test
  public void testAssetsDirAndNoAssetsFails() throws Exception {
    checkError("cpp/android", "r",
        "'assets' and 'assets_dir' should be either both empty or both non-empty",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                assets_dir = 'assets',",
        "                )");
  }

  @Test
  public void testAssetsNotUnderAssetsDirFails() throws Exception {
    checkError("java/android", "r",
        "'java/android/r/t/f/m' (generated by '//java/android:r/t/f/m') is not beneath 'assets'",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                assets_dir = 'assets',",
        "                assets = ['assets/valuable', 'r/t/f/m']",
        "                )");
  }

  @Test
  public void testAssetsAndNoAssetsDirFails() throws Exception {
    scratch.file("java/android/assets/values/strings.xml",
        "<resources><string name = 'hello'>Hello Android!</string></resources>");
    checkError("java/android", "r",
        "'assets' and 'assets_dir' should be either both empty or both non-empty",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                assets = glob(['assets/**']),",
        "                )");
  }

  @Test
  public void testFileLocation() throws Exception {
    scratch.file("java/android/BUILD",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                )");
    ConfiguredTarget foo = getConfiguredTarget("//java/android:r");
    assertEquals(getTargetConfiguration().getBinDirectory(RepositoryName.MAIN),
        ActionsTestUtil.getFirstArtifactEndingWith(getFilesToBuild(foo), "r.srcjar").getRoot());
  }

  // regression test for #3294893
  @Test
  public void testNoJavaPathFoundDoesNotThrow() throws Exception {
    checkError("third_party/java_src/android/app", "r",
        "The location of your BUILD file determines the Java package used for Android resource "
            + "processing. A directory named \"java\" or \"javatests\" will be used as your Java "
            + "source root and the path of your BUILD file relative to the Java source root will "
            + "be used as the package for Android resource processing. The Java source root could "
            + "not be determined for \"third_party/java_src/android/app\". Move your BUILD file "
            + "under a java or javatests directory, or set the 'custom_package' attribute.",
        "licenses(['notice'])",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                )");
  }

  @Test
  public void testWithRenameManifestPackage() throws Exception {
    scratch.file("a/r/BUILD",
        "android_library(name = 'r',",
        "               srcs = ['Foo.java'],",
        "               custom_package = 'com.google.android.bar',",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = ['res/values/strings.xml'],",
        "               )");
    ConfiguredTarget r = getConfiguredTarget("//a/r:r");
    assertNoEvents();
    List<String> args = ((SpawnAction) getGeneratingAction(getResourceArtifact(r))).getArguments();
    assertContainsSublist(args,
        ImmutableList.of("--packageForR", "com.google.android.bar"));
  }

  @Test
  public void testDebugConfiguration() throws Exception {
    scratch.file("java/apps/android/BUILD",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                )");
    checkDebugMode("//java/apps/android:r", true);
    useConfiguration("--compilation_mode=opt");
    checkDebugMode("//java/apps/android:r", false);
  }

  @Test
  public void testNeverlinkResources_AndroidResourcesProvider() throws Exception {
    scratch.file("java/apps/android/BUILD",
        "android_library(name = 'foo',",
        "                manifest = 'AndroidManifest.xml',",
        "                deps = [':lib', ':lib_neverlink'])",
        "android_library(name = 'lib_neverlink',",
        "                neverlink = 1,",
        "                manifest = 'AndroidManifest.xml',",
        "                deps = [':bar'])",
        "android_library(name = 'lib',",
        "                manifest = 'AndroidManifest.xml',",
        "                deps = [':bar'])",
        "android_library(name = 'bar',",
        "                manifest = 'AndroidManifest.xml')");
    Function<ResourceContainer, Label> getLabel = new Function<ResourceContainer, Label>() {
      @Override
      public Label apply(ResourceContainer container) {
        return container.getLabel();
      }
    };
    ConfiguredTarget foo = getConfiguredTarget("//java/apps/android:foo");
    assertThat(Iterables.transform(
        foo.getProvider(AndroidResourcesProvider.class).getTransitiveAndroidResources(), getLabel))
        .containsExactly(
            Label.parseAbsolute("//java/apps/android:lib"),
            Label.parseAbsolute("//java/apps/android:bar"));
    assertThat(Iterables.transform(
        foo.getProvider(AndroidResourcesProvider.class).getDirectAndroidResources(), getLabel))
        .containsExactly(Label.parseAbsolute("//java/apps/android:foo"));

    ConfiguredTarget lib = getConfiguredTarget("//java/apps/android:lib");
    assertThat(Iterables.transform(
        lib.getProvider(AndroidResourcesProvider.class).getTransitiveAndroidResources(), getLabel))
        .containsExactly(Label.parseAbsolute("//java/apps/android:bar"));
    assertThat(Iterables.transform(
        lib.getProvider(AndroidResourcesProvider.class).getDirectAndroidResources(), getLabel))
        .containsExactly(Label.parseAbsolute("//java/apps/android:lib"));

    ConfiguredTarget libNeverlink = getConfiguredTarget("//java/apps/android:lib_neverlink");
    assertThat(libNeverlink.getProvider(AndroidResourcesProvider.class)
        .getTransitiveAndroidResources())
        .isEmpty();
    assertThat(libNeverlink.getProvider(AndroidResourcesProvider.class)
        .getDirectAndroidResources())
        .isEmpty();
  }

  @Test
  public void testNeverlinkResources_JavaCompileAction() throws Exception {
    scratch.file("java/apps/android/BUILD",
        "android_library(name = 'foo',",
        "                manifest = 'AndroidManifest.xml',",
        "                deps = [':lib', ':lib_neverlink'])",
        "android_library(name = 'lib_neverlink',",
        "                neverlink = 1,",
        "                manifest = 'AndroidManifest.xml',",
        "                deps = [':bar'])",
        "android_library(name = 'lib',",
        "                manifest = 'AndroidManifest.xml',",
        "                deps = [':bar'])",
        "android_library(name = 'bar',",
        "                manifest = 'AndroidManifest.xml')");

    ConfiguredTarget foo = getConfiguredTarget("//java/apps/android:foo");
    JavaCompilationArgsProvider argsProvider = foo.getProvider(JavaCompilationArgsProvider.class);
    JavaCompileAction javacAction =
        (JavaCompileAction) getGeneratingActionForLabel("//java/apps/android:libfoo.jar");

    assertThat(argsProvider.getJavaCompilationArgs().getCompileTimeJars())
        .contains(ActionsTestUtil.getFirstArtifactEndingWith(
            javacAction.getInputs(), "lib_neverlink_resources.jar"));
    assertThat(argsProvider.getJavaCompilationArgs().getRuntimeJars())
        .doesNotContain(ActionsTestUtil.getFirstArtifactEndingWith(
            javacAction.getInputs(), "lib_neverlink_resources.jar"));
  }

  @Test
  public void testResourceMergeAndProcessParallel() throws Exception {
    // Test that for android_library, we can divide the resource processing action into
    // smaller actions.
    useConfiguration("--experimental_use_parallel_android_resource_processing");
    scratch.file(
        "java/android/app/foo/BUILD",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = glob(['res/**']),",
        "                )");
    scratch.file(
        "java/android/app/foo/res/values/strings.xml",
        "<resources>",
        "<string name='hello'>Aloha!</string>",
        "<string name='goodbye'>Aloha!</string>",
        "</resources>");
    ConfiguredTarget target = getConfiguredTarget("//java/android/app/foo:r");

    NestedSet<Artifact> filesToBuild = getFilesToBuild(target);
    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(filesToBuild);

    ResourceContainer resources =
        Iterables.getOnlyElement(
            target.getProvider(AndroidResourcesProvider.class).getDirectAndroidResources());

    SpawnAction resourceParserAction =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(artifacts,
                    "/" + resources.getSymbols().getFilename());
    SpawnAction resourceClassJarAction =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(artifacts,
                    "/" + resources.getJavaClassJar().getFilename());
    SpawnAction resourceSrcJarAction =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(artifacts,
                    "/" + resources.getJavaSourceJar().getFilename());
    assertThat(resourceParserAction.getMnemonic()).isEqualTo("AndroidResourceParser");
    assertThat(resourceClassJarAction.getMnemonic()).isEqualTo("AndroidResourceMerger");
    assertThat(resourceSrcJarAction.getMnemonic()).isEqualTo("AndroidResourceValidator");
    // Validator also generates an R.txt.
    assertThat(resourceSrcJarAction.getOutputs()).contains(resources.getRTxt());
  }

  private void checkDebugMode(String target, boolean isDebug) throws Exception {
    ConfiguredTarget foo = getConfiguredTarget(target);
    SpawnAction action = (SpawnAction) actionsTestUtil().getActionForArtifactEndingWith(
        getFilesToBuild(foo), "r.srcjar");
    assertEquals(isDebug, action.getArguments().contains("--debug"));
  }

  @Test
  public void testGeneratedManifestPackage() throws Exception {
    scratch.file("java/android/BUILD",
        "android_library(name = 'l',",
        "    srcs = ['foo.java'])",
        "android_library(name = 'l2',",
        "    custom_package = 'foo',",
        "    srcs = ['foo.java'])");
    scratch.file("third_party/android/BUILD",
        "licenses(['notice'])",
        "android_library(name = 'l',",
        "    srcs = ['foo.java'])");

    ConfiguredTarget target = getConfiguredTarget("//java/android:l");
    Artifact manifest = getBinArtifact("l_generated/l/AndroidManifest.xml", target);
    FileWriteAction action = (FileWriteAction) getGeneratingAction(manifest);
    assertThat(action.getFileContents()).contains("package=\"android\"");

    target = getConfiguredTarget("//java/android:l2");
    manifest = getBinArtifact("l2_generated/l2/AndroidManifest.xml", target);
    action = (FileWriteAction) getGeneratingAction(manifest);
    assertThat(action.getFileContents()).contains("package=\"foo\"");

    target = getConfiguredTarget("//third_party/android:l");
    manifest = getBinArtifact("l_generated/l/AndroidManifest.xml", target);
    action = (FileWriteAction) getGeneratingAction(manifest);
    assertThat(action.getFileContents()).contains("package=\"third_party.android\"");
  }

  @Test
  public void testGeneratedIdlSrcs() throws Exception {
    scratch.file("java/android/BUILD",
        "genrule(name = 'idl',",
        "        outs = ['MyInterface.aidl'],",
        "        cmd = 'touch $@')",
        "android_library(name = 'lib',",
        "                idl_srcs = [':idl'],",
        "                idl_parcelables = ['MyParcelable.aidl'])");
    ConfiguredTarget target = getConfiguredTarget("//java/android:lib");

    PathFragment genfilesJavaPath =
        getTargetConfiguration()
            .getOutputDirectory(RepositoryName.MAIN)
            .getExecPath()
            .getRelative("genfiles/java");
    SpawnAction action = (SpawnAction) actionsTestUtil().getActionForArtifactEndingWith(
        actionsTestUtil().artifactClosureOf(getFilesToBuild(target)), "MyInterface.java");
    assertThat(action.getArguments())
        .containsAllIn(ImmutableList.of("-Ijava", "-I" + genfilesJavaPath.getPathString()));
  }

  @Test
  public void testMultipleLibsSameIdls() throws Exception {
    scratch.file("java/android/BUILD",
        "android_library(name = 'idl1',",
        "                idl_srcs = ['MyInterface.aidl'])",
        "android_library(name = 'idl2',",
        "                idl_srcs = ['MyInterface.aidl'])");
    getConfiguredTarget("//java/android:idl1");
    getConfiguredTarget("//java/android:idl2");
  }

  @Test
  public void testIdeInfoProvider() throws Exception {
    scratch.file("java/android/BUILD",
        "genrule(name='genrule', srcs=[], outs=['assets/genrule.so'], cmd='')",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                idl_srcs = [ 'MyInterface.aidl' ],",
        "                resource_files = glob(['res/**']),",
        "                assets_dir = 'assets',",
        "                assets = glob(['assets/**']) + [':genrule']",
        "                )");
    scratch.file("java/android/res/values/strings.xml",
        "<resources><string name = 'hello'>Hello Android!</string></resources>");
    scratch.file("java/android/assets/values/orc.txt",
        "Nabu nabu!");
    ConfiguredTarget target = getConfiguredTarget("//java/android:r");
    final AndroidIdeInfoProvider provider = target.getProvider(AndroidIdeInfoProvider.class);
    assertEquals(
        ImmutableList.of(
            SourceDirectory.fromSourceRoot(
                rootDirectory.asFragment(),
                new PathFragment("java/android/assets")),
            SourceDirectory.fromRoot(
                targetConfig.getGenfilesDirectory(RepositoryName.MAIN),
                new PathFragment("java/android/assets"))),
        provider.getAssetDirs());
    assertEquals(
        ImmutableList.of(
            SourceDirectory.fromSourceRoot(
                rootDirectory.asFragment(),
                new PathFragment("java/android/res"))),
        provider.getResourceDirs());

    assertEquals(ImmutableList.of(SourceDirectory.fromSourceRoot(
        rootDirectory.asFragment(),
        new PathFragment("java/android")
    )), provider.getIdlImports());

    Set<Artifact> artifactClosure = actionsTestUtil().artifactClosureOf(getFilesToBuild(target));
    assertEquals(
        ActionsTestUtil.getFirstArtifactEndingWith(artifactClosure,
            "java/android/AndroidManifest.xml"),
        provider.getManifest());
    ResourceContainer resources = getOnlyElement(getConfiguredTarget("//java/android:r")
        .getProvider(AndroidResourcesProvider.class).getDirectAndroidResources());
    assertEquals(resources.getManifest(),
        provider.getGeneratedManifest());
  }

  @Test
  public void testIdeInfoProviderOutsideJavaRoot() throws Exception {
    String rootPath = "research/handwriting/java/com/google/research/handwriting/";
    scratch.file(rootPath + "BUILD",
        "genrule(name='genrule', srcs=[], outs=['assets/genrule.so'], cmd='')",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                idl_srcs = [ 'MyInterface.aidl' ],",
        "                resource_files = glob(['res/**']),",
        "                assets_dir = 'assets',",
        "                assets = glob(['assets/**']) + [':genrule']",
        "                )");
    scratch.file(rootPath + "res/values/strings.xml",
        "<resources><string name = 'hello'>Hello Android!</string></resources>");
    scratch.file(rootPath + "/assets/values/orc.txt",
        "Nabu nabu!");
    ConfiguredTarget target = getConfiguredTarget(
        "//research/handwriting/java/com/google/research/handwriting:r");
    final AndroidIdeInfoProvider provider = target.getProvider(AndroidIdeInfoProvider.class);
    assertEquals(
        ImmutableList.of(
            SourceDirectory.fromSourceRoot(
                rootDirectory.asFragment(),
                new PathFragment("research/handwriting/java/com/google/research/handwriting/assets")
            ),
            SourceDirectory.fromRoot(
                targetConfig.getGenfilesDirectory(RepositoryName.MAIN),
                new PathFragment("research/handwriting/java/com/google/research/handwriting/assets")
            )
        ),
        provider.getAssetDirs());
    assertEquals(
        ImmutableList.of(
            SourceDirectory.fromSourceRoot(
                rootDirectory.asFragment(),
                new PathFragment("research/handwriting/java/com/google/research/handwriting/res")
            )
        ),
        provider.getResourceDirs());

    assertEquals(ImmutableList.of(SourceDirectory.fromSourceRoot(
        rootDirectory.asFragment(),
        new PathFragment("research/handwriting/java/com/google/research/handwriting")
    )), provider.getIdlImports());

    Set<Artifact> artifactClosure = actionsTestUtil().artifactClosureOf(getFilesToBuild(target));
    assertEquals(
        ActionsTestUtil.getFirstArtifactEndingWith(artifactClosure,
            "handwriting/AndroidManifest.xml"),
        provider.getManifest());
    ResourceContainer resources = getOnlyElement(
        getConfiguredTarget("//research/handwriting/java/com/google/research/handwriting:r")
            .getProvider(AndroidResourcesProvider.class).getDirectAndroidResources());
    assertEquals(resources.getManifest(),
        provider.getGeneratedManifest());
  }

  @Test
  public void testIdeInfoProviderGeneratedIdl() throws Exception {
    scratch.file("java/android/BUILD",
        "genrule(name='genrule', srcs=[], outs=['assets/genrule.so'], cmd='')",
        "genrule(name = 'idl',",
        "        outs = ['MyGeneratedInterface.aidl'],",
        "        cmd = 'touch $@')",
        "android_library(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                idl_srcs = [ ':idl' ],",
        "                idl_parcelables = [ 'MyInterface.aidl' ],",
        "                resource_files = glob(['res/**']),",
        "                assets_dir = 'assets',",
        "                assets = glob(['assets/**']) + [':genrule']",
        "                )");
    scratch.file("java/android/res/values/strings.xml",
        "<resources><string name = 'hello'>Hello Android!</string></resources>");
    scratch.file("java/android/assets/values/orc.txt",
        "Nabu nabu!");
    ConfiguredTarget target = getConfiguredTarget("//java/android:r");
    final AndroidIdeInfoProvider provider = target.getProvider(AndroidIdeInfoProvider.class);
    assertEquals(
        ImmutableList.of(
            SourceDirectory.fromSourceRoot(
                rootDirectory.asFragment(),
                new PathFragment("java/android/assets")),
            SourceDirectory.fromRoot(
                targetConfig.getGenfilesDirectory(RepositoryName.MAIN),
                new PathFragment("java/android/assets"))
        ),
        provider.getAssetDirs());
    assertEquals(
        ImmutableList.of(
            SourceDirectory.fromSourceRoot(
                rootDirectory.asFragment(),
                new PathFragment("java/android/res"))
        ),
        provider.getResourceDirs());

    assertEquals(
        ImmutableList.of(
            SourceDirectory.fromSourceRoot(
                rootDirectory.asFragment(),
                new PathFragment("java/android")),
            SourceDirectory.fromRoot(
                targetConfig.getGenfilesDirectory(RepositoryName.MAIN),
                new PathFragment("java/android"))
        ),
        provider.getIdlImports());

    Set<Artifact> artifactClosure = actionsTestUtil().artifactClosureOf(getFilesToBuild(target));
    assertEquals(
        ActionsTestUtil.getFirstArtifactEndingWith(artifactClosure,
            "java/android/AndroidManifest.xml"),
        provider.getManifest());
    ResourceContainer resources = getOnlyElement(getConfiguredTarget("//java/android:r")
        .getProvider(AndroidResourcesProvider.class).getDirectAndroidResources());
    assertEquals(resources.getManifest(),
        provider.getGeneratedManifest());
  }

  @Test
  public void testAndroidLibraryWithMessagesDoNotCrash() throws Exception {
    scratch.file("java/com/google/atest/BUILD",
        "filegroup(name = 'sources',",
        "          srcs = ['source.java', 'message.xmb'])",
        "android_library(name = 'alib',",
        "    srcs  = [':sources'])");
    getConfiguredTarget("//java/com/google/atest:alib");
  }

  @Test
  public void testMultipleDirectDependentResourceDirectories_LocalResources()
      throws Exception {
    scratch.file("java/android/resources/d1/BUILD",
        "android_library(name = 'd1',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['d1-res/values/strings.xml'],",
        "                assets = ['assets-d1/some/random/file'],",
        "                assets_dir = 'assets-d1',",
        "                deps = ['//java/android/resources/d2:d2'])");
    scratch.file("java/android/resources/d2/BUILD",
        "android_library(name = 'd2',",
        "                manifest = 'AndroidManifest.xml',",
        "                assets = ['assets-d2/some/random/file'],",
        "                assets_dir = 'assets-d2',",
        "                resource_files = ['d2-res/values/strings.xml'],",
        "                )");
    ConfiguredTarget resource = getConfiguredTarget("//java/android/resources/d1:d1");
    List<String> args = ((SpawnAction) getGeneratingAction(getResourceArtifact(resource)))
        .getArguments();
    assertPrimaryResourceDirs(resource, ImmutableList.of("java/android/resources/d1/d1-res"), args);
    Truth.assertThat(getDirectDependentResourceDirs(resource, args))
        .contains("java/android/resources/d2/d2-res");
    Truth.assertThat(getDependentAssetDirs(resource, "--directData", args))
        .contains("java/android/resources/d2/assets-d2");
    assertNoEvents();
  }


  @Test
  public void testTransitiveDependentResourceDirectories_LocalResources()
      throws Exception {
    scratch.file("java/android/resources/d1/BUILD",
        "android_library(name = 'd1',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['d1-res/values/strings.xml'],",
        "                assets = ['assets-d1/some/random/file'],",
        "                assets_dir = 'assets-d1',",
        "                deps = ['//java/android/resources/d2:d2'])");
    scratch.file("java/android/resources/d2/BUILD",
        "android_library(name = 'd2',",
        "                manifest = 'AndroidManifest.xml',",
        "                assets = ['assets-d2/some/random/file'],",
        "                assets_dir = 'assets-d2',",
        "                resource_files = ['d2-res/values/strings.xml'],",
        "                deps = ['//java/android/resources/d3:d3'],",
        "                )");
    scratch.file("java/android/resources/d3/BUILD",
        "android_library(name = 'd3',",
        "                manifest = 'AndroidManifest.xml',",
        "                assets = ['assets-d3/some/random/file'],",
        "                assets_dir = 'assets-d3',",
        "                resource_files = ['d3-res/values/strings.xml'],",
        "                )");

    ConfiguredTarget resource = getConfiguredTarget("//java/android/resources/d1:d1");
    List<String> args = ((SpawnAction) getGeneratingAction(getResourceArtifact(resource)))
        .getArguments();
    assertPrimaryResourceDirs(
        resource, ImmutableList.of("java/android/resources/d1/d1-res"), args);
    Truth.assertThat(getDirectDependentResourceDirs(resource, args))
        .contains("java/android/resources/d2/d2-res");
    Truth.assertThat(getDependentAssetDirs(resource, "--directData", args))
        .contains("java/android/resources/d2/assets-d2");
    Truth.assertThat(getTransitiveDependentResourceDirs(resource, args))
        .contains("java/android/resources/d3/d3-res");
    Truth.assertThat(getDependentAssetDirs(resource, "--data", args))
        .contains("java/android/resources/d3/assets-d3");
    assertNoEvents();
  }

  @Test
  public void testCustomJavacopts() throws Exception {
    scratch.file("java/android/BUILD",
        "android_library(name = 'a',",
        "                srcs = ['A.java'],",
        "                javacopts = ['-g:lines,source'],",
        "               )");

    JavaCompileAction javacAction =
        (JavaCompileAction) getGeneratingActionForLabel("//java/android:liba.jar");

    assertThat(javacAction.buildCommandLine()).contains("-g:lines,source");
  }

  // Regression test for b/23079127

  @Test
  public void testSrcjarStrictDeps() throws Exception {
    scratch.file("java/strict/BUILD",
        "android_library(name='a', srcs=['A.java'], deps=[':b'])",
        "android_library(name='b', srcs=['b.srcjar'], deps=[':c'])",
        "android_library(name='c', srcs=['C.java'])");

    JavaCompileAction javacAction =
        (JavaCompileAction) getGeneratingActionForLabel("//java/strict:liba.jar");

    assertThat(ActionsTestUtil.prettyArtifactNames(javacAction.getDirectJars()))
        .containsExactly("java/strict/libb-hjar.jar");
  }

  @Test
  public void testDisallowPrecompiledJars() throws Exception {
    checkError("java/precompiled", "library",
        // messages:
        "does not produce any android_library srcs files (expected .java or .srcjar)",
        // build file:
        "android_library(name = 'library',",
        "    srcs = [':jar'])",
        "filegroup(name = 'jar',",
        "    srcs = ['lib.jar'])");
  }

  @Test
  public void hjarPredecessors() throws Exception {
    scratch.file(
        "java/test/BUILD",
        "android_library(name = 'a', srcs = ['A.java'], deps = [':b'])",
        "android_library(name = 'b', srcs = ['B.java'])");

    useConfiguration("--java_header_compilation");
    Action a = getGeneratingActionForLabel("//java/test:liba.jar");
    List<String> inputs = ActionsTestUtil.prettyArtifactNames(a.getInputs());
    assertThat(inputs).doesNotContain("java/test/libb.jdeps");
    assertThat(inputs).contains("java/test/libb-hjar.jdeps");
  }

  @Test
  public void resourcesFromRuntimeDepsAreIncluded() throws Exception {
    scratch.file(
        "java/android/BUILD",
        "android_library(name = 'dummyParentLibrary',",
        "                deps = [':dummyLibraryOne', ':dummyLibraryTwo'],",
        "                srcs = ['libraryParent.java'])",
        "",
        "android_library(name = 'dummyLibraryOne',",
        "                exports_manifest = 1,",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['res/drawable/dummyResource1.png'],",
        "                srcs = ['libraryOne.java'])",
        "",
        "android_library(name = 'dummyLibraryTwo',",
        "                exports_manifest = 1,",
        "                neverlink = 1,",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['res/drawable/dummyResource2.png'],",
        "                deps = ['dummyLibraryNested'],",
        "                srcs = ['libraryTwo.java'])",
        "",
        "android_library(name = 'dummyLibraryNested',",
        "                exports_manifest = 1,",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['res/drawable/dummyResource1.png'],",
        "                srcs = ['libraryOne.java'])");

    ConfiguredTarget target = getConfiguredTarget("//java/android:dummyLibraryOne");
    AndroidLibraryAarProvider provider = target.getProvider(AndroidLibraryAarProvider.class);
    assertThat(provider).isNotNull();

    target = getConfiguredTarget("//java/android:dummyLibraryTwo");
    provider = target.getProvider(AndroidLibraryAarProvider.class);
    assertThat(provider).isNull();

    target = getConfiguredTarget("//java/android:dummyParentLibrary");
    provider = target.getProvider(AndroidLibraryAarProvider.class);
    assertThat(provider).isNotNull();
    assertThat(provider.getTransitiveAars()).hasSize(1);
  }
}
