// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static java.util.Arrays.stream;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.Tool;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.RuleConfiguredTargetValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.List;
import java.util.stream.Stream;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for cc_binary with treeArtifacts, ThinLTO and separate obj dir for thinlto. */
@RunWith(JUnit4.class)
public class CcBinaryThinLtoObjDirTest extends BuildViewTestCase {

  private String targetName = "bin";

  private ConfiguredTarget getCurrentTarget() throws Exception {
    return getConfiguredTarget("//pkg:" + targetName);
  }

  private CppLinkAction getLinkAction() throws Exception {
    ConfiguredTarget pkg = getCurrentTarget();
    Artifact pkgArtifact = getFilesToBuild(pkg).getSingleton();
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(pkgArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(pkgArtifact);
    return linkAction;
  }

  private LtoBackendAction getBackendAction(String path) throws Exception {
    return (LtoBackendAction) getPredecessorByInputName(getLinkAction(), path);
  }

  private String getRootExecPath() throws Exception {
    ConfiguredTarget pkg = getCurrentTarget();
    Artifact pkgArtifact = getFilesToBuild(pkg).getSingleton();
    return pkgArtifact.getRoot().getExecPathString();
  }

  private CppLinkAction getIndexAction(LtoBackendAction backendAction) throws Exception {
    return (CppLinkAction)
        getPredecessorByInputName(
            backendAction,
            (backendAction.getPrimaryOutput().getExecPathString() + ".thinlto.bc")
                .replaceFirst(".lto-obj/", ".lto/"));
  }

  @Before
  public void createBasePkg() throws IOException {
    scratch.overwriteFile(
        "base/BUILD",
        "cc_library(name = 'system_malloc', visibility = ['//visibility:public'])",
        "cc_library(name = 'empty_lib', visibility = ['//visibility:public'])");
  }
  
  public void createBuildFiles(String... extraCcBinaryParameters) throws Exception {
    scratch.file(
        "pkg/BUILD",
        "load(':do_gen.bzl', 'test_generation', 'test_generation_2', 'test_generation_empty')",
        "package(features = ['thin_lto', 'use_lto_native_object_directory'])",
        "",
        "test_generation(",
        "          name = 'tree',",
        ")",
        "test_generation_2(",
        "          name = 'tree_2',",
        ")",
        "test_generation_empty(",
        "          name = 'tree_empty',",
        ")",
        "cc_binary(name = '" + targetName + "',",
        "          srcs = ['binfile.cc', ],",
        "          deps = [ ':lib', ':tree', ':tree_2', 'tree_empty'], ",
        String.join("", extraCcBinaryParameters),
        "          link_extra_lib = '//base:empty_lib', ",
        "          malloc = '//base:system_malloc')",
        "cc_library(name = 'lib',",
        "        srcs = ['libfile.cc'],",
        "        hdrs = ['libfile.h'],",
        "        linkstamp = 'linkstamp.cc',",
        "       )");
    scratch.file(
        "pkg/do_gen.bzl",
        "def _create_cc_impl(ctx):",
        "    directory = ctx.actions.declare_directory(ctx.label.name + \"_gen_cc\")",
        "    ctx.actions.run_shell(",
        "        command = \"echo -e '#include \\\"pkg/treelib.h\\\"\\n"
            + "Foo::~Foo() { }' > %s/file1.cc\" % directory.path,",
        "        outputs=[directory]",
        "    )",
        "    return DefaultInfo(files=depset([directory]))",
        "",
        "_create_cc = rule(implementation=_create_cc_impl)",
        "def test_generation(name):",
        "    _create_cc(name=name + \"_ccgen\")",
        "",
        "    native.cc_library(",
        "        name = name,",
        "        hdrs = [\"treelib.h\",],",
        "        srcs = [\":\" + name + \"_ccgen\",]",
        ")",
        "",
        "def _create_cc_impl_2(ctx):",
        "    directory = ctx.actions.declare_directory(ctx.label.name + \"_gen_cc_2\")",
        "    ctx.actions.run_shell(",
        "        command = \"echo -e '#include \\\"pkg/treelib_2.h\\\"\\n"
            + "int two() { return 2; }' > %s/file1.cc\" % directory.path +"
            + " \"echo -e '#include \\\"pkg/treelib_2.h\\\"\\n"
            + "int three() { return 3; }' > %s/file2.cc\" % directory.path,",
        "        outputs=[directory]",
        "    )",
        "    return DefaultInfo(files=depset([directory]))",
        "",
        "_create_cc_2 = rule(implementation=_create_cc_impl_2)",
        "def test_generation_2(name):",
        "    _create_cc_2(name=name + \"_ccgen_2\")",
        "",
        "    native.cc_library(",
        "        name = name,",
        "        hdrs = [\"treelib_2.h\",],",
        "        srcs = [\":\" + name + \"_ccgen_2\",]",
        ")",
        "",
        "def _create_cc_impl_empty(ctx):",
        "    directory = ctx.actions.declare_directory(ctx.label.name + \"_gen_cc_empty\")",
        "    ctx.actions.run_shell(",
        "        command = \"echo  'empty'\",",
        "        outputs=[directory]",
        "    )",
        "    return DefaultInfo(files=depset([directory]))",
        "",
        "_create_cc_empty = rule(implementation=_create_cc_impl_empty)",
        "def test_generation_empty(name):",
        "    _create_cc_empty(name=name + \"_ccgen_empty\")",
        "",
        "    native.cc_library(",
        "        name = name,",
        "        srcs = [\":\" + name + \"_ccgen_empty\",]",
        ")");

    scratch.file("pkg/treelib.h", "class Foo{ public:  ~Foo(); };");
    scratch.file("pkg/treelib_2.h", "int two(); int three();");

    scratch.file(
        "pkg/binfile.cc",
        "#include \"pkg/libfile.h\"",
        "#include \"pkg/treelib.h\"",
        "#include \"pkg/treelib_2.h\"",
        "int main() {",
        "  Foo foo;",
        "  return pkg() + two() + three(); }");
    scratch.file("pkg/libfile.cc", "int pkg() { return 42; }");
    scratch.file("pkg/libfile.h", "int pkg();");
    scratch.file("pkg/linkstamp.cc");
  }

  public void createTestFiles(String extraTestParameters, String extraLibraryParameters)
      throws Exception {
    scratch.file(
        "pkg/BUILD",
        "load(':do_gen.bzl', 'test_generation')",
        "package(features = ['thin_lto', 'use_lto_native_object_directory'])",
        "",
        "test_generation(",
        "          name = 'tree',",
        ")",
        "cc_test(",
        "    name = 'bin_test',",
        "    srcs = ['bin_test.cc', ],",
        "    deps = [ ':lib', ':tree', ], ",
        extraTestParameters,
        "    link_extra_lib = '//base:empty_lib', ",
        "    malloc = '//base:system_malloc'",
        ")",
        "cc_test(",
        "    name = 'bin_test2',",
        "    srcs = ['bin_test2.cc', ],",
        "    deps = [ ':lib', ':tree', ], ",
        extraTestParameters,
        "    link_extra_lib = '//base:empty_lib', ",
        "    malloc = '//base:system_malloc'",
        ")",
        "cc_library(",
        "    name = 'lib',",
        "    srcs = ['libfile.cc'],",
        "    hdrs = ['libfile.h'],",
        extraLibraryParameters,
        "    linkstamp = 'linkstamp.cc',",
        ")");
    scratch.file(
        "pkg/do_gen.bzl",
        "def _create_cc_impl(ctx):",
        "    directory = ctx.actions.declare_directory(ctx.label.name + \"_gen_cc\")",
        "    ctx.actions.run_shell(",
        "        command = \"echo -e '#include \\\"pkg/treelib.h\\\"\\n"
            + "Foo::~Foo() { }' > %s/file.cc\" % directory.path,",
        "        outputs=[directory]",
        "    )",
        "    return DefaultInfo(files=depset([directory]))",
        "",
        "_create_cc = rule(implementation=_create_cc_impl)",
        "def test_generation(name):",
        "    _create_cc(name=name + \"_ccgen\")",
        "",
        "    native.cc_library(",
        "        name = name,",
        "        hdrs = [\"treelib.h\",],",
        "        srcs = [\":\" + name + \"_ccgen\",]",
        ")");
    scratch.file("pkg/treelib.h", "class Foo{ public:  ~Foo(); };");
    scratch.file(
        "pkg/bin_test.cc",
        "#include \"pkg/libfile.h\"",
        "#include \"pkg/treelib.h\"",
        "int main() { Foo foo; return pkg(); }");
    scratch.file(
        "pkg/bin_test2.cc",
        "#include \"pkg/libfile.h\"",
        "#include \"pkg/treelib.h\"",
        "int main() { Foo foo; return pkg(); }");
    scratch.file("pkg/libfile.cc", "int pkg() { return 42; }");
    scratch.file("pkg/libfile.h", "int pkg();");
    scratch.file("pkg/linkstamp.cc");
  }

  @Test
  public void testActionGraph() throws Exception {
    createBuildFiles();
    setupThinLTOCrosstool(CppRuleClasses.SUPPORTS_PIC);

    /*
    We follow the chain from the final product backwards.

    binary <=[Link]=
    .lto-obj/...o <=[LTOBackend]=
    {.o.thinlto.bc,.o.imports} <=[LTOIndexing]=
    .o <= [CppCompile] .cc
    */
    ConfiguredTarget pkg = getCurrentTarget();
    SpawnAction linkAction = getLinkAction();
    String rootExecPath = getRootExecPath();

    assertThat(ActionsTestUtil.getFirstArtifactEndingWith(linkAction.getInputs(), "linkstamp.o"))
        .isNotNull();

    List<String> commandLine = linkAction.getArguments();
    String prefix =
        getTargetConfiguration().getOutputDirectory(RepositoryName.MAIN).getExecPathString();
    assertThat(commandLine)
        .containsAtLeast(
            prefix + "/bin/pkg/bin.lto.merged.o",
            "thinlto_param_file=" + prefix + "/bin/pkg/bin-lto-final.params")
        .inOrder();

    // We have no bitcode files: all files have pkg/bin.lto/
    for (String arg : commandLine) {
      if (arg.contains("_objs") && !arg.contains("linkstamp.o")) {
        assertThat(arg).contains("pkg/bin.lto");
      }
    }

    assertThat(artifactsToStrings(linkAction.getInputs()))
        .containsAtLeast(
            "bin pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o",
            "bin pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.o",
            "bin pkg/bin-lto-final.params");

    LtoBackendAction backendAction =
        getBackendAction("pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");

    assertThat(artifactsToStrings(backendAction.getInputs()))
        .containsAtLeast(
            "bin pkg/bin.lto/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o.thinlto.bc",
            "bin pkg/bin.lto/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o.imports");

    assertThat(backendAction.getArguments())
        .containsAtLeast(
            "thinlto_index="
                + prefix
                + "/bin/pkg/bin.lto/"
                + rootExecPath
                + "/pkg/_objs/bin/binfile.pic.o.thinlto.bc",
            "thinlto_output_object_file="
                + prefix
                + "/bin/pkg/bin.lto-obj/"
                + rootExecPath
                + "/pkg/_objs/bin/binfile.pic.o",
            "thinlto_input_bitcode_file=" + prefix + "/bin/pkg/_objs/bin/binfile.pic.o");

    CppLinkAction indexAction = getIndexAction(backendAction);

    RuleConfiguredTargetValue configuredTargetValue =
        (RuleConfiguredTargetValue)
            getSkyframeExecutor()
                .getEvaluator()
                .getExistingEntryAtCurrentlyEvaluatingVersion(
                    ConfiguredTargetKey.builder()
                        .setLabel(pkg.getLabel())
                        .setConfiguration(getConfiguration(pkg))
                        .build())
                .getValue();
    ImmutableList<ActionAnalysisMetadata> linkstampCompileActions =
        configuredTargetValue.getActions().stream()
            .filter(a -> a.getMnemonic().equals("CppLinkstampCompile"))
            .collect(toImmutableList());
    assertThat(linkstampCompileActions).hasSize(1);
    ActionAnalysisMetadata linkstampCompileAction = linkstampCompileActions.get(0);
    assertThat(indexAction.getInputs().toList())
        .containsNoneIn(linkstampCompileAction.getOutputs());

    assertThat(indexAction.getArguments())
        .doesNotContain("thinlto_param_file=" + prefix + "/bin/pkg/bin-lto-final.params");

    assertThat(artifactsToStrings(indexAction.getOutputs()))
        .containsAtLeast(
            "bin pkg/bin.lto/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o.imports",
            "bin pkg/bin.lto/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o.thinlto.bc",
            "bin pkg/bin.lto/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.o.imports",
            "bin pkg/bin.lto/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.o.thinlto.bc",
            "bin pkg/bin-lto-final.params");

    assertThat(indexAction.getMnemonic()).isEqualTo("CppLTOIndexing");

    assertThat(artifactsToStrings(indexAction.getInputs()))
        .containsAtLeast(
            "bin pkg/_objs/bin/binfile.pic.indexing.o", "bin pkg/_objs/lib/libfile.pic.indexing.o");

    CppCompileAction bitcodeAction =
        (CppCompileAction)
            getPredecessorByInputName(indexAction, "pkg/_objs/bin/binfile.pic.indexing.o");
    assertThat(bitcodeAction.getMnemonic()).isEqualTo("CppCompile");
    assertThat(bitcodeAction.getArguments())
        .contains("lto_indexing_bitcode=" + prefix + "/bin/pkg/_objs/bin/binfile.pic.indexing.o");
  }

  @Test
  public void testLinkshared() throws Exception {
    targetName = "bin.so";
    createBuildFiles("linkshared = 1,");
    setupThinLTOCrosstool(CppRuleClasses.SUPPORTS_PIC);
    useConfiguration();

    CppLinkAction linkAction = getLinkAction();
    String rootExecPath = getRootExecPath();

    Action backendAction =
        getPredecessorByInputName(
            linkAction, "pkg/bin.so.lto-obj/" + rootExecPath + "/pkg/_objs/bin.so/binfile.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");
  }

  @Test
  public void testNoLinkstatic() throws Exception {
    createBuildFiles("linkstatic = 0,");
    setupThinLTOCrosstool(
        CppRuleClasses.SUPPORTS_DYNAMIC_LINKER,
        CppRuleClasses.SUPPORTS_PIC,
        CppRuleClasses.SUPPORTS_INTERFACE_SHARED_LIBRARIES);

    /*
    We follow the chain from the final product backwards to verify intermediate actions.

    binary <=[Link]=
    .ifso <=[SolibSymlink]=
    _S...ifso <=[SolibSymlink]=
    .ifso <=[Link]=
    .lto-obj/...o <=[LTOBackend]=
    {.o.thinlto.bc,.o.imports} <=[LTOIndexing]=
    .o <= [CppCompile] .cc
    */
    SpawnAction linkAction = getLinkAction();
    String rootExecPath = getRootExecPath();

    List<String> commandLine = linkAction.getArguments();
    String prefix =
        getTargetConfiguration().getOutputDirectory(RepositoryName.MAIN).getExecPathString();

    assertThat(commandLine).contains("-Wl,@" + prefix + "/bin/pkg/bin-lto-final.params");

    // We have no bitcode files: all files have pkg/bin.lto/
    for (String arg : commandLine) {
      if (arg.contains("_objs") && !arg.contains("linkstamp.o")) {
        assertThat(arg).contains("pkg/bin.lto");
      }
    }

    assertThat(artifactsToStrings(linkAction.getInputs()))
        .containsAtLeast(
            "bin pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o",
            "bin _solib_k8/libpkg_Sliblib.ifso",
            "bin pkg/bin-lto-final.params");

    SolibSymlinkAction solibSymlinkAction =
        (SolibSymlinkAction) getPredecessorByInputName(linkAction, "_solib_k8/libpkg_Sliblib.ifso");
    assertThat(solibSymlinkAction.getMnemonic()).isEqualTo("SolibSymlink");

    CppLinkAction libLinkAction =
        (CppLinkAction) getPredecessorByInputName(solibSymlinkAction, "bin/pkg/liblib.ifso");
    assertThat(libLinkAction.getMnemonic()).isEqualTo("CppLink");

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                libLinkAction,
                "pkg/liblib.so.lto-obj/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");

    assertThat(artifactsToStrings(backendAction.getInputs()))
        .contains(
            "bin pkg/liblib.so.lto/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.o.thinlto.bc");

    assertThat(backendAction.getArguments())
        .containsAtLeast(
            "thinlto_index="
                + prefix
                + "/bin/pkg/liblib.so.lto/"
                + rootExecPath
                + "/pkg/_objs/lib/libfile.pic.o.thinlto.bc",
            "thinlto_output_object_file="
                + prefix
                + "/bin/pkg/liblib.so.lto-obj/"
                + rootExecPath
                + "/pkg/_objs/lib/libfile.pic.o",
            "thinlto_input_bitcode_file=" + prefix + "/bin/pkg/_objs/lib/libfile.pic.o");

    CppLinkAction indexAction =
        (CppLinkAction)
            getPredecessorByInputName(
                backendAction,
                "pkg/liblib.so.lto/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.o.thinlto.bc");

    assertThat(artifactsToStrings(indexAction.getOutputs()))
        .containsAtLeast(
            "bin pkg/liblib.so.lto/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.o.imports",
            "bin pkg/liblib.so.lto/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.o.thinlto.bc",
            "bin pkg/liblib.so-lto-final.params");

    assertThat(indexAction.getMnemonic()).isEqualTo("CppLTOIndexing");

    assertThat(artifactsToStrings(indexAction.getInputs()))
        .contains("bin pkg/_objs/lib/libfile.pic.indexing.o");

    CppCompileAction bitcodeAction =
        (CppCompileAction)
            getPredecessorByInputName(indexAction, "pkg/_objs/lib/libfile.pic.indexing.o");
    assertThat(bitcodeAction.getMnemonic()).isEqualTo("CppCompile");
    assertThat(bitcodeAction.getArguments())
        .contains("lto_indexing_bitcode=" + prefix + "/bin/pkg/_objs/lib/libfile.pic.indexing.o");
  }

  /** Helper method to get the root prefix from the given dwpFile. */
  private static PathFragment dwpRootPrefix(Artifact dwpFile) throws Exception {
    return dwpFile
        .getExecPath()
        .subFragment(
            0, dwpFile.getExecPath().segmentCount() - dwpFile.getRootRelativePath().segmentCount());
  }

  /** Helper method that checks that a .dwp has the expected generating action structure. */
  private void validateDwp(
      Artifact dwpFile, CcToolchainProvider toolchain, List<String> expectedInputs)
      throws Exception {
    SpawnAction dwpAction = (SpawnAction) getGeneratingAction(dwpFile);
    String dwpToolPath =
        CcToolchainProvider.getToolPathString(
            toolchain.getToolPaths(),
            Tool.DWP,
            toolchain.getCcToolchainLabel(),
            toolchain.getToolchainIdentifier());
    assertThat(dwpAction.getMnemonic()).isEqualTo("CcGenerateDwp");
    assertThat(dwpToolPath).isEqualTo(dwpAction.getCommandFilename());
    List<String> commandArgs = dwpAction.getArguments();
    // The first argument should be the command being executed.
    assertThat(dwpToolPath).isEqualTo(commandArgs.get(0));
    // The final two arguments should be "-o dwpOutputFile".
    assertThat(commandArgs.subList(commandArgs.size() - 2, commandArgs.size()))
        .containsExactly("-o", dwpFile.getExecPathString())
        .inOrder();
    // The remaining arguments should be the set of .dwo inputs (in any order).
    assertThat(commandArgs.subList(1, commandArgs.size() - 2))
        .containsExactlyElementsIn(expectedInputs);
  }

  @Test
  public void testFission() throws Exception {
    createBuildFiles();
    setupThinLTOCrosstool(CppRuleClasses.SUPPORTS_PIC, CppRuleClasses.PER_OBJECT_DEBUG_INFO);
    useConfiguration("--fission=yes", "--copt=-g0");

    String rootExecPath = getRootExecPath();
    LtoBackendAction backendAction =
        getBackendAction("pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");
    assertThat(artifactsToStrings(backendAction.getOutputs()))
        .containsExactly(
            "bin pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o",
            "bin pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.dwo");

    assertThat(backendAction.getArguments()).containsAtLeast("-g0", "per_object_debug_info_option");

    backendAction =
        getBackendAction("pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");
    assertThat(artifactsToStrings(backendAction.getOutputs()))
        .containsExactly(
            "bin pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.o",
            "bin pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.dwo");

    assertThat(backendAction.getArguments()).contains("per_object_debug_info_option");

    // Now check the dwp action.
    ConfiguredTarget pkg = getCurrentTarget();
    Artifact dwpFile = getFileConfiguredTarget(pkg.getLabel() + ".dwp").getArtifact();
    PathFragment rootPrefix = dwpRootPrefix(dwpFile);
    RuleContext ruleContext = getRuleContext(pkg);
    CcToolchainProvider toolchain = CppHelper.getToolchain(ruleContext);
    validateDwp(
        dwpFile,
        toolchain,
        ImmutableList.of(
            rootPrefix + "/pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.dwo",
            rootPrefix + "/pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.dwo"));
  }

  @Test
  public void testNoLinkstaticFission() throws Exception {
    createBuildFiles("linkstatic = 0,");
    setupThinLTOCrosstool(
        CppRuleClasses.SUPPORTS_PIC,
        CppRuleClasses.SUPPORTS_INTERFACE_SHARED_LIBRARIES,
        CppRuleClasses.SUPPORTS_DYNAMIC_LINKER,
        CppRuleClasses.PER_OBJECT_DEBUG_INFO);
    useConfiguration("--fission=yes");

    ConfiguredTarget pkg = getConfiguredTarget("//pkg:bin");
    Artifact pkgArtifact = getFilesToBuild(pkg).getSingleton();
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(pkgArtifact);
    String rootExecPath = pkgArtifact.getRoot().getExecPathString();

    SolibSymlinkAction solibSymlinkAction =
        (SolibSymlinkAction) getPredecessorByInputName(linkAction, "_solib_k8/libpkg_Sliblib.ifso");
    assertThat(solibSymlinkAction.getMnemonic()).isEqualTo("SolibSymlink");

    CppLinkAction libLinkAction =
        (CppLinkAction) getPredecessorByInputName(solibSymlinkAction, "bin/pkg/liblib.ifso");
    assertThat(libLinkAction.getMnemonic()).isEqualTo("CppLink");

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                libLinkAction,
                "pkg/liblib.so.lto-obj/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");
    assertThat(artifactsToStrings(backendAction.getOutputs()))
        .containsExactly(
            "bin pkg/liblib.so.lto-obj/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.o",
            "bin pkg/liblib.so.lto-obj/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.dwo");

    assertThat(backendAction.getArguments()).contains("per_object_debug_info_option");

    // Check the dwp action.
    Artifact dwpFile = getFileConfiguredTarget(pkg.getLabel() + ".dwp").getArtifact();
    PathFragment rootPrefix = dwpRootPrefix(dwpFile);
    RuleContext ruleContext = getRuleContext(pkg);
    CcToolchainProvider toolchain = CppHelper.getToolchain(ruleContext);
    validateDwp(
        dwpFile,
        toolchain,
        ImmutableList.of(
            rootPrefix + "/pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.dwo"));
  }

  @Test
  public void testLinkstaticCcTestFission() throws Exception {
    createTestFiles("linkstatic = 1,", "");

    setupThinLTOCrosstool(
        CppRuleClasses.SUPPORTS_PIC,
        CppRuleClasses.THIN_LTO_LINKSTATIC_TESTS_USE_SHARED_NONLTO_BACKENDS,
        CppRuleClasses.PER_OBJECT_DEBUG_INFO);
    useConfiguration(
        "--fission=yes", "--features=thin_lto_linkstatic_tests_use_shared_nonlto_backends");

    ConfiguredTarget pkg = getConfiguredTarget("//pkg:bin_test");
    Artifact pkgArtifact = getFilesToBuild(pkg).getSingleton();
    String rootExecPath = pkgArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(pkgArtifact);

    // All backends should be shared non-LTO in this case
    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction,
                "shared.nonlto-obj/" + rootExecPath + "/pkg/_objs/bin_test/bin_test.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");
    assertThat(artifactsToStrings(backendAction.getOutputs()))
        .containsExactly(
            "bin shared.nonlto-obj/" + rootExecPath + "/pkg/_objs/bin_test/bin_test.pic.o",
            "bin shared.nonlto-obj/" + rootExecPath + "/pkg/_objs/bin_test/bin_test.pic.dwo");

    assertThat(backendAction.getArguments()).contains("per_object_debug_info_option");

    backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "shared.nonlto-obj/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");
    assertThat(backendAction.getArguments()).contains("-fPIC");
    assertThat(artifactsToStrings(backendAction.getOutputs()))
        .containsExactly(
            "bin shared.nonlto-obj/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.o",
            "bin shared.nonlto-obj/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.dwo");

    assertThat(backendAction.getArguments()).contains("per_object_debug_info_option");

    // Now check the dwp action.
    Artifact dwpFile = getFileConfiguredTarget(pkg.getLabel() + ".dwp").getArtifact();
    PathFragment rootPrefix = dwpRootPrefix(dwpFile);
    RuleContext ruleContext = getRuleContext(pkg);
    CcToolchainProvider toolchain = CppHelper.getToolchain(ruleContext);
    validateDwp(
        dwpFile,
        toolchain,
        ImmutableList.of(
            rootPrefix + "/shared.nonlto-obj/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.dwo",
            rootPrefix
                + "/shared.nonlto-obj/"
                + rootExecPath
                + "/pkg/_objs/bin_test/bin_test.pic.dwo"));
  }

  @Test
  public void testLinkstaticCcTest() throws Exception {
    createTestFiles("linkstatic = 1,", "");

    setupThinLTOCrosstool(
        CppRuleClasses.SUPPORTS_PIC,
        CppRuleClasses.THIN_LTO_LINKSTATIC_TESTS_USE_SHARED_NONLTO_BACKENDS,
        CppRuleClasses.PER_OBJECT_DEBUG_INFO);
    useConfiguration("--features=thin_lto_linkstatic_tests_use_shared_nonlto_backends");

    ConfiguredTarget pkg = getConfiguredTarget("//pkg:bin_test");
    Artifact pkgArtifact = getFilesToBuild(pkg).getSingleton();
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(pkgArtifact);

    ConfiguredTarget pkg2 = getConfiguredTarget("//pkg:bin_test2");
    Artifact pkgArtifact2 = getFilesToBuild(pkg2).getSingleton();
    CppLinkAction linkAction2 = (CppLinkAction) getGeneratingAction(pkgArtifact2);

    // All backends should be shared non-LTO in this case
    String rootExecPath1 = pkgArtifact.getRoot().getExecPathString();
    String rootExecPath2 = pkgArtifact.getRoot().getExecPathString();
    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction,
                "shared.nonlto-obj/" + rootExecPath1 + "/pkg/_objs/bin_test/bin_test.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");

    backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "shared.nonlto-obj/" + rootExecPath1 + "/pkg/_objs/lib/libfile.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");
    assertThat(backendAction.getArguments()).contains("-fPIC");

    LtoBackendAction backendAction2 =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction2, "shared.nonlto-obj/" + rootExecPath2 + "/pkg/_objs/lib/libfile.pic.o");
    assertThat(backendAction2.getMnemonic()).isEqualTo("CcLtoBackendCompile");

    assertThat(backendAction).isEqualTo(backendAction2);
  }

  @Test
  public void testTestOnlyTarget() throws Exception {
    createBuildFiles("testonly = 1,");

    setupThinLTOCrosstool(
        CppRuleClasses.SUPPORTS_PIC,
        CppRuleClasses.THIN_LTO_LINKSTATIC_TESTS_USE_SHARED_NONLTO_BACKENDS);
    useConfiguration("--features=thin_lto_linkstatic_tests_use_shared_nonlto_backends");

    ConfiguredTarget pkg = getConfiguredTarget("//pkg:bin");
    Artifact pkgArtifact = getFilesToBuild(pkg).getSingleton();
    String rootExecPath = pkgArtifact.getRoot().getExecPathString();
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(pkgArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "shared.nonlto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");
  }

  @Test
  public void testUseSharedAllLinkstatic() throws Exception {
    createBuildFiles();

    setupThinLTOCrosstool(
        CppRuleClasses.THIN_LTO_ALL_LINKSTATIC_USE_SHARED_NONLTO_BACKENDS,
        CppRuleClasses.SUPPORTS_PIC);
    useConfiguration("--features=thin_lto_all_linkstatic_use_shared_nonlto_backends");

    ConfiguredTarget pkg = getConfiguredTarget("//pkg:bin");
    Artifact pkgArtifact = getFilesToBuild(pkg).getSingleton();
    String rootExecPath = pkgArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(pkgArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "shared.nonlto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");
  }

  private Action getPredecessorByInputName(Action action, String str) {
    for (Artifact a : action.getInputs().toList()) {
      if (a.getExecPathString().contains(str)) {
        return getGeneratingAction(a);
      }
    }
    return null;
  }
  
  @Test
  public void testFdoInstrument() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "package(features = ['thin_lto', 'use_lto_native_object_directory'])",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          malloc = '//base:system_malloc')");

    scratch.file("pkg/binfile.cc", "int main() {}");

    setupThinLTOCrosstool(CppRuleClasses.SUPPORTS_PIC, CppRuleClasses.FDO_INSTRUMENT);
    useConfiguration("--fdo_instrument=profiles");

    ConfiguredTarget pkg = getConfiguredTarget("//pkg:bin");

    Artifact pkgArtifact = getFilesToBuild(pkg).getSingleton();
    String rootExecPath = pkgArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(pkgArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(pkgArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o");
    // If the LtoBackendAction incorrectly tries to add the fdo_instrument
    // feature, we will fail with an "unknown variable 'fdo_instrument_path'"
    // error. But let's also explicitly confirm that the fdo_instrument
    // option didn't end up here.
    assertThat(backendAction.getArguments()).doesNotContain("fdo_instrument_option");
  }

  @Test
  public void testLtoIndexOpt() throws Exception {
    createBuildFiles();

    setupThinLTOCrosstool(CppRuleClasses.SUPPORTS_PIC);
    useConfiguration("--ltoindexopt=anltoindexopt");

    /*
    We follow the chain from the final product backwards.

    binary <=[Link]=
    .lto-obj/...o <=[LTOBackend]=
    {.o.thinlto.bc,.o.imports} <=[LTOIndexing]=
    .o <= [CppCompile] .cc
    */
    ConfiguredTarget pkg = getConfiguredTarget("//pkg:bin");

    Artifact pkgArtifact = getFilesToBuild(pkg).getSingleton();
    String rootExecPath = pkgArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(pkgArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(pkgArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");

    CppLinkAction indexAction =
        (CppLinkAction)
            getPredecessorByInputName(
                backendAction,
                "pkg/bin.lto/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o.thinlto.bc");

    assertThat(indexAction.getArguments()).contains("anltoindexopt");
  }

  @Test
  public void testLtoStandaloneCommandLines() throws Exception {
    createBuildFiles();

    setupThinLTOCrosstool(CppRuleClasses.SUPPORTS_PIC);
    useConfiguration(
        "--ltoindexopt=anltoindexopt",
        "--incompatible_make_thinlto_command_lines_standalone",
        "--features=thin_lto",
        "--features=use_lto_native_object_directory");

    /*
    We follow the chain from the final product backwards.

    binary <=[Link]=
    .lto-obj/...o <=[LTOBackend]=
    {.o.thinlto.bc,.o.imports} <=[LTOIndexing]=
    .o <= [CppCompile] .cc
    */
    ConfiguredTarget pkg = getConfiguredTarget("//pkg:bin");

    Artifact pkgArtifact = getFilesToBuild(pkg).getSingleton();
    String rootExecPath = pkgArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(pkgArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(pkgArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");

    CppLinkAction indexAction =
        (CppLinkAction)
            getPredecessorByInputName(
                backendAction,
                "pkg/bin.lto/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o.thinlto.bc");

    assertThat(indexAction.getArguments())
        .contains("--i_come_from_standalone_lto_index=anltoindexopt");
  }

  @Test
  public void testCopt() throws Exception {
    createBuildFiles();

    setupThinLTOCrosstool(CppRuleClasses.SUPPORTS_PIC);
    useConfiguration("--copt=acopt");

    /*
    We follow the chain from the final product backwards.

    binary <=[Link]=
    .lto-obj/...o <=[LTOBackend]=
    */
    ConfiguredTarget pkg = getConfiguredTarget("//pkg:bin");

    Artifact pkgArtifact = getFilesToBuild(pkg).getSingleton();
    String rootExecPath = pkgArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(pkgArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(pkgArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");
    assertThat(backendAction.getArguments()).contains("acopt");
  }

  @Test
  public void testPerFileCopt() throws Exception {
    createBuildFiles();
    setupThinLTOCrosstool(CppRuleClasses.SUPPORTS_PIC);
    useConfiguration(
        "--per_file_copt=binfile\\.cc@copt1",
        "--per_file_copt=libfile\\.cc@copt2",
        "--per_file_copt=.*\\.cc,-binfile\\.cc@copt2");

    /*
    We follow the chain from the final product backwards.

    binary <=[Link]=
    .lto-obj/...o <=[LTOBackend]=
    */
    ConfiguredTarget pkg = getConfiguredTarget("//pkg:bin");
    Artifact pkgArtifact = getFilesToBuild(pkg).getSingleton();
    String rootExecPath = pkgArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(pkgArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(pkgArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o");
    assertThat(backendAction.getArguments()).contains("copt1");
    assertThat(backendAction.getArguments()).doesNotContain("copt2");

    backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.o");
    assertThat(backendAction.getArguments()).doesNotContain("copt1");
    assertThat(backendAction.getArguments()).contains("copt2");
  }

  @Test
  public void testCoptNoCoptAttributes() throws Exception {
    createBuildFiles("copts = ['acopt', 'nocopt1'], nocopts = 'nocopt1|nocopt2',");

    setupThinLTOCrosstool(CppRuleClasses.SUPPORTS_PIC);
    useConfiguration("--copt=nocopt2", "--noincompatible_disable_nocopts");

    /*
    We follow the chain from the final product backwards.

    binary <=[Link]=
    .lto-obj/...o <=[LTOBackend]=
    */
    ConfiguredTarget pkg = getConfiguredTarget("//pkg:bin");

    Artifact pkgArtifact = getFilesToBuild(pkg).getSingleton();
    String rootExecPath = pkgArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(pkgArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(pkgArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");
    assertThat(backendAction.getArguments()).contains("acopt");
    // TODO(b/122303926): Remove when nocopts are removed, or uncomment and fix if not removing.
    // assertThat(backendAction.getArguments()).doesNotContain("nocopt1");
    // assertThat(backendAction.getArguments()).doesNotContain("nocopt2");
  }

  @Test
  public void testLtoBackendOpt() throws Exception {
    createBuildFiles();

    setupThinLTOCrosstool(CppRuleClasses.SUPPORTS_PIC, MockCcSupport.USER_COMPILE_FLAGS);
    useConfiguration("--ltobackendopt=anltobackendopt");

    /*
    We follow the chain from the final product backwards.

    binary <=[Link]=
    .lto-obj/...o <=[LTOBackend]=
    */
    ConfiguredTarget pkg = getConfiguredTarget("//pkg:bin");

    Artifact pkgArtifact = getFilesToBuild(pkg).getSingleton();
    String rootExecPath = pkgArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(pkgArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(pkgArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");
    assertThat(backendAction.getArguments())
        .containsAtLeast("--default-compile-flag", "anltobackendopt");
  }

  @Test
  public void testPerFileLtoBackendOpt() throws Exception {
    createBuildFiles();

    setupThinLTOCrosstool(CppRuleClasses.SUPPORTS_PIC);
    useConfiguration(
        "--per_file_ltobackendopt=binfile\\.pic\\.o@ltobackendopt1",
        "--per_file_ltobackendopt=.*\\.o,-binfile\\.pic\\.o@ltobackendopt2");

    /*
    We follow the chain from the final product backwards.

    binary <=[Link]=
    .lto-obj/...o <=[LTOBackend]=
    */
    ConfiguredTarget pkg = getConfiguredTarget("//pkg:bin");
    Artifact pkgArtifact = getFilesToBuild(pkg).getSingleton();
    String rootExecPath = pkgArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(pkgArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(pkgArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o");
    assertThat(backendAction.getArguments()).contains("ltobackendopt1");
    assertThat(backendAction.getArguments()).doesNotContain("ltobackendopt2");

    backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.o");
    assertThat(backendAction.getArguments()).doesNotContain("ltobackendopt1");
    assertThat(backendAction.getArguments()).contains("ltobackendopt2");
  }

  @Test
  public void testNoUseLtoIndexingBitcodeFile() throws Exception {
    createBuildFiles();

    setupThinLTOCrosstool(
        CppRuleClasses.NO_USE_LTO_INDEXING_BITCODE_FILE, CppRuleClasses.SUPPORTS_PIC);
    useConfiguration(
        "--features=no_use_lto_indexing_bitcode_file",
        "--features=use_lto_native_object_directory");
    String rootExecPath = getRootExecPath();

    /*
    We follow the chain from the final product backwards.

    binary <=[Link]=
    .lto-obj/...o <=[LTOBackend]=
    {.o.thinlto.bc,.o.imports} <=[LTOIndexing]=
    .o <= [CppCompile] .cc
    */
    CppLinkAction indexAction =
        getIndexAction(
            getBackendAction("pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o"));

    assertThat(indexAction.getArguments()).doesNotContain("object_suffix_replace");

    assertThat(artifactsToStrings(indexAction.getInputs()))
        .containsAtLeast("bin pkg/_objs/bin/binfile.pic.o", "bin pkg/_objs/lib/libfile.pic.o");

    CppCompileAction bitcodeAction =
        (CppCompileAction) getPredecessorByInputName(indexAction, "pkg/_objs/bin/binfile.pic.o");
    assertThat(bitcodeAction.getArguments()).doesNotContain("lto_indexing_bitcode=");
  }

  @Test
  public void testAutoFdo() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "package(features = ['thin_lto', 'use_lto_native_object_directory'])",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          malloc = '//base:system_malloc')");

    scratch.file("pkg/binfile.cc", "int main() {}");
    scratch.file("pkg/profile.afdo", "");

    setupThinLTOCrosstool(CppRuleClasses.AUTOFDO);
    useConfiguration("--fdo_optimize=/pkg/profile.afdo", "--compilation_mode=opt");

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");

    // Checks that -fauto-profile is added to the LtoBackendAction.
    assertThat(Joiner.on(" ").join(backendAction.getArguments())).containsMatch(
        "-fauto-profile=[^ ]*/profile.afdo");
    assertThat(ActionsTestUtil.baseArtifactNames(backendAction.getInputs())).contains(
        "profile.afdo");
  }

  private void setupThinLTOCrosstool(String... extraFeatures) throws Exception {
    String[] allFeatures =
        Stream.concat(
                Stream.of(
                    CppRuleClasses.THIN_LTO,
                    CppRuleClasses.USE_LTO_NATIVE_OBJECT_DIRECTORY,
                    CppRuleClasses.SUPPORTS_START_END_LIB,
                    MockCcSupport.HOST_AND_NONHOST_CONFIGURATION_FEATURES),
                stream(extraFeatures))
            .toArray(String[]::new);
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withFeatures(allFeatures));
  }

  private void setupAutoFdoThinLtoCrosstool() throws Exception {
    setupThinLTOCrosstool(
        CppRuleClasses.AUTOFDO,
        CppRuleClasses.ENABLE_AFDO_THINLTO,
        CppRuleClasses.AUTOFDO_IMPLICIT_THINLTO);
  }

  /**
   * Tests that ThinLTO is not enabled for AFDO with LLVM without
   * --features=autofdo_implicit_thinlto.
   */
  @Test
  public void testAutoFdoNoImplicitThinLto() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          malloc = '//base:system_malloc')");

    scratch.file("pkg/binfile.cc", "int main() {}");
    scratch.file("pkg/profile.afdo", "");

    setupAutoFdoThinLtoCrosstool();
    useConfiguration("--fdo_optimize=/pkg/profile.afdo", "--compilation_mode=opt");

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");
    // We should not have a ThinLTO backend action
    assertThat(backendAction).isNull();
  }

  /** Tests that --features=autofdo_implicit_thinlto enables ThinLTO for AFDO with LLVM. */
  @Test
  public void testAutoFdoImplicitThinLto() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          malloc = '//base:system_malloc')");

    scratch.file("pkg/binfile.cc", "int main() {}");
    scratch.file("pkg/profile.afdo", "");

    setupAutoFdoThinLtoCrosstool();
    useConfiguration(
        "--fdo_optimize=/pkg/profile.afdo",
        "--compilation_mode=opt",
        "--features=autofdo_implicit_thinlto",
        "--features=use_lto_native_object_directory");

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");
    // For ThinLTO compilation we should have a non-null backend action
    assertThat(backendAction).isNotNull();
  }

  /**
   * Tests that --features=-thin_lto overrides --features=autofdo_implicit_thinlto and prevents
   * enabling ThinLTO for AFDO with LLVM.
   */
  @Test
  public void testAutoFdoImplicitThinLtoDisabledOption() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          malloc = '//base:system_malloc')");

    scratch.file("pkg/binfile.cc", "int main() {}");
    scratch.file("pkg/profile.afdo", "");

    setupAutoFdoThinLtoCrosstool();
    useConfiguration(
        "--fdo_optimize=/pkg/profile.afdo",
        "--compilation_mode=opt",
        "--features=autofdo_implicit_thinlto",
        "--features=-thin_lto",
        "--features=use_lto_native_object_directory");

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");
    // We should not have a ThinLTO backend action
    assertThat(backendAction).isNull();
  }

  /**
   * Tests that features=[-thin_lto] in the build rule overrides --features=autofdo_implicit_thinlto
   * and prevents enabling ThinLTO for AFDO with LLVM.
   */
  @Test
  public void testAutoFdoImplicitThinLtoDisabledRule() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          features = ['-thin_lto', 'use_lto_native_object_directory'],",
        "          malloc = '//base:system_malloc')");

    scratch.file("pkg/binfile.cc", "int main() {}");
    scratch.file("pkg/profile.afdo", "");

    setupAutoFdoThinLtoCrosstool();
    useConfiguration(
        "--fdo_optimize=/pkg/profile.afdo",
        "--compilation_mode=opt",
        "--features=autofdo_implicit_thinlto");

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");
    // We should not have a ThinLTO backend action
    assertThat(backendAction).isNull();
  }

  /**
   * Tests that features=[-thin_lto] in the package overrides --features=autofdo_implicit_thinlto
   * and prevents enabling ThinLTO for AFDO with LLVM.
   */
  @Test
  public void testAutoFdoImplicitThinLtoDisabledPackage() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "package(features = ['-thin_lto', 'use_lto_native_object_directory'])",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          malloc = '//base:system_malloc')");

    scratch.file("pkg/binfile.cc", "int main() {}");
    scratch.file("pkg/profile.afdo", "");

    setupAutoFdoThinLtoCrosstool();
    useConfiguration(
        "--fdo_optimize=/pkg/profile.afdo",
        "--compilation_mode=opt",
        "--features=autofdo_implicit_thinlto");

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");
    // We should not have a ThinLTO backend action
    assertThat(backendAction).isNull();
  }

  private void setupFdoThinLtoCrosstool() throws Exception {
    setupThinLTOCrosstool(
        CppRuleClasses.FDO_OPTIMIZE,
        CppRuleClasses.ENABLE_FDO_THINLTO,
        MockCcSupport.FDO_IMPLICIT_THINLTO);
  }

  /**
   * Tests that ThinLTO is not enabled for FDO with LLVM without --features=fdo_implicit_thinlto.
   */
  @Test
  public void testFdoNoImplicitThinLto() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          malloc = '//base:system_malloc')");

    scratch.file("pkg/binfile.cc", "int main() {}");
    scratch.file("pkg/profile.zip", "");

    setupFdoThinLtoCrosstool();
    useConfiguration("--fdo_optimize=/pkg/profile.zip", "--compilation_mode=opt");

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");
    // We should not have a ThinLTO backend action
    assertThat(backendAction).isNull();
  }

  /** Tests that --features=fdo_implicit_thinlto enables ThinLTO for FDO with LLVM. */
  @Test
  public void testFdoImplicitThinLto() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          malloc = '//base:system_malloc')");

    scratch.file("pkg/binfile.cc", "int main() {}");
    scratch.file("pkg/profile.zip", "");

    setupFdoThinLtoCrosstool();
    useConfiguration(
        "--fdo_optimize=/pkg/profile.zip",
        "--compilation_mode=opt",
        "--features=fdo_implicit_thinlto",
        "--features=use_lto_native_object_directory");

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");
    // For ThinLTO compilation we should have a non-null backend action
    assertThat(backendAction).isNotNull();
  }

  /**
   * Tests that --features=-thin_lto overrides --features=fdo_implicit_thinlto and prevents enabling
   * ThinLTO for FDO with LLVM.
   */
  @Test
  public void testFdoImplicitThinLtoDisabledOption() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          malloc = '//base:system_malloc')");

    scratch.file("pkg/binfile.cc", "int main() {}");
    scratch.file("pkg/profile.zip", "");

    setupFdoThinLtoCrosstool();
    useConfiguration(
        "--fdo_optimize=/pkg/profile.zip",
        "--compilation_mode=opt",
        "--features=fdo_implicit_thinlto",
        "--features=-thin_lto",
        "--features=use_lto_native_object_directory");

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");
    // We should not have a ThinLTO backend action
    assertThat(backendAction).isNull();
  }

  /**
   * Tests that features=[-thin_lto] in the build rule overrides --features=fdo_implicit_thinlto and
   * prevents enabling ThinLTO for FDO with LLVM.
   */
  @Test
  public void testFdoImplicitThinLtoDisabledRule() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          features = ['-thin_lto', 'use_lto_native_object_directory'],",
        "          malloc = '//base:system_malloc')");

    scratch.file("pkg/binfile.cc", "int main() {}");
    scratch.file("pkg/profile.zip", "");

    setupFdoThinLtoCrosstool();
    useConfiguration(
        "--fdo_optimize=/pkg/profile.zip",
        "--compilation_mode=opt",
        "--features=fdo_implicit_thinlto");

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");
    // We should not have a ThinLTO backend action
    assertThat(backendAction).isNull();
  }

  /**
   * Tests that features=[-thin_lto] in the package overrides --features=fdo_implicit_thinlto and
   * prevents enabling ThinLTO for FDO with LLVM.
   */
  @Test
  public void testFdoImplicitThinLtoDisabledPackage() throws Exception {
    setupThinLTOCrosstool();
    scratch.file(
        "pkg/BUILD",
        "package(features = ['-thin_lto', 'use_lto_native_object_directory'])",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          malloc = '//base:system_malloc')");

    scratch.file("pkg/binfile.cc", "int main() {}");
    scratch.file("pkg/profile.zip", "");

    setupFdoThinLtoCrosstool();
    useConfiguration(
        "--fdo_optimize=/pkg/profile.zip",
        "--compilation_mode=opt",
        "--features=fdo_implicit_thinlto");

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");
    // We should not have a ThinLTO backend action
    assertThat(backendAction).isNull();
  }

  private void setupXBinaryFdoThinLtoCrosstool() throws Exception {
    setupThinLTOCrosstool(
        CppRuleClasses.XBINARYFDO,
        CppRuleClasses.ENABLE_XFDO_THINLTO,
        MockCcSupport.XFDO_IMPLICIT_THINLTO);
  }

  /**
   * Tests that ThinLTO is not enabled for XFDO with LLVM without
   * --features=xbinaryfdo_implicit_thinlto.
   */
  @Test
  public void testXBinaryFdoNoImplicitThinLto() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ])",
        "fdo_profile(name='out.xfdo', profile='profiles.xfdo')");

    scratch.file("pkg/binfile.cc", "int main() {}");

    setupXBinaryFdoThinLtoCrosstool();
    useConfiguration("--xbinary_fdo=//pkg:out.xfdo", "--compilation_mode=opt");

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");
    // We should not have a ThinLTO backend action
    assertThat(backendAction).isNull();
  }

  /** Tests that --features=xbinaryfdo_implicit_thinlto enables ThinLTO for XFDO with LLVM. */
  @Test
  public void testXBinaryFdoImplicitThinLto() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ])",
        "fdo_profile(name='out.xfdo', profile='profiles.xfdo')");

    scratch.file("pkg/binfile.cc", "int main() {}");

    setupXBinaryFdoThinLtoCrosstool();
    useConfiguration(
        "--xbinary_fdo=//pkg:out.xfdo",
        "--compilation_mode=opt",
        "--features=xbinaryfdo_implicit_thinlto",
        "--features=use_lto_native_object_directory");

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");
    // For ThinLTO compilation we should have a non-null backend action
    assertThat(backendAction).isNotNull();
  }

  /**
   * Tests that --features=-thin_lto overrides --features=xbinaryfdo_implicit_thinlto and prevents
   * enabling ThinLTO for XFDO with LLVM.
   */
  @Test
  public void testXBinaryFdoImplicitThinLtoDisabledOption() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ])",
        "fdo_profile(name='out.xfdo', profile='profiles.xfdo')");

    scratch.file("pkg/binfile.cc", "int main() {}");

    setupXBinaryFdoThinLtoCrosstool();
    useConfiguration(
        "--xbinary_fdo=//pkg:out.xfdo",
        "--compilation_mode=opt",
        "--features=xbinaryfdo_implicit_thinlto",
        "--features=-thin_lto",
        "--features=use_lto_native_object_directory");

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");
    // We should not have a ThinLTO backend action
    assertThat(backendAction).isNull();
  }

  /**
   * Tests that features=[-thin_lto] in the build rule overrides
   * --features=xbinaryfdo_implicit_thinlto and prevents enabling ThinLTO for XFDO with LLVM.
   */
  @Test
  public void testXBinaryFdoImplicitThinLtoDisabledRule() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          features = ['-thin_lto', 'use_lto_native_object_directory'])",
        "fdo_profile(name='out.xfdo', profile='profiles.xfdo')");

    scratch.file("pkg/binfile.cc", "int main() {}");

    setupXBinaryFdoThinLtoCrosstool();
    useConfiguration(
        "--xbinary_fdo=//pkg:out.xfdo",
        "--compilation_mode=opt",
        "--features=xbinaryfdo_implicit_thinlto");

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");
    // We should not have a ThinLTO backend action
    assertThat(backendAction).isNull();
  }

  /**
   * Tests that features=[-thin_lto] in the package overrides --features=fdo_implicit_thinlto and
   * prevents enabling ThinLTO for XFDO with LLVM.
   */
  @Test
  public void testXBinaryFdoImplicitThinLtoDisabledPackage() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "package(features = ['-thin_lto', 'use_lto_native_object_directory'])",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ])",
        "fdo_profile(name='out.xfdo', profile='profiles.xfdo')");

    scratch.file("pkg/binfile.cc", "int main() {}");

    setupXBinaryFdoThinLtoCrosstool();
    useConfiguration(
        "--xbinary_fdo=//pkg:out.xfdo",
        "--compilation_mode=opt",
        "--features=xbinaryfdo_implicit_thinlto");

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");
    // We should not have a ThinLTO backend action
    assertThat(backendAction).isNull();
  }

  @Test
  public void testXBinaryFdo() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "package(features = ['thin_lto', 'use_lto_native_object_directory'])",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          malloc = '//base:system_malloc')",
        "fdo_profile(name='out.xfdo', profile='profiles.xfdo')");

    scratch.file("pkg/binfile.cc", "int main() {}");

    setupThinLTOCrosstool(CppRuleClasses.XBINARYFDO);
    useConfiguration("--xbinary_fdo=//pkg:out.xfdo", "--compilation_mode=opt");

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");

    // Checks that -fauto-profile is added to the LtoBackendAction.
    assertThat(Joiner.on(" ").join(backendAction.getArguments()))
        .containsMatch("-fauto-profile=[^ ]*/profiles.xfdo");
    assertThat(ActionsTestUtil.baseArtifactNames(backendAction.getInputs()))
        .contains("profiles.xfdo");
  }

  /**
   * Tests that ThinLTO is not enabled for XBINARYFDO with --features=autofdo_implicit_thinlto and
   * --features=fdo_implicit_thinlto.
   */
  @Test
  public void testXBinaryFdoNoAutoFdoOrFdoImplicitThinLto() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          malloc = '//base:system_malloc')",
        "fdo_profile(name='out.xfdo', profile='profiles.xfdo')");

    scratch.file("pkg/binfile.cc", "int main() {}");

    setupThinLTOCrosstool(
        CppRuleClasses.ENABLE_FDO_THINLTO,
        MockCcSupport.FDO_IMPLICIT_THINLTO,
        CppRuleClasses.ENABLE_AFDO_THINLTO,
        MockCcSupport.AUTOFDO_IMPLICIT_THINLTO,
        CppRuleClasses.XBINARYFDO);
    useConfiguration(
        "--xbinary_fdo=//pkg:out.xfdo",
        "--compilation_mode=opt",
        "--features=autofdo_implicit_thinlto",
        "--features=fdo_implicit_thinlto",
        "--features=use_lto_native_object_directory");

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/pkg/binfile.o");
    // We should not have a ThinLTO backend action
    assertThat(backendAction).isNull();
  }

  @Test
  public void testPICBackendOrder() throws Exception {
    createBuildFiles();

    setupThinLTOCrosstool(CppRuleClasses.SUPPORTS_PIC);
    useConfiguration("--copt=-fno-PIE");
    String rootExecPath = getRootExecPath();
    LtoBackendAction backendAction =
        getBackendAction("pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");
    assertThat(backendAction.getArguments()).containsAtLeast("-fno-PIE", "-fPIC").inOrder();
  }

  @Test
  public void testPropellerOptimizeAbsoluteOptions() throws Exception {
    createBuildFiles();

    setupThinLTOCrosstool(CppRuleClasses.SUPPORTS_PIC, CppRuleClasses.AUTOFDO);

    useConfiguration(
        "--propeller_optimize_absolute_cc_profile=/tmp/cc_profile.txt",
        "--propeller_optimize_absolute_ld_profile=/tmp/ld_profile.txt",
        "--compilation_mode=opt");
    Artifact binArtifact = getFilesToBuild(getCurrentTarget()).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    SpawnAction linkAction = (SpawnAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);
    assertThat(ActionsTestUtil.baseArtifactNames(linkAction.getInputs()))
        .contains("ld_profile.txt");

    List<String> commandLine = linkAction.getArguments();
    assertThat(commandLine.toString())
        .containsMatch("-Wl,--symbol-ordering-file=.*/ld_profile.txt");

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");

    String expectedCompilerFlag = "-fbasic-block-sections=list=.*/cc_profile.txt";
    assertThat(Joiner.on(" ").join(backendAction.getArguments()))
        .containsMatch(expectedCompilerFlag);
    String expectedBuildTypeFlag = "-DBUILD_PROPELLER_TYPE=\"full\"";
    assertThat(Joiner.on(" ").join(backendAction.getArguments()))
        .containsMatch(expectedBuildTypeFlag);
    assertThat(ActionsTestUtil.baseArtifactNames(backendAction.getInputs()))
        .contains("cc_profile.txt");

    CppLinkAction indexAction = getIndexAction(backendAction);
    assertThat(ActionsTestUtil.baseArtifactNames(indexAction.getInputs()))
        .doesNotContain("ld_profile.txt");
  }

  @Test
  public void testPropellerCcCompile() throws Exception {
    createBuildFiles();

    setupThinLTOCrosstool(CppRuleClasses.SUPPORTS_PIC, CppRuleClasses.AUTOFDO);

    useConfiguration(
        "--propeller_optimize_absolute_cc_profile=/tmp/cc_profile.txt",
        "--propeller_optimize_absolute_ld_profile=/tmp/ld_profile.txt",
        "--compilation_mode=opt");
    Artifact binArtifact = getFilesToBuild(getCurrentTarget()).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");
    CppLinkAction indexAction = getIndexAction(backendAction);
    CppCompileAction bitcodeAction =
        (CppCompileAction)
            getPredecessorByInputName(indexAction, "pkg/_objs/bin/binfile.indexing.o");
    assertThat(ActionsTestUtil.baseArtifactNames(bitcodeAction.getInputs()))
        .doesNotContain("cc_profile.txt");
    assertThat(Joiner.on(" ").join(bitcodeAction.getArguments()))
        .doesNotContainMatch("-fbasic-block-sections=");
  }

  /**
   * Check that the temporary opt-out from disabling Propeller profiles for ThinLTO compile actions
   * works.
   *
   * <p>TODO(b/182804945): Remove after making sure that the rollout of the new Propeller profile
   * passing logic didn't break anything.
   */
  @Test
  public void testPropellerCcCompileWithPropellerOptimizeThinLtoCompileActions() throws Exception {
    createBuildFiles();

    setupThinLTOCrosstool(CppRuleClasses.SUPPORTS_PIC, CppRuleClasses.AUTOFDO);

    useConfiguration(
        "--propeller_optimize_absolute_cc_profile=/tmp/cc_profile.txt",
        "--propeller_optimize_absolute_ld_profile=/tmp/ld_profile.txt",
        "--compilation_mode=opt",
        "--features=propeller_optimize_thinlto_compile_actions",
        "--features=use_lto_native_object_directory");
    Artifact binArtifact = getFilesToBuild(getCurrentTarget()).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");
    CppLinkAction indexAction = getIndexAction(backendAction);
    assertThat(artifactsToStrings(indexAction.getInputs()))
        .containsAtLeast(
            "bin pkg/_objs/bin/binfile.indexing.o", "bin pkg/_objs/lib/libfile.indexing.o");

    CppCompileAction bitcodeAction =
        (CppCompileAction)
            getPredecessorByInputName(indexAction, "pkg/_objs/bin/binfile.indexing.o");
    assertThat(ActionsTestUtil.baseArtifactNames(bitcodeAction.getInputs()))
        .contains("cc_profile.txt");
    assertThat(Joiner.on(" ").join(bitcodeAction.getArguments()))
        .containsMatch("-fbasic-block-sections=list=.*/cc_profile.txt");
  }

  @Test
  public void testPropellerHostBuilds() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "package(features = ['thin_lto', 'use_lto_native_object_directory'])",
        "",
        "cc_binary(name = '" + targetName + "',",
        "          srcs = ['binfile.cc', ],",
        "          deps = [ ':lib' ], ",
        "          malloc = '//base:system_malloc')",
        "cc_library(name = 'lib',",
        "        srcs = ['libfile.cc'],",
        "        hdrs = ['libfile.h'])",
        "cc_binary(name = 'gen_lib',",
        "        srcs = ['gen_lib.cc'])",
        "genrule(name = 'lib_genrule',",
        "        srcs = [],",
        "        outs = ['libfile.cc'],",
        "        cmd = '$(location gen_lib) > \"$@\"',",
        "        tools = [':gen_lib'])");

    scratch.file("pkg/binfile.cc", "#include \"pkg/libfile.h\"", "int main() { return pkg(); }");
    scratch.file(
        "pkg/gen_lib.cc",
        "#include <cstdio>",
        "int main() { puts(\"int pkg() { return 42; }\"); }");
    scratch.file("pkg/libfile.h", "int pkg();");

    setupThinLTOCrosstool(CppRuleClasses.SUPPORTS_PIC, CppRuleClasses.AUTOFDO);

    useConfiguration(
        "--propeller_optimize_absolute_cc_profile=/tmp/cc_profile.txt",
        "--propeller_optimize_absolute_ld_profile=/tmp/ld_profile.txt",
        "--compilation_mode=opt");
    Artifact binArtifact = getFilesToBuild(getCurrentTarget()).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");
    CppLinkAction indexAction = getIndexAction(backendAction);
    assertThat(artifactsToStrings(indexAction.getInputs()))
        .contains("bin pkg/_objs/lib/libfile.indexing.o");

    CppCompileAction bitcodeAction =
        (CppCompileAction)
            getPredecessorByInputName(indexAction, "pkg/_objs/lib/libfile.indexing.o");

    Action genruleAction = getPredecessorByInputName(bitcodeAction, "pkg/libfile.cc");

    SpawnAction hostLinkAction =
        (SpawnAction) getPredecessorByInputName(genruleAction, "pkg/gen_lib");
    assertThat(ActionsTestUtil.baseArtifactNames(hostLinkAction.getInputs()))
        .doesNotContain("ld_profile.txt");
    assertThat(hostLinkAction.getArguments().toString())
        .doesNotContainMatch("-Wl,--symbol-ordering-file=.*/ld_profile.txt");

    // The hostLinkAction inputs has a different root from the backendAction.
    // Here we confirm that the correct root is on the path
    String hostrootExecPath = hostLinkAction.getPrimaryOutput().getRoot().getExecPathString();
    LtoBackendAction hostBackendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                hostLinkAction,
                "pkg/gen_lib.lto-obj/" + hostrootExecPath + "/pkg/_objs/gen_lib/gen_lib.o");
    assertThat(ActionsTestUtil.baseArtifactNames(hostBackendAction.getInputs()))
        .doesNotContain("cc_profile.txt");
    assertThat(Joiner.on(" ").join(hostBackendAction.getArguments()))
        .doesNotContainMatch("-fbasic-block-sections");

    SpawnAction hostIndexAction = getIndexAction(hostBackendAction);
    assertThat(hostIndexAction).isNotNull();
    assertThat(ActionsTestUtil.baseArtifactNames(hostIndexAction.getInputs()))
        .doesNotContain("ld_profile.txt");
    assertThat(hostIndexAction.getArguments().toString())
        .doesNotContainMatch("-Wl,--symbol-ordering-file=.*/ld_profile.txt");

    CppCompileAction hostBitcodeAction =
        (CppCompileAction)
            getPredecessorByInputName(hostIndexAction, "pkg/_objs/gen_lib/gen_lib.indexing.o");
    assertThat(ActionsTestUtil.baseArtifactNames(hostBitcodeAction.getInputs()))
        .doesNotContain("cc_profile.txt");
    assertThat(Joiner.on(" ").join(hostBitcodeAction.getArguments()))
        .doesNotContainMatch("-fbasic-block-sections=");
  }

  private void testPropellerOptimizeOption(boolean label) throws Exception {
    createBuildFiles();

    if (label) {
      scratch.file(
          "fdo/BUILD",
          "propeller_optimize(name='test_propeller_optimize', cc_profile=':cc_profile.txt',"
              + " ld_profile=':ld_profile.txt')");
    } else {
      scratch.file(
          "fdo/BUILD",
          "propeller_optimize(name='test_propeller_optimize',"
              + "absolute_cc_profile='/tmp/cc_profile.txt',"
              + "absolute_ld_profile='/tmp/ld_profile.txt')");
    }
    setupThinLTOCrosstool(CppRuleClasses.SUPPORTS_PIC, CppRuleClasses.AUTOFDO);

    useConfiguration(
        "--propeller_optimize=//fdo:test_propeller_optimize", "--compilation_mode=opt");

    Artifact binArtifact = getFilesToBuild(getCurrentTarget()).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    SpawnAction linkAction = (SpawnAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    List<String> commandLine = linkAction.getArguments();
    assertThat(commandLine.toString())
        .containsMatch("-Wl,--symbol-ordering-file=.*/ld_profile.txt");

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");

    String expectedCompilerFlag = "-fbasic-block-sections=list=.*/cc_profile.txt";
    assertThat(Joiner.on(" ").join(backendAction.getArguments()))
        .containsMatch(expectedCompilerFlag);
    String expectedBuildTypeFlag = "-DBUILD_PROPELLER_TYPE=\"full\"";
    assertThat(Joiner.on(" ").join(backendAction.getArguments()))
        .containsMatch(expectedBuildTypeFlag);
    assertThat(ActionsTestUtil.baseArtifactNames(backendAction.getInputs()))
        .contains("cc_profile.txt");
  }

  @Test
  public void testPropellerOptimizeOptionFromAbsolutePath() throws Exception {
    testPropellerOptimizeOption(false);
  }

  @Test
  public void testPropellerOptimizeOptionFromLabel() throws Exception {
    testPropellerOptimizeOption(true);
  }

  private void testLLVMCachePrefetchBackendOption(String extraOption, boolean asLabel)
      throws Exception {
    createBuildFiles();
    if (asLabel) {
      scratch.file(
          "fdo/BUILD", "fdo_prefetch_hints(name='test_profile', profile=':prefetch.afdo')");
    } else {
      scratch.file(
          "fdo/BUILD",
          "fdo_prefetch_hints(name='test_profile', absolute_path_profile='/tmp/prefetch.afdo')");
    }

    setupThinLTOCrosstool(CppRuleClasses.SUPPORTS_PIC, CppRuleClasses.AUTOFDO);
    useConfiguration(
        "--fdo_prefetch_hints=//fdo:test_profile", "--compilation_mode=opt", extraOption);

    String rootExecPath = getRootExecPath();
    LtoBackendAction backendAction =
        getBackendAction("pkg/bin.lto-obj/" + rootExecPath + "/pkg/_objs/bin/binfile.o");

    String expectedCompilerFlag =
        "-prefetch-hints-file="
            + (asLabel ? ".*/prefetch.afdo" : "(blaze|bazel)-out/.*/fdo/.*/prefetch.afdo");
    assertThat(Joiner.on(" ").join(backendAction.getArguments()))
        .containsMatch("-mllvm " + expectedCompilerFlag);

    assertThat(ActionsTestUtil.baseArtifactNames(backendAction.getInputs()))
        .contains("prefetch.afdo");
  }

  @Test
  public void testFdoCachePrefetchLLVMOptionsToBackendFromPath() throws Exception {
    testLLVMCachePrefetchBackendOption("", false);
  }

  @Test
  public void testFdoCachePrefetchAndFdoLLVMOptionsToBackendFromPath() throws Exception {
    testLLVMCachePrefetchBackendOption("--fdo_optimize=/profile.zip", false);
  }

  @Test
  public void testFdoCachePrefetchLLVMOptionsToBackendFromLabel() throws Exception {
    testLLVMCachePrefetchBackendOption("", true);
  }

  @Test
  public void testFdoCachePrefetchAndFdoLLVMOptionsToBackendFromLabel() throws Exception {
    testLLVMCachePrefetchBackendOption("--fdo_optimize=/profile.zip", true);
  }
}
