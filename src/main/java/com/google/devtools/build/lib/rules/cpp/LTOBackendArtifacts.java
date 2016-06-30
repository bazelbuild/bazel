// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.LTOBackendAction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * LTOBackendArtifacts represents a set of artifacts for a single ThinLTO backend compile.
 *
 * <p>ThinLTO expands the traditional 2 step compile (N x compile .cc, 1x link (N .o files) into a 4
 * step process:
 *
 * <ul>
 * <li>1. Bitcode generation (N times). This is produces intermediate LLVM bitcode from a source
 *     file. For this product, it reuses the .o extension.
 * <li>2. Indexing (once on N files). This takes all bitcode .o files, and for each .o file, it
 *     decides from which other .o files symbols can be inlined. In addition, it generates an index
 *     for looking up these symbols, and an imports file for identifying new input files for each
 *     step 3 {@link LTOBackendAction}.
 * <li>3. Backend compile (N times). This is the traditional compilation, and uses the same command
 *     line as the Bitcode generation in 1). Since the compiler has many bit code files available,
 *     it can inline functions and propagate constants across .o files. This step is costly, as it
 *     will do traditional optimization. The result is a .lto.o file, a traditional ELF object file.
 * <li>4. Backend link (once). This is the traditional link, and produces the final executable.
 * </ul>
 */
public final class LTOBackendArtifacts {
  // A file containing mapping of symbol => bitcode file containing the symbol.
  private final Artifact index;

  // The bitcode file which is the input of the compile.
  private final Artifact bitcodeFile;

  // A file containing a list of bitcode files necessary to run the backend step. Currently
  // unused.
  private final Artifact imports;

  // The result of executing the above command line, an ELF object file.
  private final Artifact objectFile;

  // A map of all of the bitcode files. This is the universe from which the .imports file
  // distills its lists.  The map is the same across all LTOBackendArtifacts of a given
  // binary.
  private final Map<PathFragment, Artifact> bitcodeFiles;

  // Command line arguments to apply to back-end compile action, typically from
  // the feature configuration and user-provided linkopts.
  private List<String> commandLine;

  LTOBackendArtifacts(
      PathFragment ltoOutputRootPrefix,
      Artifact bitcodeFile,
      Map<PathFragment, Artifact> allBitCodeFiles,
      RuleContext ruleContext,
      CppLinkAction.LinkArtifactFactory linkArtifactFactory) {
    this.bitcodeFile = bitcodeFile;
    PathFragment obj = ltoOutputRootPrefix.getRelative(bitcodeFile.getRootRelativePath());

    objectFile = linkArtifactFactory.create(ruleContext, obj);
    imports =
        linkArtifactFactory.create(ruleContext, FileSystemUtils.appendExtension(obj, ".imports"));
    index =
        linkArtifactFactory.create(
            ruleContext, FileSystemUtils.appendExtension(obj, ".thinlto.bc"));

    bitcodeFiles = allBitCodeFiles;
  }

  public Artifact getObjectFile() {
    return objectFile;
  }

  public Artifact getBitcodeFile() {
    return bitcodeFile;
  }

  public void addIndexingOutputs(ImmutableList.Builder<Artifact> builder) {
    builder.add(imports);
    builder.add(index);
  }

  public void setCommandLine(List<String> cmdLine) {
    commandLine = cmdLine;
  }

  public void scheduleLTOBackendAction(RuleContext ruleContext, boolean usePic) {
    LTOBackendAction.Builder builder = new LTOBackendAction.Builder();
    builder.addImportsInfo(bitcodeFiles, imports);

    builder.addInput(bitcodeFile);
    builder.addInput(index);
    builder.addTransitiveInputs(CppHelper.getToolchain(ruleContext).getCompile());

    builder.addOutput(objectFile);

    builder.setProgressMessage("LTO Backend Compile " + objectFile.getFilename());
    builder.setMnemonic("CcLtoBackendCompile");

    // The command-line doesn't specify the full path to clang++, so we set it in the
    // environment.
    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);

    PathFragment compiler = cppConfiguration.getCppExecutable();

    builder.setExecutable(compiler);
    List<String> execArgs = new ArrayList<>();
    execArgs.add("-c");
    execArgs.add("-fthinlto-index=" + index.getExecPath());
    execArgs.add("-o");
    execArgs.add(objectFile.getExecPath().getPathString());
    execArgs.add("-x");
    execArgs.add("ir");
    execArgs.add(bitcodeFile.getExecPath().getPathString());
    if (usePic) {
      execArgs.add("-fPIC");
    }
    execArgs.addAll(commandLine);
    builder.addExecutableArguments(execArgs);

    ruleContext.registerAction(builder.build(ruleContext));
  }
}
