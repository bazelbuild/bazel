// Copyright 2015 Google Inc. All rights reserved.
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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * LTOBackendArtifacts represents a set of artifacts for a single LTO backend compile.
 *
 * <p>LTO expands the traditional 2 step compile (N x compile .cc, 1x link (N .o files) into a
 * 4 step process:
 * <ul>
 *   <li>1. Bitcode generation (N times). This is produces intermediate LLVM bitcode from a source
 *   file. For this product, it reuses the .o extension.
 *   </li>
 *   <li>2. Indexing (once on N files). This takes all bitcode .o files, and for each .o file, it
 *   decides from which other .o files symbols can be inlined. In addition, it generates an
 *   index for looking up these symbols.
 *   </li>
 *   <li>3. Backend compile (N times). This is the traditional compilation, and uses the same
 *   command line
 *   as the Bitcode generation in 1). Since the compiler has many bit code files available, it
 *   can inline functions and propagate constants across .o files. This step is costly, as it
 *   will do traditional optimization. The result is a .lto.o file, a traditional ELF object file.
 *   <p>
 *     For simplicity, our current prototype step 2. also generates a command line which we execute
 *     in step 3.
 *   </p>
 *   </li>
 *   <li>4. Backend link (once). This is the traditional link, and produces the final executable.
 *   </li>
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

  // A file containing a command-line to run for the backend compile.
  private final Artifact beCommandline;

  // The result of executing the above command line, an ELF object file.
  private final Artifact objectFile;

  // A collection of all of the bitcode files. This is the universe from which the .imports file
  // distills its lists.  The nested set is the same across all LTOBackendArtifacts of a given
  // binary.
  private final NestedSet<Artifact> bitcodeFiles;

  LTOBackendArtifacts(
      PathFragment ltoOutputRootPrefix,
      Artifact bitcodeFile,
      NestedSet<Artifact> allBitCodeFiles,
      RuleContext ruleContext,
      CppLinkAction.LinkArtifactFactory linkArtifactFactory) {
    this.bitcodeFile = bitcodeFile;
    PathFragment obj = ltoOutputRootPrefix.getRelative(bitcodeFile.getRootRelativePath());

    objectFile = linkArtifactFactory.create(ruleContext, obj);
    imports = linkArtifactFactory.create(
        ruleContext, FileSystemUtils.replaceExtension(obj, ".imports"));
    index = linkArtifactFactory.create(
        ruleContext, FileSystemUtils.replaceExtension(obj, ".thinlto.index"));
    beCommandline = linkArtifactFactory.create(
        ruleContext, FileSystemUtils.replaceExtension(obj, ".thinlto_commandline.txt"));

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
    builder.add(beCommandline);
  }

  public void scheduleLTOBackendAction(RuleContext ruleContext) {
    SpawnAction.Builder builder = new SpawnAction.Builder();

    // TODO(bazel-team): should prune to the files mentioned in .imports.
    builder.addTransitiveInputs(bitcodeFiles);
    builder.addInput(imports);
    builder.addInput(index);
    builder.addInput(beCommandline);
    builder.addTransitiveInputs(CppHelper.getToolchain(ruleContext).getCompile());
    builder.addOutput(objectFile);

    builder.setProgressMessage("LTO Backend Compile");
    builder.setMnemonic("CcLtoBackendCompile");

    // The command-line doesn't specify the full path to clang++, so we set it in the
    // environment.
    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);

    PathFragment compiler = cppConfiguration.getCppExecutable();

    builder.setShellCommand(beCommandline.getExecPathString());
    builder.setEnvironment(
        ImmutableMap.of("CLANGXX", compiler.replaceName("clang++").getPathString()));

    ruleContext.registerAction(builder.build(ruleContext));
  }
}
