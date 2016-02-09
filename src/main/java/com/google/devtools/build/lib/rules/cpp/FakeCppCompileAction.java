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

package com.google.devtools.build.lib.rules.cpp;

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.util.UUID;
import java.util.logging.Logger;

import javax.annotation.Nullable;

/**
 * Action that represents a fake C++ compilation step.
 */
@ThreadCompatible
public class FakeCppCompileAction extends CppCompileAction {

  private static final Logger LOG = Logger.getLogger(FakeCppCompileAction.class.getName());

  public static final UUID GUID = UUID.fromString("b2d95c91-1434-47ae-a786-816017de8494");

  private final PathFragment tempOutputFile;

  FakeCppCompileAction(
      ActionOwner owner,
      ImmutableList<String> features,
      FeatureConfiguration featureConfiguration,
      CcToolchainFeatures.Variables variables,
      Artifact sourceFile,
      boolean shouldScanIncludes,
      Label sourceLabel,
      NestedSet<Artifact> mandatoryInputs,
      Artifact outputFile,
      PathFragment tempOutputFile,
      DotdFile dotdFile,
      BuildConfiguration configuration,
      CppConfiguration cppConfiguration,
      CppCompilationContext context,
      Class<? extends CppCompileActionContext> actionContext,
      ImmutableList<String> copts,
      ImmutableList<String> pluginOpts,
      Predicate<String> nocopts,
      ImmutableList<PathFragment> extraSystemIncludePrefixes,
      @Nullable String fdoBuildStamp,
      RuleContext ruleContext,
      boolean usePic) {
    super(owner,
        features,
        featureConfiguration,
        variables,
        sourceFile,
        shouldScanIncludes,
        sourceLabel,
        mandatoryInputs,
        outputFile,
        dotdFile,
        null,
        null,
        null,
        configuration,
        cppConfiguration,
        // We only allow inclusion of header files explicitly declared in
        // "srcs", so we only use declaredIncludeSrcs, not declaredIncludeDirs.
        // (Disallowing use of undeclared headers for cc_fake_binary is needed
        // because the header files get included in the runfiles for the
        // cc_fake_binary and for the negative compilation tests that depend on
        // the cc_fake_binary, and the runfiles must be determined at analysis
        // time, so they can't depend on the contents of the ".d" file.)
        CppCompilationContext.disallowUndeclaredHeaders(context), actionContext, copts, pluginOpts,
        nocopts, extraSystemIncludePrefixes, fdoBuildStamp, VOID_INCLUDE_RESOLVER,
        ImmutableList.<IncludeScannable>of(), GUID, usePic, ruleContext);
    this.tempOutputFile = Preconditions.checkNotNull(tempOutputFile);
  }

  @Override
  @ThreadCompatible
  public void execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    Executor executor = actionExecutionContext.getExecutor();

    // First, do a normal compilation, to generate the ".d" file. The generated object file is built
    // to a temporary location (tempOutputFile) and ignored afterwards.
    LOG.info("Generating " + getDotdFile());
    CppCompileActionContext context = executor.getContext(actionContext);
    CppCompileActionContext.Reply reply = null;
    try {
      // We delegate stdout/stderr to nowhere, i.e. same as redirecting to /dev/null.
      reply = context.execWithReply(
          this, actionExecutionContext.withFileOutErr(new FileOutErr()));
    } catch (ExecException e) {
      // We ignore failures here (other than capturing the Distributor reply).
      // The compilation may well fail (that's the whole point of negative compilation tests).
      // We execute it here just for the side effect of generating the ".d" file.
      reply = context.getReplyFromException(e, this);
      if (reply == null) {
        // This can only happen if the ExecException does not come from remote execution.
        throw e.toActionExecutionException("Fake C++ Compilation of rule '"
            + getOwner().getLabel() + "'", executor.getVerboseFailures(), this);
      }
    }
    IncludeScanningContext scanningContext = executor.getContext(IncludeScanningContext.class);
    NestedSet<Artifact> discoveredInputs =
        discoverInputsFromDotdFiles(
            executor.getExecRoot(), scanningContext.getArtifactResolver(), reply);
    reply = null; // Clear in-memory .d files early.

    // Even cc_fake_binary rules need to properly declare their dependencies...
    // In fact, they need to declare their dependencies even more than cc_binary rules do.
    // CcCommonConfiguredTarget passes in an empty set of declaredIncludeDirs,
    // so this check below will only allow inclusion of header files that are explicitly
    // listed in the "srcs" of the cc_fake_binary or in the "srcs" of a cc_library that it
    // depends on.
    try {
      validateInclusions(
          discoveredInputs,
          actionExecutionContext.getArtifactExpander(),
          executor.getEventHandler());
    } catch (ActionExecutionException e) {
      // TODO(bazel-team): (2009) make this into an error, once most of the current warnings
      // are fixed.
      executor.getEventHandler().handle(Event.warn(
          getOwner().getLocation(),
          e.getMessage() + ";\n  this warning may eventually become an error"));
    }

    updateActionInputs(discoveredInputs);

    // Generate a fake ".o" file containing the command line needed to generate
    // the real object file.
    LOG.info("Generating " + outputFile);

    // A cc_fake_binary rule generates fake .o files and a fake target file,
    // which merely contain instructions on building the real target. We need to
    // be careful to use a new set of output file names in the instructions, as
    // to not overwrite the fake output files when someone tries to follow the
    // instructions. As the real compilation is executed by the test from its
    // runfiles directory (where writing is forbidden), we patch the command
    // line to write to $TEST_TMPDIR instead.
    final String outputPrefix = "$TEST_TMPDIR/";
    String argv = Joiner.on(' ').join(
      Iterables.transform(getArgv(outputFile.getExecPath()), new Function<String, String>() {
        @Override
        public String apply(String input) {
          String result = ShellEscaper.escapeString(input);
          if (input.equals(outputFile.getExecPathString())
              || input.equals(getDotdFile().getSafeExecPath().getPathString())) {
            result = outputPrefix + result;
          }
          return result;
        }
      }));

    // Write the command needed to build the real .o file to the fake .o file.
    // Generate a command to ensure that the output directory exists; otherwise
    // the compilation would fail.
    try {
      // Ensure that the .d file and .o file are siblings, so that the "mkdir" below works for
      // both.
      Preconditions.checkState(outputFile.getExecPath().getParentDirectory().equals(
          getDotdFile().getSafeExecPath().getParentDirectory()));
      FileSystemUtils.writeContent(outputFile.getPath(), ISO_8859_1,
          outputFile.getPath().getBaseName() + ": "
          + "mkdir -p " + outputPrefix + "$(dirname " + outputFile.getExecPath() + ")"
          + " && " + argv + "\n");
    } catch (IOException e) {
      throw new ActionExecutionException("failed to create fake compile command for rule '" +
                                         getOwner().getLabel() + ": " + e.getMessage(),
                                         this, false);
    }
  }

  @Override
  protected PathFragment getInternalOutputFile() {
    return tempOutputFile;
  }

  @Override
  public String getMnemonic() { return "FakeCppCompile"; }

  @Override
  public ResourceSet estimateResourceConsumptionLocal() {
    return ResourceSet.createWithRamCpuIo(/*memoryMb=*/1, /*cpuUsage=*/0.1, /*ioUsage=*/0.0);
  }

  @Override
  public ResourceSet estimateResourceConsumption(Executor executor) {
    return executor.getContext(actionContext).estimateResourceConsumption(this);
  }
}
