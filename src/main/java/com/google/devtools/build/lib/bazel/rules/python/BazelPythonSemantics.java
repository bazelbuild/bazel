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

package com.google.devtools.build.lib.bazel.rules.python;

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles.Builder;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Template;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore;
import com.google.devtools.build.lib.rules.python.PyCommon;
import com.google.devtools.build.lib.rules.python.PythonSemantics;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Functionality specific to the Python rules in Bazel.
 */
public class BazelPythonSemantics implements PythonSemantics {
  private static final Template STUB_TEMPLATE =
      Template.forResource(BazelPythonSemantics.class, "stub_template.txt");
  public static final InstrumentationSpec PYTHON_COLLECTION_SPEC = new InstrumentationSpec(
      FileTypeSet.of(BazelPyRuleClasses.PYTHON_SOURCE),
      "srcs", "deps", "data");

  @Override
  public void validate(RuleContext ruleContext, PyCommon common) {
  }

  @Override
  public void collectRunfilesForBinary(RuleContext ruleContext, Builder builder, PyCommon common) {
  }

  @Override
  public void collectDefaultRunfilesForBinary(RuleContext ruleContext, Builder builder) {
  }

  @Override
  public InstrumentationSpec getCoverageInstrumentationSpec() {
    return PYTHON_COLLECTION_SPEC;
  }

  @Override
  public Collection<Artifact> precompiledPythonFiles(
      RuleContext ruleContext, Collection<Artifact> sources, PyCommon common) {
    return ImmutableList.copyOf(sources);
  }

  @Override
  public List<PathFragment> getImports(RuleContext ruleContext) {
    List<PathFragment> result = new ArrayList<>();
    PathFragment packageFragment = ruleContext.getLabel().getPackageIdentifier()
        .getPathUnderExecRoot();
    // Python scripts start with x.runfiles/ as the module space, so everything must be manually
    // adjusted to be relative to the workspace name.
    packageFragment = new PathFragment(ruleContext.getWorkspaceName())
        .getRelative(packageFragment);
    for (String importsAttr : ruleContext.attributes().get("imports", Type.STRING_LIST)) {
      importsAttr = ruleContext.expandMakeVariables("includes", importsAttr);
      if (importsAttr.startsWith("/")) {
        ruleContext.attributeWarning("imports",
            "ignoring invalid absolute path '" + importsAttr + "'");
        continue;
      }
      PathFragment importsPath = packageFragment.getRelative(importsAttr).normalize();
      if (!importsPath.isNormalized()) {
        ruleContext.attributeError("imports",
            "Path " + importsAttr + " references a path above the execution root");
      }
      result.add(importsPath);
    }
    return result;
  }

  @Override
  public void createExecutable(
      RuleContext ruleContext,
      PyCommon common,
      CcLinkParamsStore ccLinkParamsStore,
      NestedSet<PathFragment> imports)
      throws InterruptedException {
    String main = common.determineMainExecutableSource(/*withWorkspaceName=*/ true);
    Artifact executable = common.getExecutable();
    BazelPythonConfiguration config = ruleContext.getFragment(BazelPythonConfiguration.class);
    String pythonBinary;

    switch (common.getVersion()) {
      case PY2: pythonBinary = config.getPython2Path(); break;
      case PY3: pythonBinary = config.getPython3Path(); break;
      default: throw new IllegalStateException();
    }

    if (!ruleContext.getConfiguration().buildPythonZip()) {
      ruleContext.registerAction(
          new TemplateExpansionAction(
              ruleContext.getActionOwner(),
              executable,
              STUB_TEMPLATE,
              ImmutableList.of(
                  Substitution.of("%main%", main),
                  Substitution.of("%python_binary%", pythonBinary),
                  Substitution.of("%imports%", Joiner.on(":").join(imports)),
                  Substitution.of("%workspace_name%", ruleContext.getWorkspaceName())),
              true));
    } else {
      Artifact zipFile = common.getPythonZipArtifact();
      PathFragment workspaceName = getWorkspaceNameForPythonZip(ruleContext.getWorkspaceName());
      main = workspaceName.getRelative(common.determineMainExecutableSource(false)).toString();
      PathFragment defaultWorkspacename = new PathFragment(Label.DEFAULT_REPOSITORY_DIRECTORY);
      StringBuilder importPaths = new StringBuilder();
      importPaths.append(File.pathSeparator).append("$PYTHON_RUNFILES/").append(workspaceName);
      for (PathFragment path : imports) {
        if (path.startsWith(defaultWorkspacename)) {
          path = new PathFragment(workspaceName, path.subFragment(1, path.segmentCount()));
        }
        importPaths.append(File.pathSeparator).append("$PYTHON_RUNFILES/").append(path);
      }
      // The executable zip file wil unzip itself into a tmp directory and then run from there
      String zipHeader =
          "#!/bin/sh\n"
              + "export TMPDIR=${TMPDIR:-/tmp/Bazel}\n"
              + "mkdir -p \"${TMPDIR}\"\n"
              + "export PYTHON_RUNFILES=$(mktemp -d \"${TMPDIR%%/}/runfiles.XXXXXXXX\")\n"
              + "export PYTHONPATH=\"$PYTHONPATH"
              + importPaths
              + "\"\n"
              + "unzip -q -o $0 -d \"$PYTHON_RUNFILES\" 2> /dev/null\n"
              + "retCode=0\n"
              + pythonBinary
              + " \"$PYTHON_RUNFILES/"
              + main
              + "\" $@ || retCode=$?\n"
              + "rm -rf \"$PYTHON_RUNFILES\"\n"
              + "exit $retCode\n";
      ruleContext.registerAction(
          new SpawnAction.Builder()
              .addInput(zipFile)
              .addOutput(executable)
              .setShellCommand(
                  "echo '"
                      + zipHeader
                      + "' | cat - "
                      + zipFile.getExecPathString()
                      + " > "
                      + executable.getExecPathString())
              .useDefaultShellEnvironment()
              .setMnemonic("BuildBinary")
              .build(ruleContext));
    }
  }

  @Override
  public void postInitBinary(RuleContext ruleContext, RunfilesSupport runfilesSupport,
      PyCommon common) throws InterruptedException {
    if (ruleContext.getConfiguration().buildPythonZip()) {
      FilesToRunProvider zipper = ruleContext.getExecutablePrerequisite("$zipper", Mode.HOST);
      if (!ruleContext.hasErrors()) {
        createPythonZipAction(
            ruleContext,
            common.getExecutable(),
            common.getPythonZipArtifact(),
            common.determineMainExecutableSource(false),
            zipper,
            runfilesSupport);
      }
    }
  }

  // TODO(pcloudy): This is a temporary workaround
  private static PathFragment getWorkspaceNameForPythonZip(String workspaceName) {
    // Currently, the default workspace name "__main__" will causing python can't find __main__.py
    // in executable zip file. Rename it to "main"
    if (workspaceName.equals(Label.DEFAULT_REPOSITORY_DIRECTORY)) {
      return new PathFragment("__default__");
    }
    return new PathFragment(workspaceName);
  }

  private static boolean isUnderWorkspace(PathFragment path) {
    return !path.startsWith(Label.EXTERNAL_PACKAGE_NAME);
  }

  private static String getRunfilesPath(PathFragment path, PathFragment workspaceName) {
    if (isUnderWorkspace(path)) {
      // If the file is under workspace, add workspace name as prefix
      return workspaceName.getRelative(path).normalize().toString();
    }
    // If the file is in external package, strip "external"
    return path.relativeTo(Label.EXTERNAL_PACKAGE_NAME).normalize().toString();
  }

  private static String getRunfilesPath(String path, PathFragment workspaceName) {
    return getRunfilesPath(new PathFragment(path), workspaceName);
  }

  private static void createPythonZipAction(
      RuleContext ruleContext,
      Artifact executable,
      Artifact zipFile,
      String main,
      FilesToRunProvider zipper,
      RunfilesSupport runfilesSupport) {

    NestedSetBuilder<Artifact> inputsBuilder = NestedSetBuilder.stableOrder();
    PathFragment workspaceName = getWorkspaceNameForPythonZip(ruleContext.getWorkspaceName());
    CustomCommandLine.Builder argv = new CustomCommandLine.Builder();

    argv.add("__main__.py=" + main);

    // Creating __init__.py files under each directory
    argv.add("__init__.py=");
    argv.add(getRunfilesPath("__init__.py", workspaceName) + "=");
    for (String path : runfilesSupport.getRunfiles().getEmptyFilenames()) {
      argv.add(getRunfilesPath(path, workspaceName) + "=");
    }

    // Read each runfile from execute path, add them into zip file at the right runfiles path.
    // Filter the executable file, cause we are building it.
    for (Artifact artifact : runfilesSupport.getRunfiles().getArtifacts()) {
      if (!artifact.equals(executable)) {
        argv.add(
            getRunfilesPath(artifact.getRunfilesPath(), workspaceName)
                + "="
                + artifact.getExecPathString());
        inputsBuilder.add(artifact);
      }
    }

    // zipper can only consume file list options from param file not other options,
    // so write file list in the param file first.
    Artifact paramFile =
        ruleContext.getDerivedArtifact(
            ParameterFile.derivePath(zipFile.getRootRelativePath()), zipFile.getRoot());

    ruleContext.registerAction(
        new ParameterFileWriteAction(
            ruleContext.getActionOwner(),
            paramFile,
            argv.build(),
            ParameterFile.ParameterFileType.UNQUOTED,
            ISO_8859_1));

    ruleContext.registerAction(
        new SpawnAction.Builder()
            .addInput(paramFile)
            .addTransitiveInputs(inputsBuilder.build())
            .addOutput(zipFile)
            .setExecutable(zipper)
            .useDefaultShellEnvironment()
            .addArgument("cC")
            .addArgument(zipFile.getExecPathString())
            .addArgument("@" + paramFile.getExecPathString())
            .setMnemonic("PythonZipper")
            .build(ruleContext));
  }
}
