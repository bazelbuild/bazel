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
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.ShToolchain;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.LauncherFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.LauncherFileWriteAction.LaunchInfo;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.Substitution;
import com.google.devtools.build.lib.analysis.actions.Template;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.python.PyCcLinkParamsProvider;
import com.google.devtools.build.lib.rules.python.PyCommon;
import com.google.devtools.build.lib.rules.python.PyRuntimeInfo;
import com.google.devtools.build.lib.rules.python.PythonConfiguration;
import com.google.devtools.build.lib.rules.python.PythonSemantics;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Functionality specific to the Python rules in Bazel.
 */
public class BazelPythonSemantics implements PythonSemantics {
  private static final Template STUB_TEMPLATE =
      Template.forResource(BazelPythonSemantics.class, "python_stub_template.txt");
  public static final InstrumentationSpec PYTHON_COLLECTION_SPEC = new InstrumentationSpec(
      FileTypeSet.of(BazelPyRuleClasses.PYTHON_SOURCE),
      "srcs", "deps", "data");

  public static final PathFragment ZIP_RUNFILES_DIRECTORY_NAME = PathFragment.create("runfiles");

  @Override
  public void validate(RuleContext ruleContext, PyCommon common) {
  }

  @Override
  public void collectRunfilesForBinary(
      RuleContext ruleContext, Runfiles.Builder builder, PyCommon common, CcInfo ccInfo) {
    addRuntime(ruleContext, builder);
  }

  @Override
  public void collectDefaultRunfilesForBinary(RuleContext ruleContext, Runfiles.Builder builder) {
    addRuntime(ruleContext, builder);
  }

  @Override
  public void collectDefaultRunfiles(RuleContext ruleContext, Runfiles.Builder builder) {
    builder.addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES);
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
  public List<String> getImports(RuleContext ruleContext) {
    List<String> result = new ArrayList<>();
    PathFragment packageFragment = ruleContext.getLabel().getPackageIdentifier().getRunfilesPath();
    // Python scripts start with x.runfiles/ as the module space, so everything must be manually
    // adjusted to be relative to the workspace name.
    packageFragment = PathFragment.create(ruleContext.getWorkspaceName())
        .getRelative(packageFragment);
    for (String importsAttr : ruleContext.getExpander().list("imports")) {
      if (importsAttr.startsWith("/")) {
        ruleContext.attributeWarning("imports",
            "ignoring invalid absolute path '" + importsAttr + "'");
        continue;
      }
      PathFragment importsPath = packageFragment.getRelative(importsAttr);
      if (importsPath.containsUplevelReferences()) {
        ruleContext.attributeError("imports",
            "Path " + importsAttr + " references a path above the execution root");
      }
      result.add(importsPath.getPathString());
    }
    return result;
  }

  /** @return An artifact next to the executable file with ".temp" suffix */
  public Artifact getPythonTemplateMainArtifact(RuleContext ruleContext, Artifact executable) {
    return ruleContext.getRelatedArtifact(executable.getRootRelativePath(), ".temp");
  }

  @Override
  public Artifact createExecutable(
      RuleContext ruleContext,
      PyCommon common,
      CcInfo ccInfo,
      Runfiles.Builder runfilesBuilder)
      throws InterruptedException {
    String main = common.determineMainExecutableSource(/*withWorkspaceName=*/ true);
    Artifact executable = common.getExecutable();
    BazelPythonConfiguration config = ruleContext.getFragment(BazelPythonConfiguration.class);
    String pythonBinary = getPythonBinary(ruleContext, config);

    if (!ruleContext.getFragment(PythonConfiguration.class).buildPythonZip()) {
      Artifact stubOutput = executable;
      if (OS.getCurrent() == OS.WINDOWS) {
        // On Windows, use a Windows native binary to launch the python launcher script (stub file).
        stubOutput = common.getPythonLauncherArtifact(executable);
        executable =
            createWindowsExeLauncher(ruleContext, pythonBinary, executable, /*useZipFile*/ false);
      }

      ruleContext.registerAction(
          new TemplateExpansionAction(
              ruleContext.getActionOwner(),
              stubOutput,
              STUB_TEMPLATE,
              ImmutableList.of(
                  Substitution.of("%main%", main),
                  Substitution.of("%python_binary%", pythonBinary),
                  Substitution.of("%imports%", Joiner.on(":").join(common.getImports())),
                  Substitution.of("%workspace_name%", ruleContext.getWorkspaceName()),
                  Substitution.of("%is_zipfile%", "False"),
                  Substitution.of(
                      "%import_all%", config.getImportAllRepositories() ? "True" : "False")),
              true));
    } else {
      Artifact zipFile = common.getPythonZipArtifact(executable);
      Artifact templateMain = getPythonTemplateMainArtifact(ruleContext, executable);
      // The executable zip file will unzip itself into a tmp directory and then run from there
      ruleContext.registerAction(
          new TemplateExpansionAction(
              ruleContext.getActionOwner(),
              templateMain,
              STUB_TEMPLATE,
              ImmutableList.of(
                  Substitution.of("%main%", main),
                  Substitution.of("%python_binary%", pythonBinary),
                  Substitution.of("%imports%", Joiner.on(":").join(common.getImports())),
                  Substitution.of("%workspace_name%", ruleContext.getWorkspaceName()),
                  Substitution.of("%is_zipfile%", "True"),
                  Substitution.of(
                      "%import_all%", config.getImportAllRepositories() ? "True" : "False")),
              true));

      if (OS.getCurrent() != OS.WINDOWS) {
        PathFragment shExecutable = ShToolchain.getPathOrError(ruleContext);
        ruleContext.registerAction(
            new SpawnAction.Builder()
                .addInput(zipFile)
                .addOutput(executable)
                .setShellCommand(
                    shExecutable,
                    "echo '#!/usr/bin/env python' | cat - "
                        + zipFile.getExecPathString()
                        + " > "
                        + executable.getExecPathString())
                .useDefaultShellEnvironment()
                .setMnemonic("BuildBinary")
                .build(ruleContext));
      } else {
        return createWindowsExeLauncher(ruleContext, pythonBinary, executable, true);
      }
    }

    return executable;
  }

  private static Artifact createWindowsExeLauncher(
      RuleContext ruleContext, String pythonBinary, Artifact pythonLauncher, boolean useZipFile)
      throws InterruptedException {
    LaunchInfo launchInfo =
        LaunchInfo.builder()
            .addKeyValuePair("binary_type", "Python")
            .addKeyValuePair("workspace_name", ruleContext.getWorkspaceName())
            .addKeyValuePair(
                "symlink_runfiles_enabled",
                ruleContext.getConfiguration().runfilesEnabled() ? "1" : "0")
            .addKeyValuePair("python_bin_path", pythonBinary)
            .addKeyValuePair("use_zip_file", useZipFile ? "1" : "0")
            .build();
    LauncherFileWriteAction.createAndRegister(ruleContext, pythonLauncher, launchInfo);
    return pythonLauncher;
  }

  @Override
  public void postInitExecutable(
      RuleContext ruleContext, RunfilesSupport runfilesSupport, PyCommon common) {
    if (ruleContext.getFragment(PythonConfiguration.class).buildPythonZip()) {
      FilesToRunProvider zipper = ruleContext.getExecutablePrerequisite("$zipper", Mode.HOST);
      Artifact executable = common.getExecutable();
      if (!ruleContext.hasErrors()) {
        createPythonZipAction(
            ruleContext,
            executable,
            common.getPythonZipArtifact(executable),
            getPythonTemplateMainArtifact(ruleContext, executable),
            zipper,
            runfilesSupport);
      }
    }
  }

  private static boolean isUnderWorkspace(PathFragment path) {
    return !path.startsWith(LabelConstants.EXTERNAL_PATH_PREFIX);
  }

  private static String getZipRunfilesPath(PathFragment path, PathFragment workspaceName) {
    String zipRunfilesPath;
    if (isUnderWorkspace(path)) {
      // If the file is under workspace, add workspace name as prefix
      zipRunfilesPath = workspaceName.getRelative(path).toString();
    } else {
      // If the file is in external package, strip "external"
      zipRunfilesPath = path.relativeTo(LabelConstants.EXTERNAL_PATH_PREFIX).toString();
    }
    // We put the whole runfiles tree under the ZIP_RUNFILES_DIRECTORY_NAME directory, by doing this
    // , we avoid the conflict between default workspace name "__main__" and __main__.py file.
    // Note: This name has to be the same with the one in python_stub_template.txt.
    return ZIP_RUNFILES_DIRECTORY_NAME.getRelative(zipRunfilesPath).toString();
  }

  private static String getZipRunfilesPath(String path, PathFragment workspaceName) {
    return getZipRunfilesPath(PathFragment.create(path), workspaceName);
  }

  private static void createPythonZipAction(
      RuleContext ruleContext,
      Artifact executable,
      Artifact zipFile,
      Artifact templateMain,
      FilesToRunProvider zipper,
      RunfilesSupport runfilesSupport) {

    NestedSetBuilder<Artifact> inputsBuilder = NestedSetBuilder.stableOrder();
    PathFragment workspaceName = runfilesSupport.getWorkspaceName();
    CustomCommandLine.Builder argv = new CustomCommandLine.Builder();
    inputsBuilder.add(templateMain);
    argv.addPrefixedExecPath("__main__.py=", templateMain);

    // Creating __init__.py files under each directory
    argv.add("__init__.py=");
    argv.addDynamicString(getZipRunfilesPath("__init__.py", workspaceName) + "=");
    for (String path : runfilesSupport.getRunfiles().getEmptyFilenames()) {
      argv.addDynamicString(getZipRunfilesPath(path, workspaceName) + "=");
    }

    // Read each runfile from execute path, add them into zip file at the right runfiles path.
    // Filter the executable file, cause we are building it.
    for (Artifact artifact : runfilesSupport.getRunfilesArtifacts()) {
      if (!artifact.equals(executable) && !artifact.equals(zipFile)) {
        argv.addDynamicString(
            getZipRunfilesPath(artifact.getRunfilesPath(), workspaceName)
                + "="
                + artifact.getExecPathString());
        inputsBuilder.add(artifact);
      }
    }

    ruleContext.registerAction(
        new SpawnAction.Builder()
            .addTransitiveInputs(inputsBuilder.build())
            .addOutput(zipFile)
            .setExecutable(zipper)
            .useDefaultShellEnvironment()
            .addCommandLine(CustomCommandLine.builder().add("cC").addExecPath(zipFile).build())
            // zipper can only consume file list options from param file not other options,
            // so write file list in the param file.
            .addCommandLine(
                argv.build(),
                ParamFileInfo.builder(ParameterFile.ParameterFileType.UNQUOTED)
                    .setCharset(ISO_8859_1)
                    .setUseAlways(true)
                    .build())
            .setMnemonic("PythonZipper")
            .build(ruleContext));
  }

  private static void addRuntime(RuleContext ruleContext, Runfiles.Builder builder) {
    PyRuntimeInfo provider =
        ruleContext.getPrerequisite(":py_interpreter", Mode.TARGET, PyRuntimeInfo.PROVIDER);
    if (provider != null && provider.isInBuild()) {
      builder.addArtifact(provider.getInterpreter());
      // WARNING: we are adding the all Python runtime files here,
      // and it would fail if the filenames of them contain spaces.
      // Currently, we need to exclude them in py_runtime rules.
      // Possible files in Python runtime which contain spaces in filenames:
      // - https://github.com/pypa/setuptools/blob/master/setuptools/script%20(dev).tmpl
      // - https://github.com/pypa/setuptools/blob/master/setuptools/command/launcher%20manifest.xml
      builder.addTransitiveArtifacts(provider.getFiles());
    }
  }

  private static String getPythonBinary(
      RuleContext ruleContext,
      BazelPythonConfiguration config) {

    String pythonBinary;

    PyRuntimeInfo provider =
        ruleContext.getPrerequisite(":py_interpreter", Mode.TARGET, PyRuntimeInfo.PROVIDER);

    if (provider != null) {
      // make use of py_runtime defined by --python_top
      if (!provider.isInBuild()) {
        // absolute Python path in py_runtime
        pythonBinary = provider.getInterpreterPath().getPathString();
      } else {
        // checked in Python interpreter in py_runtime
        PathFragment workspaceName =
            PathFragment.create(ruleContext.getRule().getPackage().getWorkspaceName());
        pythonBinary =
            workspaceName.getRelative(provider.getInterpreter().getRunfilesPath()).getPathString();
      }
    } else  {
      // make use of the Python interpreter in an absolute path
      pythonBinary = config.getPythonPath();
    }

    return pythonBinary;
  }

  @Override
  public CcInfo buildCcInfoProvider(Iterable<? extends TransitiveInfoCollection> deps) {
    ImmutableList<CcInfo> ccInfos =
        ImmutableList.<CcInfo>builder()
            .addAll(AnalysisUtils.getProviders(deps, CcInfo.PROVIDER))
            .addAll(
                Streams.stream(AnalysisUtils.getProviders(deps, PyCcLinkParamsProvider.PROVIDER))
                    .map(PyCcLinkParamsProvider::getCcInfo)
                    .collect(ImmutableList.toImmutableList()))
            .build();

    // TODO(plf): return empty CcInfo.
    return CcInfo.merge(ccInfos);
  }
}
