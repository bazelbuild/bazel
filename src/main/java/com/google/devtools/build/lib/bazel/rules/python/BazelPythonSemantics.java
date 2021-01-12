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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
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
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.python.PyCcLinkParamsProvider;
import com.google.devtools.build.lib.rules.python.PyCommon;
import com.google.devtools.build.lib.rules.python.PyRuntimeInfo;
import com.google.devtools.build.lib.rules.python.PythonConfiguration;
import com.google.devtools.build.lib.rules.python.PythonSemantics;
import com.google.devtools.build.lib.rules.python.PythonUtils;
import com.google.devtools.build.lib.rules.python.PythonVersion;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/** Functionality specific to the Python rules in Bazel. */
public class BazelPythonSemantics implements PythonSemantics {

  public static final Runfiles.EmptyFilesSupplier GET_INIT_PY_FILES =
      new PythonUtils.GetInitPyFiles((Predicate<PathFragment> & Serializable) source -> false);
  private static final Template STUB_TEMPLATE =
      Template.forResource(BazelPythonSemantics.class, "python_stub_template.txt");
  public static final InstrumentationSpec PYTHON_COLLECTION_SPEC =
      new InstrumentationSpec(FileTypeSet.of(BazelPyRuleClasses.PYTHON_SOURCE))
          .withSourceAttributes("srcs")
          .withDependencyAttributes("deps", "data");

  public static final PathFragment ZIP_RUNFILES_DIRECTORY_NAME = PathFragment.create("runfiles");

  @Override
  public Runfiles.EmptyFilesSupplier getEmptyRunfilesSupplier() {
    return GET_INIT_PY_FILES;
  }

  @Override
  public String getSrcsVersionDocURL() {
    // TODO(#8996): Update URL to point to rules_python's docs instead of the Bazel site.
    return "https://docs.bazel.build/versions/master/be/python.html#py_binary.srcs_version";
  }

  @Override
  public void validate(RuleContext ruleContext, PyCommon common) {
  }

  @Override
  public boolean prohibitHyphensInPackagePaths() {
    return false;
  }

  @Override
  public void collectRunfilesForBinary(
      RuleContext ruleContext, Runfiles.Builder builder, PyCommon common, CcInfo ccInfo) {
    addRuntime(ruleContext, common, builder);
  }

  @Override
  public void collectDefaultRunfilesForBinary(
      RuleContext ruleContext, PyCommon common, Runfiles.Builder builder) {
    addRuntime(ruleContext, common, builder);
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

  private static String boolToLiteral(boolean value) {
    return value ? "True" : "False";
  }

  private static String versionToLiteral(PythonVersion version) {
    Preconditions.checkArgument(version.isTargetValue());
    return version == PythonVersion.PY3 ? "\"3\"" : "\"2\"";
  }

  private static void createStubFile(
      RuleContext ruleContext, Artifact stubOutput, PyCommon common, boolean isForZipFile) {
    PythonConfiguration config = ruleContext.getFragment(PythonConfiguration.class);
    BazelPythonConfiguration bazelConfig = ruleContext.getFragment(BazelPythonConfiguration.class);

    // The second-stage Python interpreter, which may be a system absolute path or a runfiles
    // workspace-relative path. On Windows this is also passed to the launcher to use for the
    // first-stage.
    String pythonBinary = getPythonBinary(ruleContext, common, bazelConfig);

    // Version information for host config diagnostic warning.
    PythonVersion attrVersion = PyCommon.readPythonVersionFromAttribute(ruleContext.attributes());
    boolean attrVersionSpecifiedExplicitly = attrVersion != null;
    if (!attrVersionSpecifiedExplicitly) {
      attrVersion = config.getDefaultPythonVersion();
    }

    // Create the stub file.
    ruleContext.registerAction(
        new TemplateExpansionAction(
            ruleContext.getActionOwner(),
            stubOutput,
            STUB_TEMPLATE,
            ImmutableList.of(
                Substitution.of(
                    "%main%", common.determineMainExecutableSource(/*withWorkspaceName=*/ true)),
                Substitution.of("%python_binary%", pythonBinary),
                Substitution.of("%imports%", Joiner.on(":").join(common.getImports().toList())),
                Substitution.of("%workspace_name%", ruleContext.getWorkspaceName()),
                Substitution.of("%is_zipfile%", boolToLiteral(isForZipFile)),
                Substitution.of(
                    "%import_all%", boolToLiteral(bazelConfig.getImportAllRepositories())),
                Substitution.of(
                    "%enable_host_version_warning%",
                    boolToLiteral(common.shouldWarnAboutHostVersionUponFailure())),
                Substitution.of(
                    "%target%", ruleContext.getRule().getLabel().getDefaultCanonicalForm()),
                Substitution.of(
                    "%python_version_from_config%", versionToLiteral(common.getVersion())),
                Substitution.of("%python_version_from_attr%", versionToLiteral(attrVersion)),
                Substitution.of(
                    "%python_version_specified_explicitly%",
                    boolToLiteral(attrVersionSpecifiedExplicitly))),
            true));
  }

  @Override
  public void createExecutable(
      RuleContext ruleContext, PyCommon common, CcInfo ccInfo, Runfiles.Builder runfilesBuilder)
      throws InterruptedException {
    PythonConfiguration config = ruleContext.getFragment(PythonConfiguration.class);
    BazelPythonConfiguration bazelConfig = ruleContext.getFragment(BazelPythonConfiguration.class);
    boolean buildPythonZip = config.buildPythonZip();

    /*
     * Python executable targets are launched in two stages. The first stage is the stub script that
     * locates (and possibly extracts) the runfiles tree, sets up environment variables, and passes
     * control to the second stage. The second stage is payload user code, i.e. the main Python
     * file.
     *
     * When a zip file is built (--build_python_zip), the stub script becomes the __main__.py of the
     * resulting zip, so that it runs when a Python interpreter executes the zip file. The stub
     * logic will extract the zip's runfiles into a temporary directory.
     *
     * The stub script has a shebang pointing to a first-stage Python interpreter (as of this
     * writing "#!/usr/bin/env python"). When a zip file is built on unix, this shebang is also
     * prepended to the final zip artifact. On Windows shebangs are ignored, and the launcher
     * runs the first stage with an interpreter whose path is passed in as LaunchInfo.
     */

    // The initial entry point, which is the launcher on Windows, or the stub or zip file on Unix.
    Artifact executable = common.getExecutable();

    // The second-stage Python interpreter, which may be a system absolute path or a runfiles
    // workspace-relative path. On Windows this is also passed to the launcher to use for the
    // first-stage.
    String pythonBinary = getPythonBinary(ruleContext, common, bazelConfig);

    // Create the stub file used for a non-zipfile executable. If --build_python_zip is true this is
    // never used so we skip it.
    if (!buildPythonZip) {
      Artifact stubOutput =
          OS.getCurrent() == OS.WINDOWS
              ? common.getPythonStubArtifactForWindows(executable)
              : executable;
      createStubFile(ruleContext, stubOutput, common, /* isForZipFile= */ false);
    }

    // Create the zip file if requested. On unix, copy it from the intermediate artifact to the
    // final executable while prepending the shebang.
    if (buildPythonZip) {
      Artifact zipFile = common.getPythonZipArtifact(executable);

      if (OS.getCurrent() != OS.WINDOWS) {
        PathFragment shExecutable = ShToolchain.getPathOrError(ruleContext);
        // TODO(#8685): Remove this special-case handling as part of making the proper shebang a
        // property of the Python toolchain configuration.
        String pythonExecutableName = OS.getCurrent() == OS.OPENBSD ? "python3" : "python";
        // NOTE: keep the following line intact to support nix builds
        String pythonShebang = "#!/usr/bin/env " + pythonExecutableName;
        ruleContext.registerAction(
            new SpawnAction.Builder()
                .addInput(zipFile)
                .addOutput(executable)
                .setShellCommand(
                    shExecutable,
                    "echo '"
                        + pythonShebang
                        + "' | cat - "
                        + zipFile.getExecPathString()
                        + " > "
                        + executable.getExecPathString())
                .useDefaultShellEnvironment()
                .setMnemonic("BuildBinary")
                .build(ruleContext));
      }
    }

    // On Windows, create the launcher.
    if (OS.getCurrent() == OS.WINDOWS) {
      createWindowsExeLauncher(
          ruleContext,
          // In the case where the second-stage interpreter is in runfiles, the launcher is passed
          // a workspace-relative path that it combines with its own CWD to produce the full path to
          // the real interpreter executable. (It can't use a path to the runfiles since they aren't
          // yet extracted from the zip, assuming buildPythonZip is set.)
          //
          // TODO(#7947): Fix how this path is constructed for the case of a runfile interpreter in
          // a remote repo -- probably need to pass an absolute path to the launcher instead of a
          // workspace-relative one. Also ensure this is ok for remote execution, and if not, maybe
          // change the launcher to use a separate system-installed first-stage interpreter like on
          // unix. See also https://github.com/bazelbuild/bazel/issues/7947#issuecomment-491385802.
          pythonBinary,
          executable,
          /*useZipFile=*/ buildPythonZip);
    }
  }

  /** Registers an action to create a Windows Python launcher at {@code pythonLauncher}. */
  private static void createWindowsExeLauncher(
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
  }

  @Override
  public void postInitExecutable(
      RuleContext ruleContext,
      RunfilesSupport runfilesSupport,
      PyCommon common,
      RuleConfiguredTargetBuilder builder) {
    FilesToRunProvider zipper = ruleContext.getExecutablePrerequisite("$zipper");
    Artifact executable = common.getExecutable();
    Artifact zipFile = common.getPythonZipArtifact(executable);

    if (!ruleContext.hasErrors()) {
      // Create the stub file that's needed by the python zip file.
      Artifact stubFileForZipFile = common.getPythonIntermediateStubArtifact(executable);
      createStubFile(ruleContext, stubFileForZipFile, common, /* isForZipFile= */ true);

      createPythonZipAction(
          ruleContext, executable, zipFile, stubFileForZipFile, zipper, runfilesSupport);
    }
    builder.addOutputGroup("python_zip_file", zipFile);
  }

  private static String getZipRunfilesPath(
      PathFragment path, PathFragment workspaceName, boolean legacyExternalRunfiles) {
    String zipRunfilesPath;
    if (legacyExternalRunfiles && path.startsWith(LabelConstants.EXTERNAL_PATH_PREFIX)) {
      // If the path starts with 'external' and --legacy_external_runfiles is set, this file is in
      // an external repository. Convert it to the new runfiles path by removing the 'external'
      // prefix.
      zipRunfilesPath = path.relativeTo(LabelConstants.EXTERNAL_PATH_PREFIX).toString();
    } else {
      // If not, it means the runfiles path is either under the workspace or an external file path
      // in the new runfiles path format. In either case, simply appending it to the workspace name
      // works just fine.
      zipRunfilesPath = workspaceName.getRelative(path).toString();
    }
    // We put the whole runfiles tree under the ZIP_RUNFILES_DIRECTORY_NAME directory, by doing this
    // , we avoid the conflict between default workspace name "__main__" and __main__.py file.
    // Note: This name has to be the same with the one in python_stub_template.txt.
    return ZIP_RUNFILES_DIRECTORY_NAME.getRelative(zipRunfilesPath).toString();
  }

  private static String getZipRunfilesPath(
      String path, PathFragment workspaceName, boolean legacyExternalRunfiles) {
    return getZipRunfilesPath(PathFragment.create(path), workspaceName, legacyExternalRunfiles);
  }

  private static void createPythonZipAction(
      RuleContext ruleContext,
      Artifact executable,
      Artifact zipFile,
      Artifact stubFile,
      FilesToRunProvider zipper,
      RunfilesSupport runfilesSupport) {

    NestedSetBuilder<Artifact> inputsBuilder = NestedSetBuilder.stableOrder();
    PathFragment workspaceName = runfilesSupport.getWorkspaceName();
    CustomCommandLine.Builder argv = new CustomCommandLine.Builder();
    inputsBuilder.add(stubFile);
    argv.addPrefixedExecPath("__main__.py=", stubFile);
    boolean legacyExternalRunfiles = ruleContext.getConfiguration().legacyExternalRunfiles();

    // Creating __init__.py files under each directory
    argv.add("__init__.py=");
    argv.addDynamicString(
        getZipRunfilesPath("__init__.py", workspaceName, legacyExternalRunfiles) + "=");
    for (String path : runfilesSupport.getRunfiles().getEmptyFilenames().toList()) {
      argv.addDynamicString(getZipRunfilesPath(path, workspaceName, legacyExternalRunfiles) + "=");
    }

    // Read each runfile from execute path, add them into zip file at the right runfiles path.
    // Filter the executable file, cause we are building it.
    for (Artifact artifact : runfilesSupport.getRunfilesArtifacts().toList()) {
      if (!artifact.equals(executable) && !artifact.equals(zipFile)) {
        argv.addDynamicString(
            getZipRunfilesPath(artifact.getRunfilesPath(), workspaceName, legacyExternalRunfiles)
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

  /**
   * Returns the Python runtime to use, either from the toolchain or the legacy flag-based
   * mechanism.
   *
   * <p>Can only be called for an executable Python rule.
   *
   * <p>Returns {@code null} if there's a problem retrieving the runtime.
   */
  @Nullable
  private static PyRuntimeInfo getRuntime(RuleContext ruleContext, PyCommon common) {
    return common.shouldGetRuntimeFromToolchain()
        ? common.getRuntimeFromToolchain()
        : ruleContext.getPrerequisite(":py_interpreter", PyRuntimeInfo.PROVIDER);
  }

  private static void addRuntime(
      RuleContext ruleContext, PyCommon common, Runfiles.Builder builder) {
    PyRuntimeInfo provider = getRuntime(ruleContext, common);
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
      RuleContext ruleContext, PyCommon common, BazelPythonConfiguration bazelConfig) {
    String pythonBinary;
    PyRuntimeInfo provider = getRuntime(ruleContext, common);
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
      pythonBinary = bazelConfig.getPythonPath();
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
