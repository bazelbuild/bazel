// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.rules.python;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.extra.PythonInfo;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.LanguageDependentFragment;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.PseudoAction;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.Util;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesCollector.LocalMetadataCollector;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProviderImpl;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.GeneratedMessage.GeneratedExtension;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * A helper class for Python rules.
 */
public final class PyCommon {

  private static final LocalMetadataCollector METADATA_COLLECTOR = new LocalMetadataCollector() {
    @Override
    public void collectMetadataArtifacts(Iterable<Artifact> artifacts,
        AnalysisEnvironment analysisEnvironment, NestedSetBuilder<Artifact> metadataFilesBuilder) {
      // Python doesn't do any compilation, so we simply return the empty set.
    }
  };

  private final RuleContext ruleContext;

  private Artifact executable = null;

  private NestedSet<Artifact> transitivePythonSources;

  private PythonVersion sourcesVersion;
  private PythonVersion version = null;
  private Map<PathFragment, Artifact> convertedFiles;

  private NestedSet<Artifact> filesToBuild = null;

  public PyCommon(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
  }

  public void initCommon(PythonVersion defaultVersion) {
    this.sourcesVersion = getPythonVersionAttr(
        ruleContext, "srcs_version", PythonVersion.getAllValues());

    this.version = ruleContext.getFragment(PythonConfiguration.class)
        .getPythonVersion(defaultVersion);

    transitivePythonSources = collectTransitivePythonSources();

    checkSourceIsCompatible(this.version, this.sourcesVersion, ruleContext.getLabel());
  }

  public PythonVersion getVersion() {
    return version;
  }

  public void initBinary(List<Artifact> srcs) {
    Preconditions.checkNotNull(version);

    validatePackageName();
    executable = ruleContext.createOutputArtifact();
    if (this.version == PythonVersion.PY2AND3) {
      // TODO(bazel-team): we need to create two actions
      ruleContext.ruleError("PY2AND3 is not yet implemented");
    }

    filesToBuild = NestedSetBuilder.<Artifact>stableOrder()
        .addAll(srcs)
        .add(executable)
        .build();

    if (ruleContext.hasErrors()) {
      return;
    }

    addPyExtraActionPseudoAction();
  }

  public void addCommonTransitiveInfoProviders(RuleConfiguredTargetBuilder builder,
      PythonSemantics semantics, NestedSet<Artifact> filesToBuild) {
    PythonSourcesProvider sourcesProvider =
        new PythonSourcesProvider(transitivePythonSources, usesSharedLibraries());
    builder
        .add(InstrumentedFilesProvider.class, new InstrumentedFilesProviderImpl(
            new InstrumentedFilesCollector(ruleContext,
                semantics.getCoverageInstrumentationSpec(), METADATA_COLLECTOR,
                filesToBuild)))
        .add(PythonSourcesProvider.class, sourcesProvider)
        .addSkylarkTransitiveInfo(PythonSourcesProvider.SKYLARK_NAME, sourcesProvider)
        // Python targets are not really compilable. The best we can do is make sure that all
        // generated source files are ready.
        .addOutputGroup(OutputGroupProvider.FILES_TO_COMPILE, transitivePythonSources)
        .addOutputGroup(OutputGroupProvider.COMPILATION_PREREQUISITES, transitivePythonSources);
  }

  public PythonVersion getDefaultPythonVersion() {
    return ruleContext.getRule()
        .isAttrDefined("default_python_version", Type.STRING)
            ? getPythonVersionAttr(
                ruleContext, "default_python_version", PythonVersion.PY2, PythonVersion.PY3)
            : null;
  }

  public static PythonVersion getPythonVersionAttr(RuleContext ruleContext,
      String attrName, PythonVersion... allowed) {
    String stringAttr = ruleContext.attributes().get(attrName, Type.STRING);
    PythonVersion version = PythonVersion.parse(stringAttr, allowed);
    if (version != null) {
      return version;
    }
    ruleContext.attributeError(attrName,
        "'" + stringAttr + "' is not a valid value. Expected one of: " + Joiner.on(", ")
            .join(allowed));
    return PythonVersion.defaultValue();
  }

  /**
   * Returns a mutable List of the source Artifacts.
   */
  public List<Artifact> validateSrcs() {
    List<Artifact> sourceFiles = new ArrayList<>();
    // TODO(bazel-team): Need to get the transitive deps closure, not just the
    //                 sources of the rule.
    for (FileProvider src : ruleContext
        .getPrerequisites("srcs", Mode.TARGET, FileProvider.class)) {
      // Make sure that none of the sources contain hyphens.
      if (Util.containsHyphen(src.getLabel().getPackageFragment())) {
        ruleContext.attributeError("srcs", src.getLabel() + ": package name may not contain '-'");
      }
      Iterable<Artifact> pySrcs = FileType.filter(src.getFilesToBuild(),
          PyRuleClasses.PYTHON_SOURCE);
      Iterables.addAll(sourceFiles, pySrcs);
      if (Iterables.isEmpty(pySrcs)) {
        ruleContext.attributeWarning("srcs",
            "rule '" + src.getLabel() + "' does not produce any Python source files");
      }
    }

    LanguageDependentFragment.Checker.depsSupportsLanguage(ruleContext, PyRuleClasses.LANGUAGE);
    return convertedFiles != null
        ? ImmutableList.copyOf(convertedFiles.values())
        : sourceFiles;
  }

  /**
   * Checks that the package name of this Python rule does not contain a '-'.
   */
  void validatePackageName() {
    if (Util.containsHyphen(ruleContext.getLabel().getPackageFragment())) {
      ruleContext.ruleError("package name may not contain '-'");
    }
  }

  /**
   * Adds a {@link PseudoAction} to the build graph that is only used
   * for providing information to the blaze extra_action feature.
   */
  void addPyExtraActionPseudoAction() {
    if (ruleContext.getConfiguration().getActionListeners().isEmpty()) {
      return;
    }

    // Has to be unfiltered sources as filtered will give an error for
    // unsupported file types where as certain tests only expect a warning.
    Collection<Artifact> sources = ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET).list();

    // We need to do it in this convoluted way because we must not add the files declared in the
    // srcs of this rule. Note that it is not enough to remove the direct members from the nested
    // set of the current rule, because the same files may have been declared in a dependency, too.
    NestedSetBuilder<Artifact> depBuilder = NestedSetBuilder.compileOrder();
    collectTransitivePythonSourcesFromDeps(depBuilder);
    NestedSet<Artifact> dependencies = depBuilder.build();

    PythonInfo info = PythonInfo.newBuilder()
        .addAllSourceFile(Artifact.toExecPaths(sources))
        .addAllDepFile(Artifact.toExecPaths(dependencies))
        .build();

    ruleContext.getAnalysisEnvironment()
        .registerAction(new PyPseudoAction(ruleContext.getActionOwner(),
            ImmutableList.copyOf(Iterables.concat(sources, dependencies)),
            ImmutableList.of(PseudoAction.getDummyOutput(ruleContext)), "Python",
            PythonInfo.pythonInfo, info));
  }

  private void addSourceFiles(NestedSetBuilder<Artifact> builder, Iterable<Artifact> artifacts) {
    Preconditions.checkState(convertedFiles == null);
    if (sourcesVersion == PythonVersion.PY2 && version == PythonVersion.PY3) {
      convertedFiles = PythonUtils.generate2to3Actions(ruleContext, artifacts);
    }
    builder.addAll(artifacts);
  }

  private void collectTransitivePythonSourcesFromDeps(NestedSetBuilder<Artifact> builder) {
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("deps", Mode.TARGET)) {
      if (dep.getProvider(PythonSourcesProvider.class) != null) {
        PythonSourcesProvider provider = dep.getProvider(PythonSourcesProvider.class);
        builder.addTransitive(provider.getTransitivePythonSources());
      } else {
        // TODO(bazel-team): We also collect .py source files from deps (e.g. for proto_library
        // rules). Rules should implement PythonSourcesProvider instead.
        FileProvider provider = dep.getProvider(FileProvider.class);
        builder.addAll(FileType.filter(provider.getFilesToBuild(), PyRuleClasses.PYTHON_SOURCE));
      }
    }
  }

  private NestedSet<Artifact> collectTransitivePythonSources() {
    NestedSetBuilder<Artifact> builder =
        NestedSetBuilder.compileOrder();
    collectTransitivePythonSourcesFromDeps(builder);
    addSourceFiles(builder, ruleContext
        .getPrerequisiteArtifacts("srcs", Mode.TARGET).filter(PyRuleClasses.PYTHON_SOURCE).list());
    return builder.build();
  }

  /**
   * Checks that the source file version is compatible with the Python interpreter.
   */
  private void checkSourceIsCompatible(PythonVersion targetVersion, PythonVersion sourceVersion,
                                          Label source) {
    if (targetVersion == PythonVersion.PY2 || targetVersion == PythonVersion.PY2AND3) {
      if (sourceVersion == PythonVersion.PY3ONLY) {
        ruleContext.ruleError("Rule '" + source
                  + "' can only be used with Python 3, and cannot be converted to Python 2");
      } else if (sourceVersion == PythonVersion.PY3) {
        ruleContext.ruleError("Rule '" + source
                  + "' need to be converted to Python 2 (not yet implemented)");
      }
    }
    if (targetVersion == PythonVersion.PY3 || targetVersion == PythonVersion.PY2AND3) {
      if (sourceVersion == PythonVersion.PY2ONLY) {
        ruleContext.ruleError("Rule '" + source
                  + "' can only be used with Python 2, and cannot be converted to Python 3");
      }
    }
  }

  /**
   * @return A String that is the full path to the main python entry point.
   */
  public String determineMainExecutableSource() {
    String mainSourceName;
    Rule target = ruleContext.getRule();
    boolean explicitMain = target.isAttributeValueExplicitlySpecified("main");
    if (explicitMain) {
      mainSourceName = ruleContext.attributes().get("main", Type.LABEL).getName();
      if (!mainSourceName.endsWith(".py")) {
        ruleContext.attributeError("main", "main must end in '.py'");
      }
    } else {
      String ruleName = target.getName();
      if (ruleName.endsWith(".py")) {
        ruleContext.attributeError("name", "name must not end in '.py'");
      }
      mainSourceName = ruleName + ".py";
    }
    PathFragment mainSourcePath = new PathFragment(mainSourceName);

    Artifact mainArtifact = null;
    for (Artifact outItem : ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET).list()) {
      if (outItem.getRootRelativePath().endsWith(mainSourcePath)) {
        if (mainArtifact == null) {
          mainArtifact = outItem;
        } else {
          ruleContext.attributeError("srcs",
              buildMultipleMainMatchesErrorText(explicitMain, mainSourceName,
                  mainArtifact.getRootRelativePath().toString(),
                  outItem.getRootRelativePath().toString()));
        }
      }
    }

    if (mainArtifact == null) {
      ruleContext.attributeError("srcs", buildNoMainMatchesErrorText(explicitMain, mainSourceName));
      return null;
    }

    PathFragment workspaceName = new PathFragment(ruleContext.getRule().getWorkspaceName());
    return workspaceName.getRelative(mainArtifact.getRootRelativePath()).getPathString();
  }

  public Artifact getExecutable() {
    return executable;
  }

  public Map<PathFragment, Artifact> getConvertedFiles() {
    return convertedFiles;
  }

  public NestedSet<Artifact> getFilesToBuild() {
    return filesToBuild;
  }

  public boolean usesSharedLibraries() {
    return checkForSharedLibraries(Iterables.concat(
        ruleContext.getPrerequisites("deps", Mode.TARGET),
        ruleContext.getPrerequisites("data", Mode.DATA)));
  }

  /**
   * Returns true if this target has an .so file in its transitive dependency closure.
   */
  public static boolean checkForSharedLibraries(Iterable<TransitiveInfoCollection> deps) {
    for (TransitiveInfoCollection dep : deps) {
      PythonSourcesProvider provider = dep.getProvider(PythonSourcesProvider.class);
      if (provider != null) {
        if (provider.usesSharedLibraries()) {
          return true;
        }
      } else if (FileType.contains(
          dep.getProvider(FileProvider.class).getFilesToBuild(), CppFileTypes.SHARED_LIBRARY)) {
        return true;
      }
    }

    return false;
  }

  private static String buildMultipleMainMatchesErrorText(boolean explicit, String proposedMainName,
      String match1, String match2) {
    String errorText;
    if (explicit) {
      errorText = "file name '" + proposedMainName
          + "' specified by 'main' attribute matches multiple files: e.g., '" + match1
          + "' and '" + match2 + "'";
    } else {
      errorText = "default main file name '" + proposedMainName
          + "' matches multiple files.  Perhaps specify an explicit file with 'main' attribute?  "
          + "Matches were: '" + match1 + "' and '" + match2 + "'";
    }
    return errorText;
  }

  private static String buildNoMainMatchesErrorText(boolean explicit, String proposedMainName) {
    String errorText;
    if (explicit) {
      errorText = "could not find '" + proposedMainName
          + "' as specified by 'main' attribute";
    } else {
      errorText = "corresponding default '" + proposedMainName + "' does not appear in srcs. Add it"
          + " or override default file name with a 'main' attribute";
    }
    return errorText;
  }

  // Used purely to set the legacy ActionType of the ExtraActionInfo.
  private static class PyPseudoAction extends PseudoAction<PythonInfo> {
    private static final UUID ACTION_UUID = UUID.fromString("8d720129-bc1a-481f-8c4c-dbe11dcef319");

    public PyPseudoAction(ActionOwner owner,
        Collection<Artifact> inputs, Collection<Artifact> outputs,
        String mnemonic, GeneratedExtension<ExtraActionInfo, PythonInfo> infoExtension,
        PythonInfo info) {
      super(ACTION_UUID, owner, inputs, outputs, mnemonic, infoExtension, info);
    }

    @Override
    public ExtraActionInfo.Builder getExtraActionInfo() {
      return super.getExtraActionInfo();
    }
  }
}
