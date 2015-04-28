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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.base.CaseFormat.LOWER_UNDERSCORE;
import static com.google.common.base.CaseFormat.UPPER_CAMEL;
import static com.google.devtools.build.lib.rules.objc.XcodeProductType.LIBRARY_STATIC;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.proto.ProtoSourcesProvider;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

import javax.annotation.Nullable;

/**
 * Implementation for the "objc_proto_library" rule.
 */
public class ObjcProtoLibrary implements RuleConfiguredTargetFactory {
  private static final Function<Artifact, PathFragment> PARENT_PATHFRAGMENT =
      new Function<Artifact, PathFragment>() {
    @Override
    public PathFragment apply(Artifact input) {
      return input.getExecPath().getParentDirectory();
    }
  };

  @VisibleForTesting
  static final String NO_PROTOS_ERROR =
      "no protos to compile - a non-empty deps attribute is required";

  @Override
  public ConfiguredTarget create(final RuleContext ruleContext) throws InterruptedException {
    Artifact compileProtos = ruleContext.getPrerequisiteArtifact(
        ObjcRuleClasses.ObjcProtoRule.COMPILE_PROTOS_ATTR, Mode.HOST);
    Optional<Artifact> optionsFile = Optional.fromNullable(
        ruleContext.getPrerequisiteArtifact(ObjcProtoLibraryRule.OPTIONS_FILE_ATTR, Mode.HOST));
    NestedSet<Artifact> protos = NestedSetBuilder.<Artifact>stableOrder()
        .addAll(ruleContext.getPrerequisiteArtifacts("deps", Mode.TARGET)
            .filter(FileType.of(".proto"))
            .list())
        .addTransitive(maybeGetProtoSources(ruleContext))
        .build();

    if (Iterables.isEmpty(protos)) {
      ruleContext.ruleError(NO_PROTOS_ERROR);
    }

    ImmutableList<Artifact> libProtobuf = ruleContext
        .getPrerequisiteArtifacts(ObjcProtoLibraryRule.LIBPROTOBUF_ATTR, Mode.TARGET)
        .list();
    ImmutableList<Artifact> protoSupport = ruleContext
        .getPrerequisiteArtifacts(ObjcRuleClasses.ObjcProtoRule.PROTO_SUPPORT_ATTR, Mode.HOST)
        .list();

    // Generate sources in a package-and-rule-scoped directory; adds both the
    // package-and-rule-scoped directory and the header-containing-directory to the include path of
    // dependers.
    PathFragment rootRelativeOutputDir = new PathFragment(
        ruleContext.getLabel().getPackageFragment(),
        new PathFragment("_generated_protos_" + ruleContext.getLabel().getName()));
    PathFragment workspaceRelativeOutputDir = new PathFragment(
        ruleContext.getBinOrGenfilesDirectory().getExecPath(), rootRelativeOutputDir);
    PathFragment generatedProtoDir =
        new PathFragment(workspaceRelativeOutputDir, ruleContext.getLabel().getPackageFragment());

    boolean outputCpp =
        ruleContext.attributes().get(ObjcProtoLibraryRule.OUTPUT_CPP_ATTR, Type.BOOLEAN);

    ImmutableList<Artifact> protoGeneratedSources = outputArtifacts(
        ruleContext, rootRelativeOutputDir, protos, FileType.of(".pb." + (outputCpp ? "cc" : "m")),
        outputCpp);
    ImmutableList<Artifact> protoGeneratedHeaders = outputArtifacts(
        ruleContext, rootRelativeOutputDir, protos, FileType.of(".pb.h"), outputCpp);

    Artifact inputFileList = ruleContext.getAnalysisEnvironment().getDerivedArtifact(
        AnalysisUtils.getUniqueDirectory(ruleContext.getLabel(), new PathFragment("_protos"))
            .getRelative("_proto_input_files"),
            ruleContext.getConfiguration().getGenfilesDirectory());

    ruleContext.registerAction(new FileWriteAction(
        ruleContext.getActionOwner(),
        inputFileList,
        Artifact.joinExecPaths("\n", protos),
        false));

    CustomCommandLine.Builder commandLineBuilder = new CustomCommandLine.Builder()
        .add(compileProtos.getExecPathString())
        .add("--input-file-list").add(inputFileList.getExecPathString())
        .add("--output-dir").add(workspaceRelativeOutputDir.getSafePathString());
    if (optionsFile.isPresent()) {
        commandLineBuilder
            .add("--compiler-options-path")
            .add(optionsFile.get().getExecPathString());
    }
    if (outputCpp) {
      commandLineBuilder.add("--generate-cpp");
    }

    if (!Iterables.isEmpty(protos)) {
      ruleContext.registerAction(new SpawnAction.Builder()
          .setMnemonic("GenObjcProtos")
          .addInput(compileProtos)
          .addInputs(optionsFile.asSet())
          .addInputs(protos)
          .addInput(inputFileList)
          .addInputs(libProtobuf)
          .addInputs(protoSupport)
          .addOutputs(Iterables.concat(protoGeneratedSources, protoGeneratedHeaders))
          .setExecutable(new PathFragment("/usr/bin/python"))
          .setCommandLine(commandLineBuilder.build())
          .setExecutionInfo(ImmutableMap.of(ExecutionRequirements.REQUIRES_DARWIN, ""))
          .build(ruleContext));
    }

    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);
    CompilationArtifacts compilationArtifacts = new CompilationArtifacts.Builder()
        .addNonArcSrcs(protoGeneratedSources)
        .setIntermediateArtifacts(intermediateArtifacts)
        .setPchFile(Optional.<Artifact>absent())
        .build();

    ImmutableSet.Builder<PathFragment> searchPathEntriesBuilder =
        new ImmutableSet.Builder<PathFragment>()
            .add(workspaceRelativeOutputDir);
     if (ruleContext.getConfiguration().getFragment(ObjcConfiguration.class).perProtoIncludes()) {
      searchPathEntriesBuilder
          .add(generatedProtoDir)
          .addAll(Iterables.transform(protoGeneratedHeaders, PARENT_PATHFRAGMENT));
     }    
    ImmutableSet<PathFragment> searchPathEntries = searchPathEntriesBuilder.build();

    ObjcCommon common = new ObjcCommon.Builder(ruleContext)
        .setCompilationArtifacts(compilationArtifacts)
        .addUserHeaderSearchPaths(searchPathEntries)
        .addDepObjcProviders(ruleContext.getPrerequisites(
            ObjcProtoLibraryRule.LIBPROTOBUF_ATTR, Mode.TARGET, ObjcProvider.class))
        .setIntermediateArtifacts(intermediateArtifacts)
        .addHeaders(protoGeneratedHeaders)
        .addHeaders(protoGeneratedSources)
        .build();

    ObjcConfiguration configuration = ObjcRuleClasses.objcConfiguration(ruleContext);
    Iterable<XcodeProvider> protoDeps = ruleContext.getPrerequisites(
        ObjcProtoLibraryRule.LIBPROTOBUF_ATTR, Mode.TARGET, XcodeProvider.class);
    XcodeProvider xcodeProvider = new XcodeProvider.Builder()
        .setLabel(ruleContext.getLabel())
        .setArchitecture(configuration.getIosCpu())
        .addUserHeaderSearchPaths(searchPathEntries)
        .addPropagatedDependencies(protoDeps)
        .setConfigurationDistinguisher(configuration.getConfigurationDistinguisher())
        .addCopts(configuration.getCopts())
        .setProductType(LIBRARY_STATIC)
        .addHeaders(protoGeneratedHeaders)
        .setCompilationArtifacts(common.getCompilationArtifacts().get())
        .setObjcProvider(common.getObjcProvider())
        .build();

    ObjcActionsBuilder actionsBuilder = ObjcRuleClasses.actionsBuilder(ruleContext);
    new CompilationSupport(ruleContext)
        .registerCompileAndArchiveActions(common, OptionsProvider.DEFAULT);
    actionsBuilder.registerXcodegenActions(
        new ObjcRuleClasses.Tools(ruleContext),
        ruleContext.getImplicitOutputArtifact(XcodeSupport.PBXPROJ),
        XcodeProvider.Project.fromTopLevelTarget(xcodeProvider));

    return common.configuredTarget(
        NestedSetBuilder.<Artifact>stableOrder()
            .addAll(common.getCompiledArchive().asSet())
            .addAll(protoGeneratedSources)
            .addAll(protoGeneratedHeaders)
            .add(ruleContext.getImplicitOutputArtifact(XcodeSupport.PBXPROJ))
            .build(),
        Optional.of(xcodeProvider),
        Optional.of(common.getObjcProvider()),
        Optional.<XcTestAppProvider>absent(),
        Optional.<J2ObjcSrcsProvider>absent());
  }

  private NestedSet<Artifact> maybeGetProtoSources(RuleContext ruleContext) {
    NestedSetBuilder<Artifact> artifacts = new NestedSetBuilder<>(Order.STABLE_ORDER);
    Iterable<ProtoSourcesProvider> providers =
        ruleContext.getPrerequisites("deps", Mode.TARGET, ProtoSourcesProvider.class);
    for (ProtoSourcesProvider provider : providers) {
      artifacts.addTransitive(provider.getTransitiveProtoSources());
    }
    return artifacts.build();
  }

  private ImmutableList<Artifact> outputArtifacts(RuleContext ruleContext,
      PathFragment rootRelativeOutputDir, Iterable<Artifact> protos, FileType newFileType,
      boolean outputCpp) {
    ImmutableList.Builder<Artifact> builder = new ImmutableList.Builder<>();
    for (Artifact proto : protos) {
      String protoOutputName;
      if (outputCpp) {
        protoOutputName = proto.getFilename();
      } else {
        String lowerUnderscoreBaseName = proto.getFilename().replace('-', '_').toLowerCase();
        protoOutputName = LOWER_UNDERSCORE.to(UPPER_CAMEL, lowerUnderscoreBaseName);
      }
      PathFragment rawFragment = new PathFragment(
          rootRelativeOutputDir,
          proto.getExecPath().getParentDirectory(),
          new PathFragment(protoOutputName));
      @Nullable PathFragment outputFile = FileSystemUtils.replaceExtension(
          rawFragment,
          newFileType.getExtensions().get(0),
          ".proto");
      if (outputFile != null) {
        builder.add(ruleContext.getAnalysisEnvironment().getDerivedArtifact(
            outputFile, ruleContext.getBinOrGenfilesDirectory()));
      }
    }
    return builder.build();
  }
}
