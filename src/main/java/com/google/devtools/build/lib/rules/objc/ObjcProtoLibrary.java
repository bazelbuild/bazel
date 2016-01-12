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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.base.CaseFormat.LOWER_UNDERSCORE;
import static com.google.common.base.CaseFormat.UPPER_CAMEL;
import static com.google.devtools.build.lib.rules.objc.XcodeProductType.LIBRARY_STATIC;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.proto.ProtoSourcesProvider;
import com.google.devtools.build.lib.syntax.Type;
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

  @VisibleForTesting
  static final String FILES_DEPRECATED_WARNING =
      "Using files and filegroups in objc_proto_library is deprecated";

  @Override
  public ConfiguredTarget create(final RuleContext ruleContext) throws InterruptedException {
    Artifact compileProtos = ruleContext.getPrerequisiteArtifact(
        ObjcProtoLibraryRule.COMPILE_PROTOS_ATTR, Mode.HOST);
    Optional<Artifact> optionsFile = Optional.fromNullable(
        ruleContext.getPrerequisiteArtifact(ObjcProtoLibraryRule.OPTIONS_FILE_ATTR, Mode.HOST));

    NestedSet<Artifact> protos = NestedSetBuilder.<Artifact>stableOrder()
        .addAll(maybeGetProtoFiles(ruleContext))
        .addTransitive(maybeGetProtoSources(ruleContext))
        .build();

    if (Iterables.isEmpty(protos)) {
      ruleContext.ruleError(NO_PROTOS_ERROR);
    }

    ImmutableList<Artifact> libProtobuf = ruleContext
        .getPrerequisiteArtifacts(ObjcProtoLibraryRule.LIBPROTOBUF_ATTR, Mode.TARGET)
        .list();
    ImmutableList<Artifact> protoSupport = ruleContext
        .getPrerequisiteArtifacts(ObjcProtoLibraryRule.PROTO_SUPPORT_ATTR, Mode.HOST)
        .list();

    // Generate sources in a package-and-rule-scoped directory; adds both the
    // package-and-rule-scoped directory and the header-containing-directory to the include path of
    // dependers.
    String uniqueDirectoryName = "_generated_protos";

    PathFragment rootRelativeOutputDir = ruleContext.getUniqueDirectory(uniqueDirectoryName);
    PathFragment workspaceRelativeOutputDir = new PathFragment(
        ruleContext.getBinOrGenfilesDirectory().getExecPath(), rootRelativeOutputDir);
    PathFragment generatedProtoDir =
        new PathFragment(workspaceRelativeOutputDir, ruleContext.getLabel().getPackageFragment());

    boolean outputCpp =
        ruleContext.attributes().get(ObjcProtoLibraryRule.OUTPUT_CPP_ATTR, Type.BOOLEAN);

    boolean useObjcHeaderNames =
         ruleContext.attributes().get(
             ObjcProtoLibraryRule.USE_OBJC_HEADER_NAMES_ATTR, Type.BOOLEAN);

    ImmutableList<Artifact> protoGeneratedSources = outputArtifacts(
        ruleContext, uniqueDirectoryName, protos, FileType.of(".pb." + (outputCpp ? "cc" : "m")),
        outputCpp);
    ImmutableList<Artifact> protoGeneratedHeaders = outputArtifacts(
        ruleContext, uniqueDirectoryName, protos,
        FileType.of(".pb" + (useObjcHeaderNames ? "objc.h" : ".h")), outputCpp);

    Artifact inputFileList = ruleContext.getUniqueDirectoryArtifact(
        "_protos", "_proto_input_files", ruleContext.getConfiguration().getGenfilesDirectory());

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
    if (useObjcHeaderNames) {
      commandLineBuilder.add("--use-objc-header-names");
    }

    if (!Iterables.isEmpty(protos)) {
      ruleContext.registerAction(ObjcRuleClasses.spawnOnDarwinActionBuilder()
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
          .build(ruleContext));
    }

    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);
    CompilationArtifacts compilationArtifacts = new CompilationArtifacts.Builder()
        .addNonArcSrcs(protoGeneratedSources)
        .setIntermediateArtifacts(intermediateArtifacts)
        .setPchFile(Optional.<Artifact>absent())
        .addAdditionalHdrs(protoGeneratedHeaders)
        .addAdditionalHdrs(protoGeneratedSources)
        .build();

    ImmutableSet.Builder<PathFragment> searchPathEntriesBuilder =
        new ImmutableSet.Builder<PathFragment>()
            .add(workspaceRelativeOutputDir);
    if (ruleContext.attributes().get(
        ObjcProtoLibraryRule.PER_PROTO_INCLUDES, Type.BOOLEAN)) {
      searchPathEntriesBuilder
          .add(generatedProtoDir)
          .addAll(Iterables.transform(protoGeneratedHeaders, PARENT_PATHFRAGMENT));
    }
    ImmutableSet<PathFragment> searchPathEntries = searchPathEntriesBuilder.build();

    ObjcCommon common =
        new ObjcCommon.Builder(ruleContext)
            .setCompilationArtifacts(compilationArtifacts)
            .addUserHeaderSearchPaths(searchPathEntries)
            .addDepObjcProviders(
                ruleContext.getPrerequisites(
                    ObjcProtoLibraryRule.LIBPROTOBUF_ATTR, Mode.TARGET, ObjcProvider.class))
            .setIntermediateArtifacts(intermediateArtifacts)
            .setHasModuleMap()
            .build();

    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.<Artifact>stableOrder()
        .addAll(common.getCompiledArchive().asSet())
        .addAll(protoGeneratedSources)
        .addAll(protoGeneratedHeaders);

    ObjcConfiguration configuration = ObjcRuleClasses.objcConfiguration(ruleContext);
    XcodeProvider.Builder xcodeProviderBuilder = new XcodeProvider.Builder()
        .addUserHeaderSearchPaths(searchPathEntries)
        .addCopts(configuration.getCopts())
        .addHeaders(protoGeneratedHeaders)
        .setCompilationArtifacts(common.getCompilationArtifacts().get());

    new CompilationSupport(ruleContext)
        .registerCompileAndArchiveActions(common);

    new XcodeSupport(ruleContext)
        .addFilesToBuild(filesToBuild)
        .addXcodeSettings(xcodeProviderBuilder, common.getObjcProvider(), LIBRARY_STATIC)
        .addDependencies(
            xcodeProviderBuilder, new Attribute(ObjcProtoLibraryRule.LIBPROTOBUF_ATTR, Mode.TARGET))
        .registerActions(xcodeProviderBuilder.build());

    return ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build())
        .addProvider(XcodeProvider.class, xcodeProviderBuilder.build())
        .addProvider(ObjcProvider.class, common.getObjcProvider())
        .build();
  }

  /**
   * Get .proto files added to the deps attribute. This is for backwards compatibility,
   * and emits a warning.
   */
  private ImmutableList<Artifact> maybeGetProtoFiles(RuleContext ruleContext) {
    PrerequisiteArtifacts prerequisiteArtifacts =
        ruleContext.getPrerequisiteArtifacts("deps", Mode.TARGET);
    ImmutableList<Artifact> protoFiles = prerequisiteArtifacts.filter(FileType.of(".proto")).list();
    if (!protoFiles.isEmpty()) {
      ruleContext.attributeWarning("deps", FILES_DEPRECATED_WARNING);
    }
    return protoFiles;
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
      String uniqueDirectoryName, Iterable<Artifact> protos, FileType newFileType,
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
          proto.getRootRelativePath().getParentDirectory(),
          new PathFragment(protoOutputName));
      @Nullable PathFragment outputFile = FileSystemUtils.replaceExtension(
          rawFragment,
          newFileType.getExtensions().get(0),
          ".proto");
      if (outputFile != null) {
        builder.add(ruleContext.getUniqueDirectoryArtifact(uniqueDirectoryName,
            outputFile, ruleContext.getBinOrGenfilesDirectory()));
      }
    }
    return builder.build();
  }
}
