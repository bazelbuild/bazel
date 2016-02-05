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

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.rules.objc.J2ObjcSource.SourceType;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredNativeAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.proto.ProtoCommon;
import com.google.devtools.build.lib.rules.proto.ProtoConfiguration;
import com.google.devtools.build.lib.rules.proto.ProtoSourcesProvider;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * J2ObjC aspect for the proto_library rule.
 *
 * <p>J2ObjC proto is different from other proto languages in that it needs to implement both the
 * Java API and the Objective-C API. The Java API implementation is needed for compatibility with
 * ObjC code translated by J2ObjC from Java. The Objective-C API is also required for compatibility
 * with hand-written ObjC code. As an example, for an accessor of a int field called int_value in
 * message Foo, both the Java API version (getIntValue) and the ObjC version (intValue) are needed.
 *
 * <p>On the whole, the dependency chain looks like: objc_binary -> objc_library -> j2objc_library
 * -> java_library -> proto_library. The jars containing compiled Java protos provided by
 * proto_library are needed in the classpath for J2ObjC transpilation in java_library, but they
 * themselves are not transpiled or exported to objc_* rules. Instead, the J2ObjC protos generated
 * by this class and provided by proto_library will be exported all the way to objc_binary for ObjC
 * compilation and linking into the final application bundle.
 */
public abstract class AbstractJ2ObjcProtoAspect implements ConfiguredNativeAspectFactory {
  public static final String NAME = "J2ObjcProtoAspect";
  private static final Iterable<Attribute> DEPENDENT_ATTRIBUTES = ImmutableList.of(
      new Attribute("$protobuf_lib", Mode.TARGET),
      new Attribute("deps", Mode.TARGET));

  @Override
  public AspectDefinition getDefinition(AspectParameters aspectParameters) {
    AspectDefinition.Builder builder = new AspectDefinition.Builder("J2ObjcProtoAspect")
        .requireProvider(ProtoSourcesProvider.class)
        .requiresConfigurationFragments(
            AppleConfiguration.class,
            J2ObjcConfiguration.class,
            ObjcConfiguration.class,
            ProtoConfiguration.class)
        .attributeAspect("deps", getClass())
        .attributeAspect("exports", getClass())
        .attributeAspect("runtime_deps", getClass())
        .add(attr("$protobuf_lib", LABEL)
            .value(Label.parseAbsoluteUnchecked(
                Constants.TOOLS_REPOSITORY + "//third_party/java/j2objc:proto_runtime")))
        .add(attr("$xcrunwrapper", LABEL).cfg(HOST).exec()
            .value(Label.parseAbsoluteUnchecked(
                Constants.TOOLS_REPOSITORY + "//tools/objc:xcrunwrapper")))
        .add(attr(":xcode_config", LABEL)
            .allowedRuleClasses("xcode_config")
            .checkConstraints()
            .direct_compile_time_input()
            .cfg(HOST)
            .value(AppleToolchain.RequiresXcodeConfigRule.XCODE_CONFIG_LABEL));
    return addAdditionalAttributes(builder).build();
  }

  protected abstract AspectDefinition.Builder addAdditionalAttributes(
      AspectDefinition.Builder builder);

  protected abstract boolean checkShouldCreateAspect(RuleContext ruleContext);

  @Override
  public ConfiguredAspect create(
      ConfiguredTarget base, RuleContext ruleContext, AspectParameters parameters)
      throws InterruptedException {
    if (!checkShouldCreateAspect(ruleContext)) {
      return new ConfiguredAspect.Builder(NAME, ruleContext).build();
    }

    ProtoSourcesProvider protoSourcesProvider = base.getProvider(ProtoSourcesProvider.class);
    ImmutableList<Artifact> protoSources = protoSourcesProvider.getDirectProtoSources();
    NestedSet<Artifact> transitiveImports = protoSourcesProvider.getTransitiveImports();

    XcodeProvider xcodeProvider;
    Iterable<Artifact> headerMappingFiles;
    Iterable<Artifact> classMappingFiles;
    ObjcCommon common;

    if (protoSources.isEmpty()) {
      headerMappingFiles = ImmutableList.of();
      classMappingFiles = ImmutableList.of();
      common = J2ObjcAspect.common(
          ruleContext,
          ImmutableList.<Artifact>of(),
          ImmutableList.<Artifact>of(),
          ImmutableList.<PathFragment>of(),
          DEPENDENT_ATTRIBUTES);
      xcodeProvider = J2ObjcAspect.xcodeProvider(
          ruleContext,
          common,
          ImmutableList.<Artifact>of(),
          ImmutableList.<PathFragment>of(),
          DEPENDENT_ATTRIBUTES);
    } else {
      J2ObjcSource j2ObjcSource = j2ObjcSource(ruleContext, protoSources);
      headerMappingFiles = headerMappingFiles(ruleContext, protoSources);
      classMappingFiles = classMappingFiles(ruleContext, protoSources);

      createActions(base, ruleContext, protoSources, transitiveImports,
          headerMappingFiles, classMappingFiles, j2ObjcSource);
      common = J2ObjcAspect.common(
          ruleContext,
          j2ObjcSource.getObjcSrcs(),
          j2ObjcSource.getObjcHdrs(),
          j2ObjcSource.getHeaderSearchPaths(),
          DEPENDENT_ATTRIBUTES);
      xcodeProvider = J2ObjcAspect.xcodeProvider(
          ruleContext,
          common,
          j2ObjcSource.getObjcHdrs(),
          j2ObjcSource.getHeaderSearchPaths(),
          DEPENDENT_ATTRIBUTES);

      new CompilationSupport(ruleContext).registerCompileAndArchiveActions(common);
    }

    NestedSet<Artifact> j2ObjcTransitiveHeaderMappingFiles = j2ObjcTransitiveHeaderMappingFiles(
        ruleContext, headerMappingFiles);
    NestedSet<Artifact> j2ObjcTransitiveClassMappingFiles = j2ObjcTransitiveClassMappingFiles(
        ruleContext, classMappingFiles);

    return new ConfiguredAspect.Builder(NAME, ruleContext)
        .addProvider(
            J2ObjcMappingFileProvider.class,
            new J2ObjcMappingFileProvider(
                j2ObjcTransitiveHeaderMappingFiles,
                j2ObjcTransitiveClassMappingFiles,
                NestedSetBuilder.<Artifact>stableOrder().build(),
                NestedSetBuilder.<Artifact>stableOrder().build()))
        .addProvider(ObjcProvider.class, common.getObjcProvider())
        .addProvider(XcodeProvider.class, xcodeProvider)
        .build();
  }

  protected abstract void createActions(ConfiguredTarget base, RuleContext ruleContext,
      Iterable<Artifact> protoSources, NestedSet<Artifact> transitiveProtoSources,
      Iterable<Artifact> headerMappingFiles, Iterable<Artifact> classMappingFiles,
      J2ObjcSource j2ObjcSource);

  protected abstract boolean checkShouldCreateSources(RuleContext ruleContext);

  private J2ObjcSource j2ObjcSource(RuleContext ruleContext,
      ImmutableList<Artifact> protoSources) {
    Iterable<Artifact> generatedSourceFiles = checkShouldCreateSources(ruleContext)
        ? ProtoCommon.getGeneratedOutputs(ruleContext, protoSources, ".j2objc.pb.m")
        : ImmutableList.<Artifact>of();
    PathFragment objcFileRootExecPath = ruleContext.getConfiguration().getGenfilesDirectory()
        .getExecPath();
    Iterable<PathFragment> headerSearchPaths = J2ObjcLibrary.j2objcSourceHeaderSearchPaths(
        ruleContext, objcFileRootExecPath, protoSources);

    return new J2ObjcSource(
        ruleContext.getTarget().getLabel(),
        generatedSourceFiles,
        ProtoCommon.getGeneratedOutputs(ruleContext, protoSources, ".j2objc.pb.h"),
        objcFileRootExecPath,
        SourceType.PROTO,
        headerSearchPaths);
  }

  private static Iterable<Artifact> headerMappingFiles(RuleContext ruleContext,
      ImmutableList<Artifact> protoSources) {
    return ProtoCommon.getGeneratedOutputs(ruleContext, protoSources, ".j2objc.mapping");
  }

  private static Iterable<Artifact> classMappingFiles(RuleContext ruleContext,
      ImmutableList<Artifact> protoSources) {
    return ProtoCommon.getGeneratedOutputs(ruleContext, protoSources, ".clsmap.properties");
  }

  private static NestedSet<Artifact> j2ObjcTransitiveHeaderMappingFiles(RuleContext ruleContext,
      Iterable<Artifact> headerMappingFiles) {
    NestedSetBuilder<Artifact> mappingFileBuilder = NestedSetBuilder.stableOrder();
    mappingFileBuilder.addAll(headerMappingFiles);

    for (J2ObjcMappingFileProvider provider :
        ruleContext.getPrerequisites("deps", Mode.TARGET, J2ObjcMappingFileProvider.class)) {
      mappingFileBuilder.addTransitive(provider.getHeaderMappingFiles());
    }

    return mappingFileBuilder.build();
  }

  private static NestedSet<Artifact> j2ObjcTransitiveClassMappingFiles(RuleContext ruleContext,
      Iterable<Artifact> classMappingFiles) {
    NestedSetBuilder<Artifact> mappingFileBuilder = NestedSetBuilder.stableOrder();
    mappingFileBuilder.addAll(classMappingFiles);

    for (J2ObjcMappingFileProvider provider :
        ruleContext.getPrerequisites("deps", Mode.TARGET, J2ObjcMappingFileProvider.class)) {
      mappingFileBuilder.addTransitive(provider.getClassMappingFiles());
    }

    return mappingFileBuilder.build();
  }
}
