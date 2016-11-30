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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.objc.J2ObjcSource.SourceType;
import com.google.devtools.build.lib.rules.proto.ProtoCommon;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder;
import com.google.devtools.build.lib.rules.proto.ProtoConfiguration;
import com.google.devtools.build.lib.rules.proto.ProtoSourceFileBlacklist;
import com.google.devtools.build.lib.rules.proto.ProtoSourcesProvider;
import com.google.devtools.build.lib.rules.proto.ProtoSupportDataProvider;
import com.google.devtools.build.lib.rules.proto.SupportData;
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
public abstract class AbstractJ2ObjcProtoAspect extends NativeAspectClass
  implements ConfiguredAspectFactory {

  private static final Iterable<Attribute> DEPENDENT_ATTRIBUTES = ImmutableList.of(
      new Attribute("$protobuf_lib", Mode.TARGET),
      new Attribute("deps", Mode.TARGET));

  private static final String PROTO_SOURCE_FILE_BLACKLIST_ATTR = "$j2objc_proto_blacklist";

  /** Flags passed to J2ObjC proto compiler plugin. */
  protected static final Iterable<String> J2OBJC_PLUGIN_PARAMS =
      ImmutableList.of("file_dir_mapping", "generate_class_mappings");

  protected final String toolsRepository;

  public AbstractJ2ObjcProtoAspect(String toolsRepository) {
    this.toolsRepository = toolsRepository;
  }

  @Override
  public AspectDefinition getDefinition(AspectParameters aspectParameters) {
    AspectDefinition.Builder builder = new AspectDefinition.Builder(this)
        .requireProviders(ProtoSourcesProvider.class)
        .requiresConfigurationFragments(
            AppleConfiguration.class,
            J2ObjcConfiguration.class,
            ObjcConfiguration.class,
            ProtoConfiguration.class)
        .attributeAspect("deps", this)
        .attributeAspect("exports", this)
        .attributeAspect("runtime_deps", this)
        .add(attr("$protobuf_lib", LABEL)
            .value(Label.parseAbsoluteUnchecked("//third_party/java/j2objc:proto_runtime")))
        .add(attr("$xcrunwrapper", LABEL).cfg(HOST).exec()
            .value(Label.parseAbsoluteUnchecked(
                toolsRepository + "//tools/objc:xcrunwrapper")))
        .add(attr(ObjcRuleClasses.LIBTOOL_ATTRIBUTE, LABEL).cfg(HOST).exec()
              .value(Label.parseAbsoluteUnchecked(
                toolsRepository + "//tools/objc:libtool")))
        .add(attr(":xcode_config", LABEL)
            .allowedRuleClasses("xcode_config")
            .checkConstraints()
            .direct_compile_time_input()
            .cfg(HOST)
            .value(new AppleToolchain.XcodeConfigLabel(toolsRepository)))
        .add(ProtoSourceFileBlacklist.blacklistFilegroupAttribute(
            PROTO_SOURCE_FILE_BLACKLIST_ATTR,
            ImmutableList.of(Label.parseAbsoluteUnchecked(
                toolsRepository + "//tools/j2objc:j2objc_proto_blacklist"))));
    return addAdditionalAttributes(builder).build();
  }

  protected abstract AspectDefinition.Builder addAdditionalAttributes(
      AspectDefinition.Builder builder);

  protected abstract boolean checkShouldCreateAspect(RuleContext ruleContext);

  protected abstract boolean allowServices(RuleContext ruleContext);

  @Override
  public ConfiguredAspect create(
      ConfiguredTarget base, RuleContext ruleContext, AspectParameters parameters)
      throws InterruptedException {
    if (!checkShouldCreateAspect(ruleContext)) {
      return new ConfiguredAspect.Builder(this, parameters, ruleContext).build();
    }

    ProtoSourcesProvider protoSourcesProvider = base.getProvider(ProtoSourcesProvider.class);
    ImmutableList<Artifact> protoSources = protoSourcesProvider.getDirectProtoSources();

    // Avoid pulling in any generated files from blacklisted protos.
    ProtoSourceFileBlacklist protoBlacklist =
        new ProtoSourceFileBlacklist(
            ruleContext,
            ruleContext
                .getPrerequisiteArtifacts(PROTO_SOURCE_FILE_BLACKLIST_ATTR, Mode.HOST)
                .list());
    ImmutableList<Artifact> filteredProtoSources = ImmutableList.copyOf(
        protoBlacklist.filter(protoSources));

    XcodeProvider xcodeProvider;
    Iterable<Artifact> headerMappingFiles;
    Iterable<Artifact> classMappingFiles;
    ObjcCommon common;

    if (filteredProtoSources.isEmpty()) {
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
      J2ObjcSource j2ObjcSource = j2ObjcSource(ruleContext, filteredProtoSources);
      headerMappingFiles = headerMappingFiles(ruleContext, filteredProtoSources);
      classMappingFiles = classMappingFiles(ruleContext, filteredProtoSources);

      createActions(base, ruleContext, headerMappingFiles, classMappingFiles, j2ObjcSource);
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
      
      try {
        new LegacyCompilationSupport(ruleContext)
            .registerCompileAndArchiveActions(common)
            .registerFullyLinkAction(common.getObjcProvider(),
                ruleContext.getImplicitOutputArtifact(CompilationSupport.FULLY_LINKED_LIB));
      } catch (RuleErrorException e) {
        ruleContext.ruleError(e.getMessage());
      }
    }

    NestedSet<Artifact> j2ObjcTransitiveHeaderMappingFiles = j2ObjcTransitiveHeaderMappingFiles(
        ruleContext, headerMappingFiles);
    NestedSet<Artifact> j2ObjcTransitiveClassMappingFiles = j2ObjcTransitiveClassMappingFiles(
        ruleContext, classMappingFiles);

    return new ConfiguredAspect.Builder(this, parameters, ruleContext)
        .addProviders(
            new J2ObjcMappingFileProvider(
                j2ObjcTransitiveHeaderMappingFiles,
                j2ObjcTransitiveClassMappingFiles,
                NestedSetBuilder.<Artifact>stableOrder().build(),
                NestedSetBuilder.<Artifact>stableOrder().build()),
            common.getObjcProvider(),
            xcodeProvider)
        .build();
  }

  private void createActions(ConfiguredTarget base, RuleContext ruleContext,
      Iterable<Artifact> headerMappingFiles, Iterable<Artifact> classMappingFiles,
      J2ObjcSource j2ObjcSource) {
    SupportData supportData = base.getProvider(ProtoSupportDataProvider.class).getSupportData();
    ImmutableList<Artifact> outputs = ImmutableList.<Artifact>builder()
        .addAll(j2ObjcSource.getObjcSrcs())
        .addAll(j2ObjcSource.getObjcHdrs())
        .addAll(headerMappingFiles)
        .addAll(classMappingFiles)
        .build();
    String langPluginParameter = String.format(
        "%s:%s",
        Joiner.on(',').join(J2OBJC_PLUGIN_PARAMS),
        ruleContext.getConfiguration().getGenfilesFragment().getPathString());
    ProtoCompileActionBuilder actionBuilder =
        new ProtoCompileActionBuilder(ruleContext, supportData, "J2ObjC", "j2objc", outputs)
            .setLangPluginName("$j2objc_plugin")
            .setLangPluginParameter(langPluginParameter)
            .allowServices(allowServices(ruleContext));
    ruleContext.registerAction(actionBuilder.build());
  }

  private J2ObjcSource j2ObjcSource(RuleContext ruleContext, ImmutableList<Artifact> protoSources) {
    PathFragment objcFileRootExecPath = ruleContext.getConfiguration()
        .getGenfilesDirectory(ruleContext.getRule().getRepository())
        .getExecPath();
    Iterable<PathFragment> headerSearchPaths = J2ObjcLibrary.j2objcSourceHeaderSearchPaths(
        ruleContext, objcFileRootExecPath, protoSources);

    return new J2ObjcSource(
        ruleContext.getTarget().getLabel(),
        ProtoCommon.getGeneratedOutputs(ruleContext, protoSources, ".j2objc.pb.m"),
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
