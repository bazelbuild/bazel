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

import static com.google.devtools.build.lib.rules.objc.J2ObjcSource.SourceType;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.Aspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.rules.proto.ProtoCommon;
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
public abstract class AbstractJ2ObjcProtoAspect implements ConfiguredAspectFactory {
  public static final String NAME = "J2ObjcProtoAspect";

  public AspectDefinition getDefinition() {
    AspectDefinition.Builder builder = new AspectDefinition.Builder("J2ObjcProtoAspect")
        .requireProvider(ProtoSourcesProvider.class)
        .attributeAspect("deps", getClass())
        .attributeAspect("exports", getClass())
        .attributeAspect("runtime_deps", getClass());
    return addAdditionalAttributes(builder).build();
  }

  protected abstract AspectDefinition.Builder addAdditionalAttributes(
      AspectDefinition.Builder builder);

  protected static Label parseLabel(String from) {
    try {
      return Label.parseAbsolute(from);
    } catch (LabelSyntaxException e) {
      throw new IllegalArgumentException(from);
    }
  }

  protected abstract boolean checkShouldCreateAspect(RuleContext ruleContext);

  @Override
  public Aspect create(ConfiguredTarget base, RuleContext ruleContext,
      AspectParameters parameters) {
    if (!checkShouldCreateAspect(ruleContext)) {
      return new Aspect.Builder(NAME).build();
    }

    ProtoSourcesProvider protoSourcesProvider = base.getProvider(ProtoSourcesProvider.class);
    ImmutableList<Artifact> protoSources = protoSourcesProvider.getProtoSources();
    NestedSet<Artifact> transitiveImports = protoSourcesProvider.getTransitiveImports();

    J2ObjcSrcsProvider.Builder srcsBuilder = new J2ObjcSrcsProvider.Builder();
    Iterable<Artifact> headerMappingFiles;
    Iterable<Artifact> classMappingFiles;

    if (protoSources.isEmpty()) {
      headerMappingFiles = ImmutableList.of();
      classMappingFiles = ImmutableList.of();
    } else {
      J2ObjcSource j2ObjcSource = j2ObjcSource(ruleContext, protoSources);
      headerMappingFiles = headerMappingFiles(ruleContext, protoSources);
      classMappingFiles = classMappingFiles(ruleContext, protoSources);
      srcsBuilder.addSource(j2ObjcSource);

      createActions(base, ruleContext, protoSources, transitiveImports,
          headerMappingFiles, classMappingFiles, j2ObjcSource);
    }

    J2ObjcSrcsProvider j2objcSrcsProvider = srcsBuilder
        .addTransitiveJ2ObjcSrcs(ruleContext, "deps")
        .build();
    NestedSet<Artifact> j2ObjcTransitiveHeaderMappingFiles = j2ObjcTransitiveHeaderMappingFiles(
        ruleContext, headerMappingFiles);
    NestedSet<Artifact> j2ObjcTransitiveClassMappingFiles = j2ObjcTransitiveClassMappingFiles(
        ruleContext, classMappingFiles);

    return new Aspect.Builder(NAME)
        .addProvider(J2ObjcSrcsProvider.class, j2objcSrcsProvider)
        .addProvider(
            J2ObjcMappingFileProvider.class,
            new J2ObjcMappingFileProvider(
                j2ObjcTransitiveHeaderMappingFiles,
                j2ObjcTransitiveClassMappingFiles,
                NestedSetBuilder.<Artifact>stableOrder().build()))
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
