// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.StarlarkNativeAspect;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.proto.ProtoInfo;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;

/**
 * Aspect that gathers the proto dependencies of the attached rule target, and propagates the proto
 * values of its dependencies through the ObjcProtoProvider.
 */
public class ObjcProtoAspect extends StarlarkNativeAspect implements ConfiguredAspectFactory {
  public static final String NAME = "ObjcProtoAspect";

  @Override
  public AspectDefinition getDefinition(AspectParameters aspectParameters) {
    return new AspectDefinition.Builder(this)
        .propagateAlongAttribute("deps")
        .build();
  }

  @Override
  public ConfiguredAspect create(
      ConfiguredTargetAndData ctadBase,
      RuleContext ruleContext,
      AspectParameters parameters,
      String toolsRepository)
      throws InterruptedException, ActionConflictException {
    ConfiguredAspect.Builder aspectBuilder = new ConfiguredAspect.Builder(ruleContext);

    ObjcProtoProvider.Builder aspectObjcProtoProvider = new ObjcProtoProvider.Builder();

    if (ruleContext.attributes().has("deps", BuildType.LABEL_LIST)) {
      Iterable<ObjcProtoProvider> depObjcProtoProviders =
          ruleContext.getPrerequisites(
              "deps", TransitionMode.TARGET, ObjcProtoProvider.STARLARK_CONSTRUCTOR);
      aspectObjcProtoProvider.addTransitive(depObjcProtoProviders);
    }

    ProtoAttributes attributes = new ProtoAttributes(ruleContext);

    // If this current target is an objc_proto_library, only then read the protos and portable
    // filters it has configured in the attributes and add them to the ObjcProtoProvider.Builder
    // instance. Validation for the correct target attributes is already done in ProtoSupport.java.
    if (attributes.isObjcProtoLibrary()) {

      // Gather up all the dependency protos depended by this target.
      List<ProtoInfo> protoInfos =
          ruleContext.getPrerequisites("deps", TransitionMode.TARGET, ProtoInfo.PROVIDER);

      for (ProtoInfo protoInfo : protoInfos) {
        aspectObjcProtoProvider.addProtoFiles(protoInfo.getTransitiveProtoSources());
      }

      NestedSet<Artifact> portableProtoFilters =
          PrerequisiteArtifacts.nestedSet(
              ruleContext, ProtoAttributes.PORTABLE_PROTO_FILTERS_ATTR, TransitionMode.HOST);

      // If this target does not provide filters but specifies direct proto_library dependencies,
      // generate a filter file only for those proto files.
      if (portableProtoFilters.isEmpty() && !protoInfos.isEmpty()) {
        Artifact generatedFilter =
            ProtobufSupport.getGeneratedPortableFilter(ruleContext, ruleContext.getConfiguration());
        ProtobufSupport.registerPortableFilterGenerationAction(
            ruleContext, generatedFilter, protoInfos);
        portableProtoFilters = NestedSetBuilder.create(Order.STABLE_ORDER, generatedFilter);
      }

      aspectObjcProtoProvider.addPortableProtoFilters(portableProtoFilters);

      // Propagate protobuf's headers and search paths so the BinaryLinkingTargetFactory subclasses
      // (i.e. objc_binary) don't have to depend on it.
      ObjcConfiguration objcConfiguration =
          ruleContext.getConfiguration().getFragment(ObjcConfiguration.class);
      CcCompilationContext protobufCcCompilationContext;
      if (objcConfiguration.compileInfoMigration()) {
        CcInfo protobufCcInfo =
            ruleContext.getPrerequisite(
                ObjcRuleClasses.PROTO_LIB_ATTR, TransitionMode.TARGET, CcInfo.PROVIDER);
        protobufCcCompilationContext = protobufCcInfo.getCcCompilationContext();
      } else {
        ObjcProvider protobufObjcProvider =
            ruleContext.getPrerequisite(
                ObjcRuleClasses.PROTO_LIB_ATTR,
                TransitionMode.TARGET,
                ObjcProvider.STARLARK_CONSTRUCTOR);
        protobufCcCompilationContext = protobufObjcProvider.getCcCompilationContext();
      }
      aspectObjcProtoProvider.addProtobufHeaders(
          protobufCcCompilationContext.getDeclaredIncludeSrcs());
      aspectObjcProtoProvider.addProtobufHeaderSearchPaths(
          NestedSetBuilder.<PathFragment>linkOrder()
              .addAll(protobufCcCompilationContext.getIncludeDirs())
              .build());
    }

    // Only add the provider if it has any values, otherwise skip it.
    if (!aspectObjcProtoProvider.isEmpty()) {
      aspectBuilder.addNativeDeclaredProvider(aspectObjcProtoProvider.build());
    }
    return aspectBuilder.build();
  }

  @Override
  public String getName() {
    return NAME;
  }
}
