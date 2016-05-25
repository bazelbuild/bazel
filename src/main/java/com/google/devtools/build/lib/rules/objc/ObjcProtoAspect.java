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

import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.rules.proto.ProtoSourcesProvider;

/**
 * Aspect that gathers the proto dependencies of the attached rule target, and propagates the proto
 * values of its dependencies through the ObjcProtoProvider.
 */
public class ObjcProtoAspect extends NativeAspectClass implements ConfiguredAspectFactory {
  public static final String NAME = "ObjcProtoAspect";

  @Override
  public AspectDefinition getDefinition(AspectParameters aspectParameters) {
    return new AspectDefinition.Builder(NAME)
        .attributeAspect("deps", this)
        .requiresConfigurationFragments(ObjcConfiguration.class)
        .build();
  }

  @Override
  public ConfiguredAspect create(
      ConfiguredTarget base, RuleContext ruleContext, AspectParameters parameters)
      throws InterruptedException {
    ConfiguredAspect.Builder aspectBuilder = new ConfiguredAspect.Builder(NAME, ruleContext);
    ObjcConfiguration objcConfiguration = ruleContext.getFragment(ObjcConfiguration.class);

    if (!objcConfiguration.experimentalAutoTopLevelUnionObjCProtos()) {
      // Only process the aspect if the experimental flag is set.
      return aspectBuilder.build();
    }

    ObjcProtoProvider.Builder aspectObjcProtoProvider = new ObjcProtoProvider.Builder();

    if (ruleContext.attributes().has("deps", BuildType.LABEL_LIST)) {
      Iterable<ObjcProtoProvider> depObjcProtoProviders =
          ruleContext.getPrerequisites("deps", Mode.TARGET, ObjcProtoProvider.class);
      aspectObjcProtoProvider.addTransitive(depObjcProtoProviders);
    }

    // If the rule has the portable_proto_filters, it must be an objc_proto_library configured
    // to use the third party protobuf library, in contrast with the PB2 internal library. Only
    // the third party library is enabled to propagate the protos with this aspect.
    // Validation for the correct target attributes is done in ProtoSupport.java.
    if (ruleContext
        .attributes()
        .isAttributeValueExplicitlySpecified(ObjcProtoLibraryRule.PORTABLE_PROTO_FILTERS_ATTR)) {
      aspectObjcProtoProvider.addPortableProtoFilters(
          PrerequisiteArtifacts.nestedSet(
              ruleContext, ObjcProtoLibraryRule.PORTABLE_PROTO_FILTERS_ATTR, Mode.HOST));

      // Gather up all the dependency protos depended by this target.
      Iterable<ProtoSourcesProvider> protoProviders =
          ruleContext.getPrerequisites("deps", Mode.TARGET, ProtoSourcesProvider.class);

      for (ProtoSourcesProvider protoProvider : protoProviders) {
        aspectObjcProtoProvider.addProtoSources(protoProvider.getTransitiveProtoSources());
      }
    }

    // Only add the provider if it has any values, otherwise skip it.
    if (!aspectObjcProtoProvider.isEmpty()) {
      aspectBuilder.addProvider(ObjcProtoProvider.class, aspectObjcProtoProvider.build());
    }
    return aspectBuilder.build();
  }
}
