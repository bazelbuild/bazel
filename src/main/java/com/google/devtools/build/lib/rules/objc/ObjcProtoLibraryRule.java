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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.PROTOBUF_WELL_KNOWN_TYPES;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.PROTO_COMPILER_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.PROTO_COMPILER_SUPPORT_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.PROTO_LIB_ATTR;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.proto.ProtoSourceFileBlacklist;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;

/**
 * Rule definition for objc_proto_library.
 *
 * <p>This is a temporary rule until it is better known how to support proto_library rules.
 */
public class ObjcProtoLibraryRule implements RuleDefinition {
  static final String PORTABLE_PROTO_FILTERS_ATTR = "portable_proto_filters";

  private final ObjcProtoAspect objcProtoAspect;

  /**
   * Returns a newly built rule definition for objc_proto_library.
   *
   * @param objcProtoAspect Aspect that traverses the dependency graph through the deps attribute to
   *     gather all proto files and portable filters depended by objc_proto_library targets.
   */
  public ObjcProtoLibraryRule(ObjcProtoAspect objcProtoAspect) {
    this.objcProtoAspect = objcProtoAspect;
  }

  @Override
  public RuleClass build(RuleClass.Builder builder, final RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(
            CppConfiguration.class, ObjcConfiguration.class, AppleConfiguration.class)
        /* <!-- #BLAZE_RULE(objc_proto_library).ATTRIBUTE(deps) -->
        The directly depended upon proto_library and objc_proto_library rules.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .override(
            attr("deps", LABEL_LIST)
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .allowedRuleClasses("proto_library", "objc_proto_library")
                .aspect(objcProtoAspect))
        /* <!-- #BLAZE_RULE(objc_proto_library).ATTRIBUTE(portable_proto_filters) -->
        List of portable proto filters to be passed on to the protobuf compiler. If no filter files
        are passed, one will be generated that whitelists every proto file listed in the
        proto_library dependencies (i.e. proto files depended through other objc_proto_library
        won't be automatically whitelisted).
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr(PORTABLE_PROTO_FILTERS_ATTR, LABEL_LIST)
                .legacyAllowAnyFileType()
                .allowedRuleClasses("filegroup")
                .cfg(HostTransition.createFactory()))
        .add(
            attr(PROTO_COMPILER_ATTR, LABEL)
                .allowedFileTypes(FileType.of(".py"), FileType.of(".sh"))
                .cfg(HostTransition.createFactory())
                .singleArtifact()
                .value(env.getToolsLabel("//tools/objc:protobuf_compiler_wrapper")))
        .add(
            attr(PROTO_COMPILER_SUPPORT_ATTR, LABEL)
                .legacyAllowAnyFileType()
                .cfg(HostTransition.createFactory())
                .value(env.getToolsLabel("//tools/objc:protobuf_compiler_support")))
        .add(
            attr(PROTO_LIB_ATTR, LABEL)
                .allowedRuleClasses("objc_library")
                .value(env.getToolsLabel("//tools/objc:protobuf_lib")))
        .add(
            ProtoSourceFileBlacklist.blacklistFilegroupAttribute(
                PROTOBUF_WELL_KNOWN_TYPES,
                ImmutableList.of(env.getToolsLabel("//tools/objc:protobuf_well_known_types"))))
        .cfg(AppleCrosstoolTransition.APPLE_CROSSTOOL_TRANSITION)
        .addRequiredToolchains(CppRuleClasses.ccToolchainTypeAttribute(env))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("objc_proto_library")
        .factoryClass(ObjcProtoLibrary.class)
        .ancestors(
            BaseRuleClasses.RuleBase.class,
            ObjcRuleClasses.LibtoolRule.class,
            ObjcRuleClasses.XcrunRule.class,
            ObjcRuleClasses.CrosstoolRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = objc_proto_library, TYPE = LIBRARY, FAMILY = Objective-C) -->

<p>This rule generates ObjC headers for the proto files given through the proto_library and
objc_proto_library dependencies.</p>

<!-- #END_BLAZE_RULE -->*/
