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

import static com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileType;

/**
 * Rule definition for objc_proto_library.
 *
 * This is a temporary rule until it is better known how to support proto_library rules.
 */
public class ObjcProtoLibraryRule implements RuleDefinition {
  static final String OPTIONS_FILE_ATTR = "options_file";
  static final String OUTPUT_CPP_ATTR = "output_cpp";
  static final String USE_OBJC_HEADER_NAMES_ATTR = "use_objc_header_names";
  static final String PER_PROTO_INCLUDES_ATTR = "per_proto_includes";
  static final String PORTABLE_PROTO_FILTERS_ATTR = "portable_proto_filters";

  static final String PROTO_COMPILER_ATTR = "$googlemac_proto_compiler";
  static final String PROTO_COMPILER_SUPPORT_ATTR = "$googlemac_proto_compiler_support";
  static final String PROTO_LIB_ATTR = "$lib_protobuf";
  static final String XCODE_GEN_ATTR = "$xcodegen";

  @Override
  public RuleClass build(Builder builder, final RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(ObjcConfiguration.class, AppleConfiguration.class)
        /* <!-- #BLAZE_RULE(objc_proto_library).ATTRIBUTE(deps) -->
        The directly depended upon proto_library rules.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .override(
            attr("deps", LABEL_LIST)
                // Support for files in deps is for backwards compatibility.
                .allowedRuleClasses("proto_library", "filegroup")
                .legacyAllowAnyFileType())
        /* <!-- #BLAZE_RULE(objc_proto_library).ATTRIBUTE(options_file) -->
        Optional options file to apply to protos which affects compilation (e.g. class
        whitelist/blacklist settings).
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr(OPTIONS_FILE_ATTR, LABEL).legacyAllowAnyFileType().singleArtifact().cfg(HOST))
        /* <!-- #BLAZE_RULE(objc_proto_library).ATTRIBUTE(output_cpp) -->
        If true, output C++ rather than ObjC.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr(OUTPUT_CPP_ATTR, BOOLEAN).value(false))
        /* <!-- #BLAZE_RULE(objc_proto_library).ATTRIBUTE(use_objc_header_names) -->
        If true, output headers with .pbobjc.h, rather than .pb.h.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr(USE_OBJC_HEADER_NAMES_ATTR, BOOLEAN).value(false))
        /* <!-- #BLAZE_RULE(objc_proto_library).ATTRIBUTE(per_proto_includes) -->
        If true, always add all directories to objc_library includes.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr(PER_PROTO_INCLUDES_ATTR, BOOLEAN).value(false))
        /* <!-- #BLAZE_RULE(objc_proto_library).ATTRIBUTE(portable_proto_filters) -->
        List of portable proto filters to be passed on to the protobuf compiler. This attribute
        cannot be used together with the options_file, output_cpp, per_proto_includes and
        use_objc_header_names attributes.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr(PORTABLE_PROTO_FILTERS_ATTR, LABEL_LIST)
                .legacyAllowAnyFileType()
                .allowedRuleClasses("filegroup")
                .cfg(HOST))
        .add(
            attr(PROTO_COMPILER_ATTR, LABEL)
                .allowedFileTypes(FileType.of(".py"))
                .cfg(HOST)
                .singleArtifact()
                .value(
                    new ComputedDefault(PORTABLE_PROTO_FILTERS_ATTR) {
                      @Override
                      public Object getDefault(AttributeMap rule) {
                        return rule.isAttributeValueExplicitlySpecified(PORTABLE_PROTO_FILTERS_ATTR)
                            ? env.getToolsLabel("//tools/objc:protobuf_compiler")
                            : env.getToolsLabel("//tools/objc:compile_protos");
                      }
                    }))
        .add(
            attr(PROTO_COMPILER_SUPPORT_ATTR, LABEL)
                .legacyAllowAnyFileType()
                .cfg(HOST)
                .value(
                    new ComputedDefault(PORTABLE_PROTO_FILTERS_ATTR) {
                      @Override
                      public Object getDefault(AttributeMap rule) {
                        return rule.isAttributeValueExplicitlySpecified(PORTABLE_PROTO_FILTERS_ATTR)
                            ? env.getToolsLabel("//tools/objc:protobuf_compiler_support")
                            : env.getToolsLabel("//tools/objc:proto_support");
                      }
                    }))
        .add(
            attr(PROTO_LIB_ATTR, LABEL)
                .allowedRuleClasses("objc_library")
                .value(
                    new ComputedDefault(PORTABLE_PROTO_FILTERS_ATTR, OUTPUT_CPP_ATTR) {
                      @Override
                      public Object getDefault(AttributeMap rule) {
                        if (rule.isAttributeValueExplicitlySpecified(PORTABLE_PROTO_FILTERS_ATTR)) {
                          return env.getLabel("//external:objc_protobuf_lib");
                        } else {
                          return rule.get(OUTPUT_CPP_ATTR, Type.BOOLEAN)
                              ? env.getLabel("//external:objc_proto_cpp_lib")
                              : env.getLabel("//external:objc_proto_lib");
                        }
                      }
                    }))
        .add(
            attr(XCODE_GEN_ATTR, LABEL)
                .cfg(HOST)
                .exec()
                .value(env.getToolsLabel("//tools/objc:xcodegen")))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("objc_proto_library")
        .factoryClass(ObjcProtoLibrary.class)
        .ancestors(BaseRuleClasses.RuleBase.class, ObjcRuleClasses.XcrunRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = objc_proto_library, TYPE = LIBRARY, FAMILY = Objective-C) -->

<p>This rule produces a static library from the given proto_library dependencies, after applying an
options file.</p>

<!-- #END_BLAZE_RULE -->*/
