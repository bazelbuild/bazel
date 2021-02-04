// Copyright 2015 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.CharMatcher;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.proto.ProtoInfo;
import java.util.ArrayList;

/** Common rule attributes used by an objc_proto_library. */
final class ProtoAttributes {

  /**
   * List of file name segments that should be upper cased when being generated. More information
   * available in the generateProtobufFilename() method.
   */
  private static final ImmutableSet<String> UPPERCASE_SEGMENTS =
      ImmutableSet.of("url", "http", "https");

  @VisibleForTesting
  static final String PORTABLE_PROTO_FILTERS_EMPTY_ERROR =
      "The portable_proto_filters attribute can't be empty";

  @VisibleForTesting
  static final String NO_PROTOS_ERROR =
      "no protos to compile - a non-empty deps attribute is required";

  static final String PORTABLE_PROTO_FILTERS_ATTR = "portable_proto_filters";

  private final RuleContext ruleContext;

  /**
   * Creates a new ProtoAttributes object that wraps over objc_proto_library's attributes.
   *
   * @param ruleContext context of the objc_proto_library to wrap
   */
  ProtoAttributes(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
  }

  /**
   * Returns whether the target is an objc_proto_library. It does so by making sure that the
   * portable_proto_filters attribute exists in this target's attributes (even if it's empty).
   */
  boolean isObjcProtoLibrary() {
    return ruleContext.attributes().has(PORTABLE_PROTO_FILTERS_ATTR);
  }

  /** Returns whether to use the protobuf library instead of the PB2 library. */
  boolean hasPortableProtoFilters() {
    return ruleContext
        .attributes()
        .isAttributeValueExplicitlySpecified(PORTABLE_PROTO_FILTERS_ATTR);
  }

  /** Returns the list of portable proto filters. */
  ImmutableList<Artifact> getPortableProtoFilters() {
    if (ruleContext.attributes().has(PORTABLE_PROTO_FILTERS_ATTR, LABEL_LIST)) {
      return ruleContext.getPrerequisiteArtifacts(PORTABLE_PROTO_FILTERS_ATTR).list();
    }
    return ImmutableList.of();
  }

  /** Returns the list of well known type protos. */
  NestedSet<Artifact> getWellKnownTypeProtos() {
    NestedSetBuilder<Artifact> wellKnownTypeProtos = NestedSetBuilder.stableOrder();
    for (TransitiveInfoCollection protos :
        ruleContext.getPrerequisites(ObjcRuleClasses.PROTOBUF_WELL_KNOWN_TYPES)) {
      ProtoInfo protoInfo = protos.get(ProtoInfo.PROVIDER);
      if (protoInfo != null) {
        wellKnownTypeProtos.addTransitive(protoInfo.getTransitiveProtoSources());
      } else {
        wellKnownTypeProtos.addTransitive(protos.getProvider(FileProvider.class).getFilesToBuild());
      }
    }
    return wellKnownTypeProtos.build();
  }

  /** Returns the list of proto files to compile. */
  NestedSet<Artifact> getProtoFiles() {
    return NestedSetBuilder.<Artifact>stableOrder()
        .addTransitive(getProtoDepsSources())
        .build();
  }

  /** Returns the proto compiler to be used. */
  Artifact getProtoCompiler() {
    return ruleContext.getPrerequisiteArtifact(ObjcRuleClasses.PROTO_COMPILER_ATTR);
  }

  /** Returns the list of files needed by the proto compiler. */
  Iterable<Artifact> getProtoCompilerSupport() {
    return ruleContext.getPrerequisiteArtifacts(ObjcRuleClasses.PROTO_COMPILER_SUPPORT_ATTR).list();
  }

  /**
   * Processes the case of the proto file name in the same fashion as the objective_c generator's
   * UnderscoresToCamelCase function. This converts snake case to camel case by splitting words
   * by non alphabetic characters. This also treats the URL, HTTP and HTTPS as special words that
   * need to be completely uppercase.
   *
   * Examples:
   *   - j2objc_descriptor -> J2ObjcDescriptor (notice that O is uppercase after the 2)
   *   - my_http_url_array -> MyHTTPURLArray
   *   - proto-descriptor  -> ProtoDescriptor
   *
   * Original code reference:
   * <p>https://github.com/google/protobuf/blob/master/src/google/protobuf/compiler/objectivec/objectivec_helpers.cc
   */
  String getGeneratedProtoFilename(String protoFilename, boolean upcaseReservedWords) {
    boolean lastCharWasDigit = false;
    boolean lastCharWasUpper = false;
    boolean lastCharWasLower = false;

    StringBuilder currentSegment = new StringBuilder();

    ArrayList<String> segments = new ArrayList<>();

    for (int i = 0; i < protoFilename.length(); i++) {
      char currentChar = protoFilename.charAt(i);
      if (CharMatcher.javaDigit().matches(currentChar)) {
        if (!lastCharWasDigit) {
          segments.add(currentSegment.toString());
          currentSegment = new StringBuilder();
        }
        currentSegment.append(currentChar);
        lastCharWasDigit = true;
        lastCharWasUpper = false;
        lastCharWasLower = false;
      } else if (CharMatcher.javaLowerCase().matches(currentChar)) {
        if (!lastCharWasLower && !lastCharWasUpper) {
          segments.add(currentSegment.toString());
          currentSegment = new StringBuilder();
        }
        currentSegment.append(currentChar);
        lastCharWasDigit = false;
        lastCharWasUpper = false;
        lastCharWasLower = true;
      } else if (CharMatcher.javaUpperCase().matches(currentChar)) {
        if (!lastCharWasUpper) {
          segments.add(currentSegment.toString());
          currentSegment = new StringBuilder();
        }
        currentSegment.append(Character.toLowerCase(currentChar));
        lastCharWasDigit = false;
        lastCharWasUpper = true;
        lastCharWasLower = false;
      } else {
        lastCharWasDigit = false;
        lastCharWasUpper = false;
        lastCharWasLower = false;
      }
    }

    segments.add(currentSegment.toString());

    StringBuilder casedSegments = new StringBuilder();
    for (String segment : segments) {
      if (upcaseReservedWords && UPPERCASE_SEGMENTS.contains(segment)) {
        casedSegments.append(segment.toUpperCase());
      } else {
        casedSegments.append(LOWER_UNDERSCORE.to(UPPER_CAMEL, segment));
      }
    }
    return casedSegments.toString();
  }

  /** Returns the sets of proto files that were added using proto_library dependencies. */
  private NestedSet<Artifact> getProtoDepsSources() {
    NestedSetBuilder<Artifact> artifacts = NestedSetBuilder.stableOrder();
    Iterable<ProtoInfo> providers = ruleContext.getPrerequisites("deps", ProtoInfo.PROVIDER);
    for (ProtoInfo provider : providers) {
      artifacts.addTransitive(provider.getTransitiveProtoSources());
    }
    return artifacts.build();
  }
}
