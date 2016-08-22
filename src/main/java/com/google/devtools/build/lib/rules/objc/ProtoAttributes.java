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

import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AbstractConfiguredTarget;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.proto.ProtoSourcesProvider;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileType;

/** Common rule attributes used by an objc_proto_library. */
final class ProtoAttributes {
  @VisibleForTesting
  static final String FILES_DEPRECATED_WARNING =
      "Using files and filegroups in objc_proto_library is deprecated";

  @VisibleForTesting
  static final String NO_PROTOS_ERROR =
      "no protos to compile - a non-empty deps attribute is required";

  @VisibleForTesting
  static final String PORTABLE_PROTO_FILTERS_NOT_EXCLUSIVE_ERROR =
      "The portable_proto_filters attribute is incompatible with the options_file, output_cpp, "
          + "per_proto_includes and use_objc_header_names attributes.";

  @VisibleForTesting
  static final String PORTABLE_PROTO_FILTERS_EMPTY_ERROR =
      "The portable_proto_filters attribute can't be empty";

  @VisibleForTesting
  static final String OBJC_PROTO_LIB_DEP_IN_PROTOCOL_BUFFERS2_DEPS_ERROR =
      "Protocol Buffers 2 objc_proto_library targets can't depend on other objc_proto_library "
          + "targets. Please migrate your Protocol Buffers 2 objc_proto_library targets to use the "
          + "portable_proto_filters attribute.";

  @VisibleForTesting
  static final String PROTOCOL_BUFFERS2_IN_PROTOBUF_DEPS_ERROR =
      "Protobuf objc_proto_library targets can't depend on Protocol Buffers 2 objc_proto_library "
          + "targets. Please migrate your Protocol Buffers 2 objc_proto_library targets to use the "
          + "portable_proto_filters attribute.";

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
   * Validates the proto attributes for this target.
   *
   * <ul>
   * <li>Validates that there are protos specified to be compiled.
   * <li>Validates that, when enabling the open source protobuf library, the options for the PB2 are
   *     not specified also.
   * <li>Validates that, when enabling the open source protobuf library, the rule specifies at least
   *     one portable proto filter file.
   * </ul>
   */
  public void validate() throws RuleErrorException {
    PrerequisiteArtifacts prerequisiteArtifacts =
        ruleContext.getPrerequisiteArtifacts("deps", Mode.TARGET);
    ImmutableList<Artifact> protos = prerequisiteArtifacts.filter(FileType.of(".proto")).list();
    if (!protos.isEmpty()) {
      ruleContext.attributeWarning("deps", FILES_DEPRECATED_WARNING);
    }

    if (ruleContext
        .attributes()
        .isAttributeValueExplicitlySpecified(ObjcProtoLibraryRule.PORTABLE_PROTO_FILTERS_ATTR)) {
      if (getProtoFiles().isEmpty() && !hasObjcProtoLibraryDependencies()) {
        ruleContext.throwWithRuleError(NO_PROTOS_ERROR);
      }

      if (getPortableProtoFilters().isEmpty()) {
        ruleContext.throwWithRuleError(PORTABLE_PROTO_FILTERS_EMPTY_ERROR);
      }

      if (outputsCpp()
          || usesObjcHeaderNames()
          || needsPerProtoIncludes()
          || getOptionsFile().isPresent()) {
        ruleContext.throwWithRuleError(PORTABLE_PROTO_FILTERS_NOT_EXCLUSIVE_ERROR);
      }
      if (hasPB2Dependencies()) {
        ruleContext.throwWithRuleError(PROTOCOL_BUFFERS2_IN_PROTOBUF_DEPS_ERROR);
      }

    } else {
      if (getProtoFiles().isEmpty()) {
        ruleContext.throwWithRuleError(NO_PROTOS_ERROR);
      }

      if (outputsCpp()) {
        ruleContext.ruleWarning(
            "The output_cpp attribute has been deprecated. Please "
                + "refer to b/29342376 for information on possible alternatives.");
      }
      if (!usesObjcHeaderNames()) {
        ruleContext.ruleWarning(
            "As part of the migration process, it is recommended to enable "
                + "use_objc_header_names. Please refer to b/29368416 for more information.");
      }
      if (hasObjcProtoLibraryDependencies()) {
        ruleContext.throwWithRuleError(OBJC_PROTO_LIB_DEP_IN_PROTOCOL_BUFFERS2_DEPS_ERROR);
      }
    }
  }

  /** Returns whether the generated files should be C++ or Objective C. */
  boolean outputsCpp() {
    if (ruleContext.attributes().has(ObjcProtoLibraryRule.OUTPUT_CPP_ATTR, Type.BOOLEAN)) {
      return ruleContext.attributes().get(ObjcProtoLibraryRule.OUTPUT_CPP_ATTR, Type.BOOLEAN);
    }
    return false;
  }

  /** Returns whether the generated header files should have be of type pb.h or pbobjc.h. */
  boolean usesObjcHeaderNames() {
    if (ruleContext
        .attributes()
        .has(ObjcProtoLibraryRule.USE_OBJC_HEADER_NAMES_ATTR, Type.BOOLEAN)) {
      return ruleContext
          .attributes()
          .get(ObjcProtoLibraryRule.USE_OBJC_HEADER_NAMES_ATTR, Type.BOOLEAN);
    }
    return false;
  }

  /** Returns whether the includes should include each of the proto generated headers. */
  boolean needsPerProtoIncludes() {
    if (ruleContext.attributes().has(ObjcProtoLibraryRule.PER_PROTO_INCLUDES_ATTR, Type.BOOLEAN)) {
      return ruleContext
          .attributes()
          .get(ObjcProtoLibraryRule.PER_PROTO_INCLUDES_ATTR, Type.BOOLEAN);
    }
    return false;
  }

  /** Returns whether to use the protobuf library instead of the PB2 library. */
  boolean hasPortableProtoFilters() {
    return ruleContext
        .attributes()
        .isAttributeValueExplicitlySpecified(ObjcProtoLibraryRule.PORTABLE_PROTO_FILTERS_ATTR);
  }

  /** Returns the list of portable proto filters. */
  ImmutableList<Artifact> getPortableProtoFilters() {
    if (ruleContext
        .attributes()
        .has(ObjcProtoLibraryRule.PORTABLE_PROTO_FILTERS_ATTR, LABEL_LIST)) {
      return ruleContext
          .getPrerequisiteArtifacts(ObjcProtoLibraryRule.PORTABLE_PROTO_FILTERS_ATTR, Mode.HOST)
          .list();
    }
    return ImmutableList.of();
  }

  /** Returns the list of well known type protos. */
  ImmutableList<Artifact> getWellKnownTypeProtos() {
    return ruleContext
        .getPrerequisiteArtifacts(ObjcRuleClasses.PROTOBUF_WELL_KNOWN_TYPES, Mode.HOST)
        .list();
  }

  /** Returns the options file, or {@link Optional#absent} if it was not specified. */
  Optional<Artifact> getOptionsFile() {
    if (ruleContext.attributes().has(ObjcProtoLibraryRule.OPTIONS_FILE_ATTR, LABEL)) {
      return Optional.fromNullable(
          ruleContext.getPrerequisiteArtifact(ObjcProtoLibraryRule.OPTIONS_FILE_ATTR, Mode.HOST));
    }
    return Optional.absent();
  }

  /** Returns the list of proto files to compile. */
  NestedSet<Artifact> getProtoFiles() {
    return NestedSetBuilder.<Artifact>stableOrder()
        .addAll(getProtoDepsFiles())
        .addTransitive(getProtoDepsSources())
        .build();
  }

  /** Returns the proto compiler to be used. */
  Artifact getProtoCompiler() {
    return ruleContext.getPrerequisiteArtifact(ObjcRuleClasses.PROTO_COMPILER_ATTR, Mode.HOST);
  }

  /** Returns the list of files needed by the proto compiler. */
  Iterable<Artifact> getProtoCompilerSupport() {
    return ruleContext
        .getPrerequisiteArtifacts(ObjcRuleClasses.PROTO_COMPILER_SUPPORT_ATTR, Mode.HOST)
        .list();
  }

  /** Returns the sets of proto files that were added using proto_library dependencies. */
  private NestedSet<Artifact> getProtoDepsSources() {
    NestedSetBuilder<Artifact> artifacts = NestedSetBuilder.stableOrder();
    Iterable<ProtoSourcesProvider> providers =
        ruleContext.getPrerequisites("deps", Mode.TARGET, ProtoSourcesProvider.class);
    for (ProtoSourcesProvider provider : providers) {
      artifacts.addTransitive(provider.getTransitiveProtoSources());
    }
    return artifacts.build();
  }

  /**
   * Returns the list of proto files that were added directly into the deps attributes. This way of
   * specifying the protos is deprecated, and displays a warning when the target does so.
   */
  private ImmutableList<Artifact> getProtoDepsFiles() {
    PrerequisiteArtifacts prerequisiteArtifacts =
        ruleContext.getPrerequisiteArtifacts("deps", Mode.TARGET);
    return prerequisiteArtifacts.filter(FileType.of(".proto")).list();
  }

  private boolean hasPB2Dependencies() {
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("deps", Mode.TARGET)) {
      if (isObjcProtoLibrary(dep) && !hasProtobufProvider(dep)) {
        return true;
      }
    }
    return false;
  }

  private boolean hasObjcProtoLibraryDependencies() {
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("deps", Mode.TARGET)) {
      if (isObjcProtoLibrary(dep)) {
        return true;
      }
    }
    return false;
  }

  private boolean isObjcProtoLibrary(TransitiveInfoCollection dependency) {
    try {
      AbstractConfiguredTarget target = (AbstractConfiguredTarget) dependency;
      String targetName = target.getTarget().getTargetKind();
      return targetName.equals("objc_proto_library rule");
    } catch (Exception e) {
      return false;
    }
  }

  private boolean hasProtobufProvider(TransitiveInfoCollection dependency) {
    return dependency.getProvider(ObjcProtoProvider.class) != null;
  }
}
