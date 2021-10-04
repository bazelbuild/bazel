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

package com.google.devtools.build.lib.rules.proto;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;

/**
 * A list proto source files to be excluded from code generation.
 *
 * <p>There are cases where we need to identify proto files that we should not create generated
 * files for. For example, we should not create generated code for google/protobuf/any.proto, which
 * is already baked into the language support libraries. This class provides us with the ability to
 * identify these proto files and avoid linking in their associated generated files.
 */
public class ProtoSourceFileExcludeList {
  private static final PathFragment BAZEL_TOOLS_PREFIX =
      PathFragment.create("external/bazel_tools/");
  private final RuleContext ruleContext;
  private final ImmutableSet<PathFragment> noGenerateProtoFilePaths;

  /**
   * Creates a proto source file exclusion list.
   *
   * @param ruleContext the proto rule context.
   * @param noGenerateProtoFiles a list of .proto files. The list will be iterated.
   */
  public ProtoSourceFileExcludeList(
      RuleContext ruleContext, NestedSet<Artifact> noGenerateProtoFiles) {
    this.ruleContext = ruleContext;
    ImmutableSet.Builder<PathFragment> noGenerateProtoFilePathsBuilder =
        new ImmutableSet.Builder<>();
    for (Artifact noGenerateProtoFile : noGenerateProtoFiles.toList()) {
      PathFragment execPath = noGenerateProtoFile.getExecPath();
      // For listed protos bundled with the Bazel tools repository, their exec paths start
      // with external/bazel_tools/. This prefix needs to be removed first, because the protos in
      // user repositories will not have that prefix.
      if (execPath.startsWith(BAZEL_TOOLS_PREFIX)) {
        noGenerateProtoFilePathsBuilder.add(execPath.relativeTo(BAZEL_TOOLS_PREFIX));
      } else {
        noGenerateProtoFilePathsBuilder.add(execPath);
      }
    }
    noGenerateProtoFilePaths = noGenerateProtoFilePathsBuilder.build();
  }

  /** Filters the listed protos from the given protos. */
  public Iterable<Artifact> filter(Iterable<Artifact> protoFiles) {
    return Streams.stream(protoFiles).filter(f -> !isExcluded(f)).collect(toImmutableSet());
  }

  /**
   * Checks the proto sources for mixing excluded and non-excluded protos in one single
   * proto_library rule. Registers an attribute error if proto mixing is detected.
   *
   * @param protoSources the protos to filter.
   * @param topLevelProtoRuleName the name of the top-level rule that generates the protos.
   * @return whether the proto sources are clean without mixing.
   */
  public boolean checkSrcs(ImmutableList<ProtoSource> protoSources, String topLevelProtoRuleName) {
    ImmutableList.Builder<Artifact> protoFiles = ImmutableList.builder();
    for (ProtoSource protoSource : protoSources) {
      protoFiles.add(protoSource.getOriginalSourceFile());
    }
    return checkSrcs(protoFiles.build(), topLevelProtoRuleName);
  }

  @Deprecated
  public boolean checkSrcs(Iterable<Artifact> protoFiles, String topLevelProtoRuleName) {
    List<Artifact> excluded = new ArrayList<>();
    List<Artifact> nonExcluded = new ArrayList<>();
    for (Artifact protoFile : protoFiles) {
      if (isExcluded(protoFile)) {
        excluded.add(protoFile);
      } else {
        nonExcluded.add(protoFile);
      }
    }
    if (!nonExcluded.isEmpty() && !excluded.isEmpty()) {
      ruleContext.attributeError(
          "srcs",
          createBlacklistedProtosMixError(
              Artifact.toRootRelativePaths(excluded),
              Artifact.toRootRelativePaths(nonExcluded),
              ruleContext.getLabel().toString(),
              topLevelProtoRuleName));
    }

    return excluded.isEmpty();
  }

  /** Returns whether the given proto file is excluded. */
  public boolean isExcluded(Artifact protoFile) {
    return noGenerateProtoFilePaths.contains(protoFile.getExecPath());
  }

  /**
   * Returns an attribute for the implicit dependency on blacklist proto filegroups.
   *
   * @param attributeName the name of the attribute.
   * @param blacklistFileGroups a list of labels pointin to the filegroups containing excluded
   *     protos.
   */
  public static Attribute.Builder<List<Label>> blacklistFilegroupAttribute(
      String attributeName, List<Label> blacklistFileGroups) {
    return attr(attributeName, LABEL_LIST)
        .cfg(ExecutionTransitionFactory.create())
        .value(blacklistFileGroups);
  }

  @VisibleForTesting
  public static String createBlacklistedProtosMixError(
      Iterable<String> excluded,
      Iterable<String> nonExcluded,
      String protoLibraryRuleLabel,
      String topLevelProtoRuleName) {
    return String.format(
        "The 'srcs' attribute of '%s' contains protos for which '%s' "
            + "shouldn't generate code (%s), in addition to protos for which it should (%s).\n"
            + "Separate '%1$s' into 2 proto_library rules.",
        protoLibraryRuleLabel,
        topLevelProtoRuleName,
        Joiner.on(", ").join(excluded),
        Joiner.on(", ").join(nonExcluded));
  }
}
