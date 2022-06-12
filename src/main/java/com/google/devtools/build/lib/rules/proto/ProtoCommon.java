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

package com.google.devtools.build.lib.rules.proto;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.StrictDepsMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.packages.BazelModuleContext;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/** Utility functions for proto_library and proto aspect implementations. */
public class ProtoCommon {
  private ProtoCommon() {
    throw new UnsupportedOperationException();
  }

  // Keep in sync with the migration label in
  // https://github.com/bazelbuild/rules_proto/blob/master/proto/defs.bzl.
  @VisibleForTesting
  public static final String PROTO_RULES_MIGRATION_LABEL =
      "__PROTO_RULES_MIGRATION_DO_NOT_USE_WILL_BREAK__";

  private static final Interner<PathFragment> PROTO_SOURCE_ROOT_INTERNER =
      BlazeInterners.newWeakInterner();

  /**
   * Returns a memory efficient version of the passed protoSourceRoot.
   *
   * <p>Any sizable proto graph will contain many {@code .proto} sources with the same source root.
   * We can't afford to have all of them represented as individual objects in memory.
   *
   * @param protoSourceRoot
   * @return
   */
  static PathFragment memoryEfficientProtoSourceRoot(PathFragment protoSourceRoot) {
    return PROTO_SOURCE_ROOT_INTERNER.intern(protoSourceRoot);
  }

  // =================================================================
  // Protocol compiler invocation stuff.

  /**
   * Each language-specific initialization method will call this to construct Artifacts representing
   * its protocol compiler outputs.
   *
   * @param extension Remove ".proto" and replace it with this to produce the output file name, e.g.
   *     ".pb.cc".
   * @param pythonNames If true, replace hyphens in the file name with underscores, and dots in the
   *     file name with forward slashes, as required for Python modules.
   */
  public static ImmutableList<Artifact> getGeneratedOutputs(
      RuleContext ruleContext,
      ImmutableList<Artifact> protoSources,
      String extension,
      boolean pythonNames) {
    ImmutableList.Builder<Artifact> outputsBuilder = new ImmutableList.Builder<>();
    ArtifactRoot genfiles = ruleContext.getGenfilesDirectory();
    for (Artifact src : protoSources) {
      PathFragment srcPath =
          src.getOutputDirRelativePath(ruleContext.getConfiguration().isSiblingRepositoryLayout());
      if (pythonNames) {
        srcPath = srcPath.replaceName(srcPath.getBaseName().replace('-', '_'));

        // Protoc python plugin converts dots in filenames to slashes when generating python source
        // paths. For example, "myproto.gen.proto" generates "myproto/gen_pb2.py".
        String baseName = srcPath.getBaseName();
        int lastDot = baseName.lastIndexOf('.');
        if (lastDot > 0) {
          String baseNameNoExtension = baseName.substring(0, lastDot);
          srcPath =
              srcPath.replaceName(
                  baseNameNoExtension.replace('.', '/') + baseName.substring(lastDot));
        }
      }
      // Note that two proto_library rules can have the same source file, so this is actually a
      // shared action. NB: This can probably result in action conflicts if the proto_library rules
      // are not the same.
      outputsBuilder.add(
          ruleContext.getShareableArtifact(
              FileSystemUtils.replaceExtension(srcPath, extension), genfiles));
    }
    return outputsBuilder.build();
  }

  /**
   * Each language-specific initialization method will call this to construct Artifacts representing
   * its protocol compiler outputs.
   *
   * @param extension Remove ".proto" and replace it with this to produce the output file name, e.g.
   *     ".pb.cc".
   */
  public static ImmutableList<Artifact> getGeneratedOutputs(
      RuleContext ruleContext, ImmutableList<Artifact> protoSources, String extension) {
    return getGeneratedOutputs(ruleContext, protoSources, extension, false);
  }

  /**
   * Decides whether this proto_library should check for strict proto deps.
   *
   * <p>Only takes into account the command-line flag --strict_proto_deps.
   */
  @VisibleForTesting
  public static boolean areDepsStrict(RuleContext ruleContext) {
    StrictDepsMode getBool = ruleContext.getFragment(ProtoConfiguration.class).strictProtoDeps();
    return getBool != StrictDepsMode.OFF && getBool != StrictDepsMode.DEFAULT;
  }

  public static void checkPrivateStarlarkificationAllowlist(StarlarkThread thread)
      throws EvalException {
    Label label =
        ((BazelModuleContext) Module.ofInnermostEnclosingStarlarkFunction(thread).getClientData())
            .label();
    if (!label.getPackageIdentifier().getRepository().toString().equals("@_builtins")) {
      throw Starlark.errorf("Rule in '%s' cannot use private API", label.getPackageName());
    }
  }
}
