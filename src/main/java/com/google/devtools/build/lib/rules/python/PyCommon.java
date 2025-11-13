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
package com.google.devtools.build.lib.rules.python;


import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.extra.PythonInfo;
import com.google.devtools.build.lib.analysis.PseudoAction;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.protobuf.GeneratedMessage.GeneratedExtension;
import java.util.Collection;
import java.util.List;
import java.util.UUID;

/** A helper class for analyzing a Python configured target. */
public final class PyCommon {
  /** Name of the version attribute. */
  public static final String PYTHON_VERSION_ATTRIBUTE = "python_version";

  private PyCommon() {}

  // Public so that Starlark bindings can access it. Should only be called by PyStarlarkBuiltins.
  // TODO(b/253059598): Remove support for this; https://github.com/bazelbuild/bazel/issues/16455
  public static void registerPyExtraActionPseudoAction(
      RuleContext ruleContext, NestedSet<Artifact> dependencyTransitivePythonSources) {
    ruleContext.registerAction(
        makePyExtraActionPseudoAction(
            ruleContext.getActionOwner(),
            // Has to be unfiltered sources as filtered will give an error for
            // unsupported file types where as certain tests only expect a warning.
            ruleContext.getPrerequisiteArtifacts("srcs").list(),
            // We must not add the files declared in the srcs of this rule.;
            dependencyTransitivePythonSources,
            PseudoAction.getDummyOutput(ruleContext)));
  }

  /**
   * Creates a {@link PseudoAction} that is only used for providing information to the blaze
   * extra_action feature.
   */
  private static Action makePyExtraActionPseudoAction(
      ActionOwner owner,
      List<Artifact> sources,
      NestedSet<Artifact> dependencies,
      Artifact output) {

    PythonInfo info =
        PythonInfo.newBuilder()
            .addAllSourceFile(Artifact.toExecPaths(sources))
            .addAllDepFile(Artifact.toExecPaths(dependencies.toList()))
            .build();

    return new PyPseudoAction(
        owner,
        NestedSetBuilder.<Artifact>stableOrder()
            .addAll(sources)
            .addTransitive(dependencies)
            .build(),
        ImmutableList.of(output),
        "Python",
        PYTHON_INFO,
        info);
  }

  @SerializationConstant @VisibleForSerialization
  static final GeneratedExtension<ExtraActionInfo, PythonInfo> PYTHON_INFO = PythonInfo.pythonInfo;

  // Used purely to set the legacy ActionType of the ExtraActionInfo.
  @Immutable
  private static final class PyPseudoAction extends PseudoAction<PythonInfo> {
    private static final UUID ACTION_UUID = UUID.fromString("8d720129-bc1a-481f-8c4c-dbe11dcef319");

    public PyPseudoAction(
        ActionOwner owner,
        NestedSet<Artifact> inputs,
        Collection<Artifact> outputs,
        String mnemonic,
        GeneratedExtension<ExtraActionInfo, PythonInfo> infoExtension,
        PythonInfo info) {
      super(ACTION_UUID, owner, inputs, outputs, mnemonic, infoExtension, info);
    }
  }
}
