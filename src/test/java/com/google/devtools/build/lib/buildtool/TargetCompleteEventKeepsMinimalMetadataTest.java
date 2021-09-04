// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Iterables;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.TargetCompleteEvent;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import java.util.Collection;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Validates that TargetCompleteEvents do not keep a map of action output metadata for the
 * _validation output group, which can be quite large.
 */
@RunWith(JUnit4.class)
public class TargetCompleteEventKeepsMinimalMetadataTest extends BuildIntegrationTestCase {

  @Test
  public void artifactsNotRetained() throws Exception {
    write(
        "validation_actions/defs.bzl",
        "def _rule_with_implicit_outs_and_validation_impl(ctx):",
        "",
        "  ctx.actions.write(ctx.outputs.main, \"main output\\n\")",
        "",
        "  ctx.actions.write(ctx.outputs.implicit, \"implicit output\\n\")",
        "",
        "  validation_output = ctx.actions.declare_file(ctx.attr.name + \".validation\")",
        "  # The actual tool will be created in individual tests, depending on whether",
        "  # validation should pass or fail.",
        "  ctx.actions.run(",
        "      outputs = [validation_output],",
        "      executable = ctx.executable._validation_tool,",
        "      arguments = [validation_output.path])",
        "",
        "  return [",
        "    DefaultInfo(files = depset([ctx.outputs.main])),",
        "    OutputGroupInfo(_validation = depset([validation_output])),",
        "  ]",
        "",
        "",
        "rule_with_implicit_outs_and_validation = rule(",
        "  implementation = _rule_with_implicit_outs_and_validation_impl,",
        "  outputs = {",
        "    \"main\": \"%{name}.main\",",
        "    \"implicit\": \"%{name}.implicit\",",
        "  },",
        "  attrs = {",
        "    \"_validation_tool\": attr.label(",
        "        allow_single_file = True,",
        "        default = Label(\"//validation_actions:validation_tool\"),",
        "        executable = True,",
        "        cfg = \"host\"),",
        "  }",
        ")");
    write("validation_actions/validation_tool", "#!/bin/bash", "echo \"validation output\" > $1")
        .setExecutable(true);
    write(
        "validation_actions/BUILD",
        "load(",
        "    \":defs.bzl\",",
        "    \"rule_with_implicit_outs_and_validation\")",
        "",
        "rule_with_implicit_outs_and_validation(name = \"foo0\")");

    AtomicReference<TargetCompleteEvent> targetCompleteEventRef = new AtomicReference<>();
    runtimeWrapper.registerSubscriber(
        new Object() {
          @SuppressWarnings("unused")
          @Subscribe
          public void accept(TargetCompleteEvent event) {
            targetCompleteEventRef.set(event);
          }
        });

    addOptions("--experimental_run_validations");
    BuildResult buildResult = buildTarget("//validation_actions:foo0");

    Collection<ConfiguredTarget> successfulTargets = buildResult.getSuccessfulTargets();
    ConfiguredTarget fooTarget = Iterables.getOnlyElement(successfulTargets);

    // Check that the primary output, :foo0.main, has its metadata retained and the
    // CompletionContext can confirm it is an output file.
    Artifact main =
        ((RuleConfiguredTarget) fooTarget)
            .getArtifactByOutputLabel(
                Label.parseAbsoluteUnchecked("//validation_actions:foo0.main"));
    assertThat(targetCompleteEventRef.get().getCompletionContext().isGuaranteedToBeOutputFile(main))
        .isTrue();

    // Check that the validation output, :foo0.validation, does not have its metadata retained and
    // the CompletionContext cannot confirm it is an output file (even though it is).
    OutputGroupInfo outputGroups = fooTarget.get(OutputGroupInfo.STARLARK_CONSTRUCTOR);
    NestedSet<Artifact> validationArtifacts =
        outputGroups.getOutputGroup(OutputGroupInfo.VALIDATION);
    assertThat(validationArtifacts.isEmpty()).isFalse();

    Artifact validationArtifact = Iterables.getOnlyElement(validationArtifacts.toList());

    assertThat(targetCompleteEventRef.get()).isNotNull();
    assertThat(
            targetCompleteEventRef
                .get()
                .getCompletionContext()
                .isGuaranteedToBeOutputFile(validationArtifact))
        .isFalse();
  }
}
