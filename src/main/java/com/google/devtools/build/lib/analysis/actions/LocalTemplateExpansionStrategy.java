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

package com.google.devtools.build.lib.analysis.actions;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.unsafe.StringUnsafe;
import com.google.devtools.build.lib.util.DeterministicWriter;
import com.google.devtools.build.lib.util.StringUtilities;
import java.io.IOException;
import java.util.List;
import net.starlark.java.eval.EvalException;

/** Strategy to perform template expansion locally. */
public class LocalTemplateExpansionStrategy implements TemplateExpansionContext {
  public static final Class<LocalTemplateExpansionStrategy> TYPE =
      LocalTemplateExpansionStrategy.class;

  public static LocalTemplateExpansionStrategy INSTANCE = new LocalTemplateExpansionStrategy();

  @Override
  public ImmutableList<SpawnResult> expandTemplate(
      AbstractAction action,
      ActionExecutionContext ctx,
      TemplateExpansionContext.TemplateMetadata templateMetadata)
      throws InterruptedException, ExecException {
    try {
      FileWriteActionContext fileWriteActionContext = ctx.getContext(FileWriteActionContext.class);
      DeterministicWriter deterministicWriter;
      if (fileWriteActionContext.mayRetainWriter()) {
        // Snapshot the input before retaining the writer. The input file and the action's path
        // resolver may no longer be valid when the output is materialized in a later build.
        String template = templateMetadata.template().getContent(ctx.getPathResolver());
        List<Substitution> substitutions = templateMetadata.substitutions();
        // Report substitution errors during execution, without retaining the expanded values.
        for (Substitution substitution : substitutions) {
          var unused = substitution.getValue();
        }
        deterministicWriter =
            out -> {
              try {
                out.write(
                    StringUnsafe.getInternalStringBytes(expandTemplate(template, substitutions)));
              } catch (EvalException e) {
                throw new IllegalStateException(
                    "Previously validated template substitution failed", e);
              } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrupted while expanding template", e);
              }
            };
      } else {
        String expandedTemplate =
            getExpandedTemplateUnsafe(
                templateMetadata.template(),
                templateMetadata.substitutions(),
                ctx.getPathResolver());
        deterministicWriter =
            out -> out.write(StringUnsafe.getInternalStringBytes(expandedTemplate));
      }
      return fileWriteActionContext.writeOutputToFile(
          action,
          ctx,
          deterministicWriter,
          templateMetadata.makeExecutable(),
          /* isRemotable= */ true);
    } catch (IOException | EvalException e) {
      throw new EnvironmentalExecException(
          e,
          FailureDetail.newBuilder()
              .setExecution(
                  Execution.newBuilder().setCode(Execution.Code.LOCAL_TEMPLATE_EXPANSION_FAILURE))
              .build());
    }
  }

  /**
   * Get the result of the template expansion prior to executing the action. TODO(b/110418949): Stop
   * public access to this method as it's unhealthy to evaluate the action result without the action
   * being executed.
   */
  public String getExpandedTemplateUnsafe(
      Template template, List<Substitution> substitutions, ArtifactPathResolver resolver)
      throws EvalException, IOException, InterruptedException {
    return expandTemplate(template.getContent(resolver), substitutions);
  }

  private static String expandTemplate(String templateString, List<Substitution> substitutions)
      throws EvalException, InterruptedException {
    for (Substitution entry : substitutions) {
      templateString =
          StringUtilities.replaceAllLiteral(templateString, entry.getKey(), entry.getValue());
    }
    return templateString;
  }
}
