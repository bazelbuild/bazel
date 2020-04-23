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

import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.SpawnContinuation;
import com.google.devtools.build.lib.util.StringUtilities;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

/** Strategy to perform tempate expansion locally */
public class LocalTemplateExpansionStrategy implements TemplateExpansionContext {
  public static final Class<LocalTemplateExpansionStrategy> TYPE =
      LocalTemplateExpansionStrategy.class;

  public static LocalTemplateExpansionStrategy INSTANCE = new LocalTemplateExpansionStrategy();

  @Override
  public SpawnContinuation expandTemplate(
      TemplateExpansionAction action, ActionExecutionContext ctx) throws InterruptedException {
    try {
      final String expandedTemplate = getExpandedTemplateUnsafe(action, ctx.getPathResolver());
      DeterministicWriter deterministicWriter =
          new DeterministicWriter() {
            @Override
            public void writeOutputFile(OutputStream out) throws IOException {
              out.write(expandedTemplate.getBytes(StandardCharsets.UTF_8));
            }
          };
      return ctx.getContext(FileWriteActionContext.class)
          .beginWriteOutputToFile(
              action, ctx, deterministicWriter, action.makeExecutable(), /*isRemotable=*/ true);
    } catch (IOException e) {
      return SpawnContinuation.failedWithExecException(new EnvironmentalExecException(e));
    }
  }

  /**
   * Get the result of the template expansion prior to executing the action.
   * TODO(b/110418949): Stop public access to this method as it's unhealthy to evaluate the
   * action result without the action being executed.
   */
  public String getExpandedTemplateUnsafe(TemplateExpansionAction action,
      ArtifactPathResolver resolver) throws IOException {
    String templateString;
    templateString = action.getTemplate().getContent(resolver);
    for (Substitution entry : action.getSubstitutions()) {
      templateString =
          StringUtilities.replaceAllLiteral(templateString, entry.getKey(), entry.getValue());
    }
    return templateString;
  }
}
