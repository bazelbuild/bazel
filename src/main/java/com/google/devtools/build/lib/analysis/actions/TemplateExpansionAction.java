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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionContinuationOrResult;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.SpawnContinuation;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.util.Fingerprint;
import java.io.IOException;
import java.util.List;
import javax.annotation.Nullable;

/** Action to expand a template and write the expanded content to a file. */
@AutoCodec
@Immutable // if all substitutions are immutable
public final class TemplateExpansionAction extends AbstractAction {

  private static final String GUID = "786c1fe0-dca8-407a-b108-e1ecd6d1bc7f";

  private final Template template;
  private final ImmutableList<Substitution> substitutions;
  private final boolean makeExecutable;

  /**
   * Creates a new TemplateExpansionAction instance.
   *
   * @param owner the action owner.
   * @param inputs the Artifacts that this Action depends on
   * @param primaryOutput the Artifact that will be created by executing this Action.
   * @param template the template that will be expanded by this Action.
   * @param substitutions the substitutions that will be applied to the template. All substitutions
   *     will be applied in order.
   * @param makeExecutable iff true will change the output file to be executable.
   */
  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  TemplateExpansionAction(
      ActionOwner owner,
      NestedSet<Artifact> inputs,
      Artifact primaryOutput,
      Template template,
      List<Substitution> substitutions,
      boolean makeExecutable) {
    super(owner, inputs, ImmutableSet.of(primaryOutput));
    this.template = template;
    this.substitutions = ImmutableList.copyOf(substitutions);
    this.makeExecutable = makeExecutable;
  }

  /**
   * Creates a new TemplateExpansionAction instance for an artifact template.
   *
   * @param owner the action owner.
   * @param templateArtifact the Artifact that will be read as the text template
   *   file
   * @param output the Artifact that will be created by executing this Action.
   * @param substitutions the substitutions that will be applied to the
   *   template. All substitutions will be applied in order.
   * @param makeExecutable iff true will change the output file to be
   *   executable.
   */
  public TemplateExpansionAction(ActionOwner owner,
                                 Artifact templateArtifact,
                                 Artifact output,
                                 List<Substitution> substitutions,
                                 boolean makeExecutable) {
    this(
        owner,
        NestedSetBuilder.create(Order.STABLE_ORDER, templateArtifact),
        output,
        Template.forArtifact(templateArtifact),
        substitutions,
        makeExecutable);
  }

  /**
   * Creates a new TemplateExpansionAction instance without inputs.
   *
   * @param owner the action owner.
   * @param output the Artifact that will be created by executing this Action.
   * @param template the template
   * @param substitutions the substitutions that will be applied to the
   *   template. All substitutions will be applied in order.
   * @param makeExecutable iff true will change the output file to be
   *   executable.
   */
  public TemplateExpansionAction(ActionOwner owner,
                                 Artifact output,
                                 Template template,
                                 List<Substitution> substitutions,
                                 boolean makeExecutable) {
    this(
        owner,
        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        output,
        template,
        substitutions,
        makeExecutable);
  }

  @VisibleForTesting
  public String getFileContents() throws IOException {
    return LocalTemplateExpansionStrategy.INSTANCE.getExpandedTemplateUnsafe(this,
        ArtifactPathResolver.IDENTITY);
  }

  @Override
  public String getStarlarkContent() throws IOException {
    return getFileContents();
  }

  @Override
  public final ActionContinuationOrResult beginExecution(
      ActionExecutionContext actionExecutionContext) throws InterruptedException {
    SpawnContinuation first =
        actionExecutionContext
            .getContext(TemplateExpansionContext.class)
            .expandTemplate(TemplateExpansionAction.this, actionExecutionContext);
    return new ActionContinuationOrResult() {
      private SpawnContinuation spawnContinuation = first;

      @Nullable
      @Override
      public ListenableFuture<?> getFuture() {
        return spawnContinuation.getFuture();
      }

      @Override
      public ActionContinuationOrResult execute()
          throws ActionExecutionException, InterruptedException {
        SpawnContinuation nextContinuation;
        try {
          nextContinuation = spawnContinuation.execute();
          if (!nextContinuation.isDone()) {
            spawnContinuation = nextContinuation;
            return this;
          }
        } catch (ExecException e) {
          throw e.toActionExecutionException(
              "Error expanding template '" + Label.print(getOwner().getLabel()) + "'",
              actionExecutionContext.getVerboseFailures(),
              TemplateExpansionAction.this);
        }
        return ActionContinuationOrResult.of(ActionResult.create(nextContinuation.get()));
      }
    };
  }

  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {
    fp.addString(GUID);
    fp.addString(String.valueOf(makeExecutable));
    fp.addString(template.getKey());
    fp.addInt(substitutions.size());
    for (Substitution entry : substitutions) {
      fp.addString(entry.getKey());
      fp.addString(entry.getValue());
    }
  }

  @Override
  public String getMnemonic() {
    return "TemplateExpand";
  }

  @Override
  protected String getRawProgressMessage() {
    return "Expanding template " + Iterables.getOnlyElement(getOutputs()).prettyPrint();
  }

  public List<Substitution> getSubstitutions() {
    return substitutions;
  }

  public Template getTemplate() {
    return template;
  }

  public boolean makeExecutable() {
    return makeExecutable;
  }

  @Override
  public Dict<String, String> getStarlarkSubstitutions() {
    ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();
    for (Substitution entry : substitutions) {
      builder.put(entry.getKey(), entry.getValue());
    }
    return Dict.copyOf(null, builder.build());
  }
}
