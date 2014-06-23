// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules;

import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionContextConsumer;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Executor.ActionContext;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.blaze.BlazeModule;
import com.google.devtools.build.lib.blaze.BlazeRuntime;
import com.google.devtools.build.lib.blaze.Command;
import com.google.devtools.build.lib.blaze.GotOptionsEvent;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.BuildInfoHelper;
import com.google.devtools.build.lib.view.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.view.WorkspaceStatusAction;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;

import java.io.IOException;
import java.util.Map;
import java.util.UUID;

/**
 * Module implementing the rule set of Bazel.
 */
public class BazelRulesModule extends BlazeModule {
  private static class BazelWorkspaceStatusAction extends WorkspaceStatusAction {
    private final Artifact stableStatus;
    private final Artifact volatileStatus;

    private BazelWorkspaceStatusAction(Artifact stableStatus, Artifact volatileStatus) {
      super(BuildInfoHelper.BUILD_INFO_ACTION_OWNER, Artifact.NO_ARTIFACTS,
          ImmutableList.of(stableStatus, volatileStatus));
      this.stableStatus = stableStatus;
      this.volatileStatus = volatileStatus;
    }

    @Override
    public String describeStrategy(Executor executor) {
      return "";
    }

    @Override
    public void execute(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      try {
        FileSystemUtils.writeContent(stableStatus.getPath(), new byte[] {});
        FileSystemUtils.writeContent(volatileStatus.getPath(), new byte[] {});
      } catch (IOException e) {
        throw new ActionExecutionException(e, this, true);
      }
    }

    @Override
    public String getMnemonic() {
      return "BazelWorkspaceStatusAction";
    }

    @Override
    public ResourceSet estimateResourceConsumption(Executor executor) {
      return ResourceSet.ZERO;
    }

    @Override
    protected String computeKey() {
      return "";
    }

    @Override
    public Artifact getVolatileStatus() {
      return volatileStatus;
    }

    @Override
    public Artifact getStableStatus() {
      return stableStatus;
    }
  }

  private class BazelStatusActionFactory implements WorkspaceStatusAction.Factory {
    @Override
    public Map<String, String> createDummyWorkspaceStatus() {
      return ImmutableMap.of();
    }

    @Override
    public WorkspaceStatusAction createWorkspaceStatusAction(
        ArtifactFactory factory, ArtifactOwner artifactOwner, Supplier<UUID> buildId) {
      Root root = runtime.getDirectories().getBuildDataDirectory();

      Artifact stableArtifact =
          factory.getSpecialMetadataHandlingArtifact(new PathFragment("stable-status.txt"),
              root, artifactOwner,
              /*forceConstantMetadata=*/false, /*forceDigestMetadata=*/true);
      Artifact volatileArtifact =
          factory.getSpecialMetadataHandlingArtifact(new PathFragment("volatile-status.txt"),
              root, artifactOwner,
            /*forceConstantMetadata=*/true, /*forceDigestMetadata=*/false);

      return new BazelWorkspaceStatusAction(stableArtifact, volatileArtifact);
    }
  }

  /**
   * Execution options for google3 rules.
   */
  public static class Google3ExecutionOptions extends OptionsBase {
    @Option(name = "spawn_strategy", defaultValue = "", category = "strategy", help =
        "Specify where spawn actions are executed by default. This is "
        + "overridden by the more specific strategy options.")
    public String spawnStrategy;

    @Option(name = "genrule_strategy",
        defaultValue = "",
        category = "strategy",
        help = "Specify how to execute genrules. "
            + "'standalone' means run all of them locally.")
    public String genruleStrategy;
  }

  private static class Google3ActionContextConsumer implements ActionContextConsumer {
    Google3ExecutionOptions options;

    private Google3ActionContextConsumer(Google3ExecutionOptions options) {
      this.options = options;

    }
    @Override
    public Map<String, String> getSpawnActionContexts() {
      ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();

      builder.put("Genrule", options.genruleStrategy);

      // TODO(bazel-team): put this in getActionContexts (key=SpawnActionContext.class) instead
      builder.put("", options.spawnStrategy);

      return builder.build();
    }

    @Override
    public Map<Class<? extends ActionContext>, String> getActionContexts() {
      ImmutableMap.Builder<Class<? extends ActionContext>, String> builder =
          ImmutableMap.builder();
      return builder.build();
    }
  }

  private BlazeRuntime runtime;
  private OptionsProvider optionsProvider;

  @Override
  public void beforeCommand(BlazeRuntime blazeRuntime, Command command) {
    this.runtime = blazeRuntime;
    runtime.getEventBus().register(this);
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return command.builds()
        ? ImmutableList.<Class<? extends OptionsBase>>of(Google3ExecutionOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Override
  public ActionContextConsumer getActionContextConsumer() {
    return new Google3ActionContextConsumer(
        optionsProvider.getOptions(Google3ExecutionOptions.class));
  }

  @Subscribe
  public void gotOptions(GotOptionsEvent event) {
    optionsProvider = event.getOptions();
  }

  @Override
  public void initializeRuleClasses(ConfiguredRuleClassProvider.Builder builder) {
    BazelRuleClassProvider.setup(builder);
  }

  @Override
  public WorkspaceStatusAction.Factory getWorkspaceStatusActionFactory() {
    return new BazelStatusActionFactory();
  }
}
