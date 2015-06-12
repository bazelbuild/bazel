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
package com.google.devtools.build.lib.bazel;

import com.google.common.base.Joiner;
import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.actions.SimpleActionContextProvider;
import com.google.devtools.build.lib.analysis.BuildInfo;
import com.google.devtools.build.lib.analysis.BuildInfoHelper;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction.Key;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction.KeyType;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.GotOptionsEvent;
import com.google.devtools.build.lib.util.NetUtil;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsBase;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Provides information about the workspace (e.g. source control context, current machine, current
 * user, etc).
 */
public class BazelWorkspaceStatusModule extends BlazeModule {
  private static class BazelWorkspaceStatusAction extends WorkspaceStatusAction {
    private final Artifact stableStatus;
    private final Artifact volatileStatus;
    private final AtomicReference<Options> options;
    
    private final String username;
    private final String hostname;
    private final long timestamp;

    private BazelWorkspaceStatusAction(
        AtomicReference<WorkspaceStatusAction.Options> options,
        Artifact stableStatus,
        Artifact volatileStatus) {
      super(BuildInfoHelper.BUILD_INFO_ACTION_OWNER, Artifact.NO_ARTIFACTS,
          ImmutableList.of(stableStatus, volatileStatus));
      this.options = options;
      this.stableStatus = stableStatus;
      this.volatileStatus = volatileStatus;
      this.username = System.getProperty("user.name");
      this.hostname = NetUtil.findShortHostName();
      this.timestamp = System.currentTimeMillis();
    }

    @Override
    public String describeStrategy(Executor executor) {
      return "";
    }

    @Override
    public void execute(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      try {
        Joiner joiner = Joiner.on('\n');
        String info =
            joiner.join(
                BuildInfo.BUILD_EMBED_LABEL + " " + options.get().embedLabel,
                BuildInfo.BUILD_HOST + " " + hostname,
                BuildInfo.BUILD_USER + " " + username);
        FileSystemUtils.writeContent(stableStatus.getPath(), info.getBytes(StandardCharsets.UTF_8));
        String volatileInfo = BuildInfo.BUILD_TIMESTAMP + " " + timestamp + "\n";

        FileSystemUtils.writeContent(
            volatileStatus.getPath(), volatileInfo.getBytes(StandardCharsets.UTF_8));
      } catch (IOException e) {
        throw new ActionExecutionException(e, this, true);
      }
    }

    // TODO(bazel-team): Add test for equals, add hashCode.
    @Override
    public boolean equals(Object o) {
      if (!(o instanceof BazelWorkspaceStatusAction)) {
        return false;
      }

      BazelWorkspaceStatusAction that = (BazelWorkspaceStatusAction) o;
      return this.stableStatus.equals(that.stableStatus)
          && this.volatileStatus.equals(that.volatileStatus);
    }

    @Override
    public int hashCode() {
      return Objects.hash(stableStatus, volatileStatus);
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
    public boolean executeUnconditionally() {
      return true;
    }

    @Override
    public boolean isVolatile() {
      return true;
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

      Artifact stableArtifact = factory.getDerivedArtifact(
          new PathFragment("stable-status.txt"), root, artifactOwner);
      Artifact volatileArtifact = factory.getConstantMetadataArtifact(
          new PathFragment("volatile-status.txt"), root, artifactOwner);

      return new BazelWorkspaceStatusAction(options, stableArtifact, volatileArtifact);
    }
  }

  @ExecutionStrategy(contextType = WorkspaceStatusAction.Context.class)
  private class BazelWorkspaceStatusActionContext implements WorkspaceStatusAction.Context {
    @Override
    public ImmutableMap<String, Key> getStableKeys() {
      return ImmutableMap.of(
          BuildInfo.BUILD_EMBED_LABEL,
          Key.of(KeyType.STRING, options.get().embedLabel, "redacted"),
          BuildInfo.BUILD_HOST,
          Key.of(KeyType.STRING, "hostname", "redacted"),
          BuildInfo.BUILD_USER,
          Key.of(KeyType.STRING, "username", "redacted"));
    }

    @Override
    public ImmutableMap<String, Key> getVolatileKeys() {
      return ImmutableMap.of(BuildInfo.BUILD_TIMESTAMP, Key.of(KeyType.INTEGER, "0", "0"));
    }
  }

  private BlazeRuntime runtime;
  private AtomicReference<WorkspaceStatusAction.Options> options = new AtomicReference<>();

  @Override
  public void beforeCommand(BlazeRuntime runtime, Command command) {
    this.runtime = runtime;
    runtime.getEventBus().register(this);
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return command.builds()
        ? ImmutableList.<Class<? extends OptionsBase>>of(WorkspaceStatusAction.Options.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Subscribe
  public void gotOptionsEvent(GotOptionsEvent event) {
    options.set(event.getOptions().getOptions(WorkspaceStatusAction.Options.class));
  }

  @Override
  public Iterable<ActionContextProvider> getActionContextProviders() {
    return SimpleActionContextProvider.of(new BazelWorkspaceStatusActionContext());
  }

  @Override
  public WorkspaceStatusAction.Factory getWorkspaceStatusActionFactory() {
    return new BazelStatusActionFactory();
  }
}
