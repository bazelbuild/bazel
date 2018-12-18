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
package com.google.devtools.build.lib.bazel;

import static com.google.common.base.StandardSystemProperty.USER_NAME;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.stream.Collectors.joining;

import com.google.common.base.Preconditions;
import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BuildInfo;
import com.google.devtools.build.lib.analysis.BuildInfoEvent;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction.Key;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction.KeyType;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.GotOptionsEvent;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;
import com.google.devtools.build.lib.shell.BadExitStatusException;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.CommandBuilder;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.NetUtil;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsBase;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.TreeMap;

/**
 * Provides information about the workspace (e.g. source control context, current machine, current
 * user, etc).
 *
 * <p>Note that the <code>equals()</code> method is necessary so that Skyframe knows when to
 * invalidate the node representing the workspace status action.
 */
public class BazelWorkspaceStatusModule extends BlazeModule {
  @AutoCodec
  @AutoCodec.VisibleForSerialization
  static class BazelWorkspaceStatusAction extends WorkspaceStatusAction {
    private final Artifact stableStatus;
    private final Artifact volatileStatus;
    private final Supplier<Options> options;
    private final String username;
    private final String hostname;
    private final Supplier<ImmutableMap<String, String>> clientEnv;

    @SuppressWarnings("unused") // Read by serialization.
    private final Path workspace;

    @AutoCodec.VisibleForSerialization
    BazelWorkspaceStatusAction(
        Supplier<Options> options,
        Supplier<ImmutableMap<String, String>> clientEnv,
        Path workspace,
        Artifact stableStatus,
        Artifact volatileStatus,
        String hostname) {
      super(
          ActionOwner.SYSTEM_ACTION_OWNER,
          Artifact.NO_ARTIFACTS,
          ImmutableList.of(stableStatus, volatileStatus));
      this.options = Preconditions.checkNotNull(options);
      this.stableStatus = stableStatus;
      this.volatileStatus = volatileStatus;
      this.username = USER_NAME.value();
      this.hostname = hostname;
      this.clientEnv = clientEnv;
      this.workspace = workspace;
    }

    private com.google.devtools.build.lib.shell.Command getGetWorkspaceStatusCommand(
        Options options, ImmutableMap<String, String> clientEnv) {
      return options.workspaceStatusCommand.equals(PathFragment.EMPTY_FRAGMENT)
          ? null
          : new CommandBuilder()
              .addArgs(options.workspaceStatusCommand.toString())
              // Pass client env, because certain SCM client(like
              // perforce, git) relies on environment variables to work
              // correctly.
              .setEnv(clientEnv)
              .setWorkingDir(workspace)
              .useShell(true)
              .build();
    }

    private String getAdditionalWorkspaceStatus(
        Options options,
        ImmutableMap<String, String> clientEnv,
        ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      com.google.devtools.build.lib.shell.Command getWorkspaceStatusCommand =
          getGetWorkspaceStatusCommand(options, clientEnv);
      try {
        if (getWorkspaceStatusCommand != null) {
          actionExecutionContext
              .getEventHandler()
              .handle(
                  Event.progress(
                      "Getting additional workspace status by running "
                          + options.workspaceStatusCommand));
          CommandResult result = getWorkspaceStatusCommand.execute();
          if (result.getTerminationStatus().success()) {
            return new String(result.getStdout(), UTF_8);
          }
          throw new BadExitStatusException(
              getWorkspaceStatusCommand,
              result,
              "workspace status command failed: " + result.getTerminationStatus());
        }
      } catch (BadExitStatusException e) {
        String errorMessage = e.getMessage();
        try {
          actionExecutionContext.getFileOutErr().getOutputStream().write(
              e.getResult().getStdout());
          actionExecutionContext.getFileOutErr().getErrorStream().write(e.getResult().getStderr());
        } catch (IOException e2) {
          errorMessage = errorMessage + " and could not get stdout/stderr: " + e2.getMessage();
        }
        throw new ActionExecutionException(errorMessage, e, this, true);
      } catch (CommandException e) {
        throw new ActionExecutionException(e, this, true);
      }
      return "";
    }

    private static boolean isStableKey(String key) {
        return key.startsWith("STABLE_");
    }

    private static Map<String, String> parseWorkspaceStatus(String input) {
      TreeMap<String, String> result = new TreeMap<>();
      for (String line : input.trim().split("\n")) {
        String[] splitLine = line.split(" ", 2);
        if (splitLine.length >= 2) {
          result.put(splitLine[0], splitLine[1]);
        }
      }

      return result;
    }

    private static byte[] printStatusMap(Map<String, String> map) {
      String s =
          map.entrySet()
              .stream()
              .map(entry -> entry.getKey() + " " + entry.getValue())
              .collect(joining("\n"));
      s += "\n";
      return s.getBytes(StandardCharsets.UTF_8);
    }

    @Override
    public void prepare(FileSystem fileSystem, Path execRoot) throws IOException {
      // The default implementation of this method deletes all output files; override it to keep
      // the old stableStatus around. This way we can reuse the existing file (preserving its mtime)
      // if the contents haven't changed.
      deleteOutput(fileSystem, volatileStatus);
    }

    @Override
    public ActionResult execute(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      Options options = this.options.get();
      ImmutableMap<String, String> clientEnv = this.clientEnv.get();
      try {
        Map<String, String> statusMap =
            parseWorkspaceStatus(
                getAdditionalWorkspaceStatus(options, clientEnv, actionExecutionContext));
        Map<String, String> volatileMap = new TreeMap<>();
        Map<String, String> stableMap = new TreeMap<>();

        for (Map.Entry<String, String> entry : statusMap.entrySet()) {
          if (isStableKey(entry.getKey())) {
            stableMap.put(entry.getKey(), entry.getValue());
          } else {
            volatileMap.put(entry.getKey(), entry.getValue());
          }
        }

        stableMap.put(BuildInfo.BUILD_EMBED_LABEL, options.embedLabel);
        stableMap.put(BuildInfo.BUILD_HOST, hostname);
        stableMap.put(BuildInfo.BUILD_USER, username);
        volatileMap.put(
            BuildInfo.BUILD_TIMESTAMP, Long.toString(getCurrentTimeMillis(clientEnv) / 1000));

        Map<String, String> overallMap = new TreeMap<>();
        overallMap.putAll(volatileMap);
        overallMap.putAll(stableMap);
        actionExecutionContext.getEventHandler().post(new BuildInfoEvent(overallMap));

        // Only update the stableStatus contents if they are different than what we have on disk.
        // This is to preserve the old file's mtime so that we do not generate an unnecessary dirty
        // file on each incremental build.
        FileSystemUtils.maybeUpdateContent(
            actionExecutionContext.getInputPath(stableStatus), printStatusMap(stableMap));

        // Contrary to the stableStatus, write the contents of volatileStatus unconditionally
        // because we know it will be different. This output file is marked as "constant metadata"
        // so its dirtiness will be ignored anyway.
        FileSystemUtils.writeContent(
            actionExecutionContext.getInputPath(volatileStatus), printStatusMap(volatileMap));
      } catch (IOException e) {
        throw new ActionExecutionException(
            "Failed to run workspace status command " + options.workspaceStatusCommand,
            e,
            this,
            true);
      }
      return ActionResult.EMPTY;
    }

    /**
     * This method returns the current time for stamping, using SOURCE_DATE_EPOCH
     * (https://reproducible-builds.org/specs/source-date-epoch/) if provided.
     */
    private static long getCurrentTimeMillis(ImmutableMap<String, String> clientEnv) {
      if (clientEnv.containsKey("SOURCE_DATE_EPOCH")) {
        String value = clientEnv.get("SOURCE_DATE_EPOCH").trim();
        if (!value.isEmpty()) {
          try {
            return Long.parseLong(value) * 1000;
          } catch (NumberFormatException ex) {
            // Fall-back to use the current time if SOURCE_DATE_EPOCH is not a long.
          }
        }
      }
      return System.currentTimeMillis();
    }

    @Override
    public String getMnemonic() {
      return "BazelWorkspaceStatusAction";
    }

    @Override
    protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {}

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
        ArtifactFactory factory, ArtifactOwner artifactOwner, String workspaceName) {
      ArtifactRoot root = env.getDirectories().getBuildDataDirectory(workspaceName);

      Artifact stableArtifact = factory.getDerivedArtifact(
          PathFragment.create("stable-status.txt"), root, artifactOwner);
      Artifact volatileArtifact = factory.getConstantMetadataArtifact(
          PathFragment.create("volatile-status.txt"), root, artifactOwner);

      return new BazelWorkspaceStatusAction(
          () -> options,
          () -> ImmutableMap.copyOf(env.getClientEnv()),
          env.getDirectories().getWorkspace(),
          stableArtifact,
          volatileArtifact,
          getHostname());
    }
  }

  /**
   * Returns cached short hostname.
   *
   * <p>Hostname lookup performs reverse DNS lookup which in bad cases can take seconds. To speed up
   * builds we only lookup hostname once and cache the result. Therefore if the hostname changes
   * during bazel server lifetime, bazel will not see the change.
   */
  private static String getHostname() {
    return NetUtil.getCachedShortHostName();
  }

  @ExecutionStrategy(contextType = WorkspaceStatusAction.Context.class)
  private static final class BazelWorkspaceStatusActionContext
      implements WorkspaceStatusAction.Context {
    private final WorkspaceStatusAction.Options options;

    private BazelWorkspaceStatusActionContext(WorkspaceStatusAction.Options options) {
      this.options = options;
    }

    @Override
    public ImmutableMap<String, Key> getStableKeys() {
      ImmutableMap.Builder<String, Key> builder = ImmutableMap.builder();
      builder.put(
          BuildInfo.BUILD_EMBED_LABEL, Key.of(KeyType.STRING, options.embedLabel, "redacted"));
      builder.put(BuildInfo.BUILD_HOST, Key.of(KeyType.STRING, "hostname", "redacted"));
      builder.put(BuildInfo.BUILD_USER, Key.of(KeyType.STRING, "username", "redacted"));
      return builder.build();
    }

    @Override
    public ImmutableMap<String, Key> getVolatileKeys() {
      return ImmutableMap.of(
          BuildInfo.BUILD_TIMESTAMP,
          Key.of(KeyType.INTEGER, "0", "0"),
          BuildInfo.BUILD_SCM_REVISION,
          Key.of(KeyType.STRING, "0", "0"),
          BuildInfo.BUILD_SCM_STATUS,
          Key.of(KeyType.STRING, "", "redacted"));
    }
  }

  private CommandEnvironment env;
  private WorkspaceStatusAction.Options options;

  @Override
  public void beforeCommand(CommandEnvironment env) {
    this.env = env;
    env.getEventBus().register(this);
  }

  @Override
  public void afterCommand() {
    this.env = null;
    this.options = null;
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.<Class<? extends OptionsBase>>of(WorkspaceStatusAction.Options.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Subscribe
  public void gotOptionsEvent(GotOptionsEvent event) {
    options = event.getOptions().getOptions(WorkspaceStatusAction.Options.class);
  }

  @Override
  public void workspaceInit(
      BlazeRuntime runtime, BlazeDirectories directories, WorkspaceBuilder builder) {
    builder.setWorkspaceStatusActionFactory(new BazelStatusActionFactory());
  }

  @Override
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder) {
    builder.addActionContext(new BazelWorkspaceStatusActionContext(options));
  }

}
