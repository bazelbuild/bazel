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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BuildInfo;
import com.google.devtools.build.lib.analysis.BuildInfoEvent;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction.Key;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction.KeyType;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.ModuleActionContextRegistry;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.WorkspaceStatus;
import com.google.devtools.build.lib.server.FailureDetails.WorkspaceStatus.Code;
import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.BadExitStatusException;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.util.CommandBuilder;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.NetUtil;
import com.google.devtools.build.lib.vfs.BulkDeleter;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsBase;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * Provides information about the workspace (e.g. source control context, current machine, current
 * user, etc).
 *
 * <p>Note that the <code>equals()</code> method is necessary so that Skyframe knows when to
 * invalidate the node representing the workspace status action.
 */
public class BazelWorkspaceStatusModule extends BlazeModule {
  static class BazelWorkspaceStatusAction extends WorkspaceStatusAction {
    private final Artifact stableStatus;
    private final Artifact volatileStatus;
    private final String username;
    private final String hostname;

    BazelWorkspaceStatusAction(
        Artifact stableStatus, Artifact volatileStatus, String username, String hostname) {
      super(
          ActionOwner.SYSTEM_ACTION_OWNER,
          NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          ImmutableSet.of(stableStatus, volatileStatus),
          "workspace status");
      this.stableStatus = stableStatus;
      this.volatileStatus = volatileStatus;
      this.username = username;
      this.hostname = hostname;
    }

    private String getAdditionalWorkspaceStatus(
        Options options, ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException, InterruptedException {
      com.google.devtools.build.lib.shell.Command getWorkspaceStatusCommand =
          actionExecutionContext.getContext(WorkspaceStatusAction.Context.class).getCommand();
      try {
        if (getWorkspaceStatusCommand != null) {
          actionExecutionContext
              .getEventHandler()
              .handle(
                  Event.progress(
                      "Getting additional workspace status by running "
                          + options.workspaceStatusCommand));
          ByteArrayOutputStream stdoutStream = new ByteArrayOutputStream();
          try (OutputStream errStream =
              actionExecutionContext.getFileOutErr().getErrorPath().getOutputStream()) {
            getWorkspaceStatusCommand.execute(stdoutStream, errStream);
          } catch (IOException e) {
            throw createExecutionException(e, Code.STDERR_IO_EXCEPTION);
          }
          return new String(stdoutStream.toByteArray(), UTF_8);
        }
      } catch (BadExitStatusException e) {
        throw createExecutionException(e, Code.NON_ZERO_EXIT);
      } catch (AbnormalTerminationException e) {
        throw createExecutionException(e, Code.ABNORMAL_TERMINATION);
      } catch (CommandException e) {
        throw createExecutionException(e, Code.EXEC_FAILED);
      }
      return "";
    }

    private static final ImmutableSet<String> SPECIAL_STABLE_KEYS =
        ImmutableSet.of(BuildInfo.BUILD_EMBED_LABEL, BuildInfo.BUILD_HOST, BuildInfo.BUILD_USER);

    private static boolean isStableKey(String key) {
      return key.startsWith("STABLE_") || SPECIAL_STABLE_KEYS.contains(key);
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
    public void prepare(
        Path execRoot, ArtifactPathResolver pathResolver, @Nullable BulkDeleter bulkDeleter)
        throws IOException {
      // The default implementation of this method deletes all output files; override it to keep
      // the old stableStatus around. This way we can reuse the existing file (preserving its mtime)
      // if the contents haven't changed.
      deleteOutput(pathResolver.toPath(volatileStatus), volatileStatus.getRoot());
    }

    @Override
    public ActionResult execute(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException, InterruptedException {
      WorkspaceStatusAction.Context context =
          actionExecutionContext.getContext(WorkspaceStatusAction.Context.class);
      Options options = context.getOptions();
      ImmutableMap<String, String> clientEnv = context.getClientEnv();
      Map<String, String> volatileMap = new TreeMap<>();
      Map<String, String> stableMap = new TreeMap<>();

      stableMap.put(BuildInfo.BUILD_EMBED_LABEL, options.embedLabel);
      stableMap.put(BuildInfo.BUILD_HOST, hostname);
      stableMap.put(BuildInfo.BUILD_USER, username);
      volatileMap.put(
          BuildInfo.BUILD_TIMESTAMP, Long.toString(getCurrentTimeMillis(clientEnv) / 1000));
      try {
        Map<String, String> statusMap =
            parseWorkspaceStatus(getAdditionalWorkspaceStatus(options, actionExecutionContext));
        for (Map.Entry<String, String> entry : statusMap.entrySet()) {
          if (isStableKey(entry.getKey())) {
            stableMap.put(entry.getKey(), entry.getValue());
          } else {
            volatileMap.put(entry.getKey(), entry.getValue());
          }
        }

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
        String message =
            String.format(
                "Failed to run workspace status command %s: %s",
                options.workspaceStatusCommand, e.getMessage());
        DetailedExitCode code = createDetailedCode(message, Code.CONTENT_UPDATE_IO_EXCEPTION);
        throw new ActionExecutionException(message, e, this, true, code);
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
    protected void computeKey(
        ActionKeyContext actionKeyContext,
        @Nullable Artifact.ArtifactExpander artifactExpander,
        Fingerprint fp) {}

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

  private static DetailedExitCode createDetailedCode(String message, Code detailedCode) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setWorkspaceStatus(WorkspaceStatus.newBuilder().setCode(detailedCode))
            .build());
  }

  private static class BazelStatusActionFactory implements WorkspaceStatusAction.Factory {
    @Override
    public Map<String, String> createDummyWorkspaceStatus(
        WorkspaceStatusAction.DummyEnvironment env) {
      return ImmutableMap.of();
    }

    @Override
    public WorkspaceStatusAction createWorkspaceStatusAction(
        WorkspaceStatusAction.Environment env) {
      Artifact stableArtifact = env.createStableArtifact("stable-status.txt");
      Artifact volatileArtifact = env.createVolatileArtifact("volatile-status.txt");
      return new BazelWorkspaceStatusAction(
          stableArtifact, volatileArtifact, USER_NAME.value(), NetUtil.getCachedShortHostName());
    }
  }

  private static final class BazelWorkspaceStatusActionContext
      implements WorkspaceStatusAction.Context {
    private final CommandEnvironment env;

    private BazelWorkspaceStatusActionContext(CommandEnvironment env) {
      this.env = env;
    }

    @Override
    public ImmutableMap<String, Key> getStableKeys() {
      WorkspaceStatusAction.Options options =
          env.getOptions().getOptions(WorkspaceStatusAction.Options.class);
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

    @Override
    public WorkspaceStatusAction.Options getOptions() {
      return env.getOptions().getOptions(WorkspaceStatusAction.Options.class);
    }

    @Override
    public ImmutableMap<String, String> getClientEnv() {
      return ImmutableMap.copyOf(env.getClientEnv());
    }

    @Override
    public com.google.devtools.build.lib.shell.Command getCommand() {
      WorkspaceStatusAction.Options options =
          env.getOptions().getOptions(WorkspaceStatusAction.Options.class);
      return options.workspaceStatusCommand.equals(PathFragment.EMPTY_FRAGMENT)
          ? null
          : new CommandBuilder()
              .addArgs(options.workspaceStatusCommand.toString())
              // Pass client env to allow SCM clients (like git) relying on environment variables to
              // work correctly.
              .setEnv(env.getClientEnv())
              .setWorkingDir(env.getWorkspace())
              .useShell(true)
              .build();
    }
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.<Class<? extends OptionsBase>>of(WorkspaceStatusAction.Options.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Override
  public void workspaceInit(
      BlazeRuntime runtime, BlazeDirectories directories, WorkspaceBuilder builder) {
    builder.setWorkspaceStatusActionFactory(new BazelStatusActionFactory());
  }

  @Override
  public void registerActionContexts(
      ModuleActionContextRegistry.Builder registryBuilder,
      CommandEnvironment env,
      BuildRequest buildRequest) {
    registryBuilder.register(
        WorkspaceStatusAction.Context.class, new BazelWorkspaceStatusActionContext(env));
  }
}
