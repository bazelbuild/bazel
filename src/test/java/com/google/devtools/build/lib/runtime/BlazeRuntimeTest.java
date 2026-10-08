// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.util.io.CommandExtensionReporter.NO_OP_COMMAND_EXTENSION_REPORTER;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.compress.CompressionServiceImpl;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.RunfilesTreeUpdater;
import com.google.devtools.build.lib.runtime.BlazeWorkspace.ActionCacheGarbageCollectorIdleTask;
import com.google.devtools.build.lib.runtime.commands.VersionCommand;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.server.FailureDetails.Crash;
import com.google.devtools.build.lib.server.FailureDetails.Crash.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.GcAndInternerShrinkingIdleTask;
import com.google.devtools.build.lib.server.IdleTask;
import com.google.devtools.build.lib.server.InstallBaseGarbageCollectorIdleTask;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.protobuf.BytesValue;
import com.google.protobuf.StringValue;
import java.io.FileDescriptor;
import java.time.Duration;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BlazeRuntime} static methods. */
@RunWith(JUnit4.class)
public class BlazeRuntimeTest {

  private final ManualClock clock = new ManualClock();
  private final FileSystem fs = new InMemoryFileSystem(clock, DigestHashFunction.SHA256);
  private final ServerDirectories serverDirectories =
      new ServerDirectories(
          fs.getPath("/install"), fs.getPath("/output"), fs.getPath("/output_user"));
  private final BlazeDirectories blazeDirectories =
      new BlazeDirectories(serverDirectories, fs.getPath("/workspace"), "blaze");
  private final OptionsParser optionsParser =
      OptionsParser.builder()
          .optionsClasses(
              ImmutableList.of(
                  CommonCommandOptions.class,
                  KeepStateAfterBuildOption.class,
                  ClientOptions.class,
                  BuildRequestOptions.class))
          .build();
  private final Thread commandThread = mock(Thread.class);
  private final AtomicReference<String> shutdownReason = new AtomicReference<>();

  @Test
  public void manageProfiles() throws Exception {
    var dir = fs.getPath("/output");
    dir.createDirectory();
    dir.getChild("foo").createDirectory();
    dir.getChild("bar").getOutputStream().close();
    clock.advanceMillis(10);
    var p1 = BlazeRuntime.manageProfiles(dir, "p1", 3);
    assertThat(p1.getBaseName()).isEqualTo("command-p1.profile.gz");
    p1.getOutputStream().close();
    clock.advanceMillis(10);
    var p2 = BlazeRuntime.manageProfiles(dir, "p2", 3);
    assertThat(p2.getBaseName()).isEqualTo("command-p2.profile.gz");
    p2.getOutputStream().close();
    clock.advanceMillis(10);
    var p3 = BlazeRuntime.manageProfiles(dir, "p3", 3);
    assertThat(p3.getBaseName()).isEqualTo("command-p3.profile.gz");
    p3.getOutputStream().close();
    clock.advanceMillis(10);
    var p4 = BlazeRuntime.manageProfiles(dir, "p4", 3);
    assertThat(p4.getBaseName()).isEqualTo("command-p4.profile.gz");
    p4.getOutputStream().close();
    assertThat(dir.readdir(Symlinks.FOLLOW).stream().map(Dirent::getName))
        .containsExactly(
            "foo",
            "bar",
            "command-p2.profile.gz",
            "command-p3.profile.gz",
            "command-p4.profile.gz");
    clock.advanceMillis(10);
    var p5 = BlazeRuntime.manageProfiles(dir, "p5", 1);
    assertThat(p5.getBaseName()).isEqualTo("command-p5.profile.gz");
    p5.getOutputStream().close();
    assertThat(dir.readdir(Symlinks.FOLLOW).stream().map(Dirent::getName))
        .containsExactly("foo", "bar", "command-p5.profile.gz");
  }

  @Test
  public void optionSplitting() {
    BlazeRuntime.CommandLineOptions options =
        BlazeRuntime.splitStartupOptions(
            ImmutableList.of(),
            "--install_base=/foo --host_jvm_args=-Xmx1B",
            "build",
            "//foo:bar",
            "--nobuild");
    assertThat(options.getStartupArgs())
        .containsExactly("--install_base=/foo --host_jvm_args=-Xmx1B");
    assertThat(options.getOtherArgs()).isEqualTo(Arrays.asList("build", "//foo:bar", "--nobuild"));
  }

  // A regression test to make sure that the 'no' prefix is handled correctly.
  @Test
  public void optionSplittingNoPrefix() {
    BlazeRuntime.CommandLineOptions options =
        BlazeRuntime.splitStartupOptions(ImmutableList.of(), "--nobatch", "build");
    assertThat(options.getStartupArgs()).containsExactly("--nobatch");
    assertThat(options.getOtherArgs()).containsExactly("build");
  }

  @Test
  public void crashTest() throws Exception {
    BlazeRuntime runtime = createRuntime();
    CommandEnvironment env = createCommandEnvironment(runtime);
    runtime.beforeCommand(env, optionsParser.getOptions(CommonCommandOptions.class));
    DetailedExitCode oom =
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setCrash(Crash.newBuilder().setCode(Code.CRASH_OOM))
                .build());
    runtime.cleanUpForCrash(oom);
    BlazeCommandResult mainThreadCrash =
        BlazeCommandResult.failureDetail(
            FailureDetail.newBuilder()
                .setCrash(Crash.newBuilder().setCode(Code.CRASH_UNKNOWN))
                .build());
    assertThat(
            runtime
                .afterCommand(/* forceKeepStateForTesting= */ false, env, mainThreadCrash)
                .getDetailedExitCode())
        .isEqualTo(oom);
    // Confirm that runtime interrupted the command thread.
    verify(commandThread).interrupt();
    assertThat(shutdownReason.get()).isEqualTo("foo product is crashing: ");
  }

  @Test
  public void addsResponseExtensions() throws Exception {
    BlazeRuntime runtime = createRuntime();
    CommandEnvironment env = createCommandEnvironment(runtime);
    Any anyFoo = Any.pack(StringValue.of("foo"));
    Any anyBar = Any.pack(BytesValue.of(ByteString.copyFromUtf8("bar")));
    env.addResponseExtensions(ImmutableList.of(anyFoo, anyBar));
    assertThat(
            runtime
                .afterCommand(
                    /* forceKeepStateForTesting= */ false, env, BlazeCommandResult.success())
                .getResponseExtensions())
        .containsExactly(anyFoo, anyBar);
  }

  @Test
  public void addsGcAndInternerShrinkingIdleTask_noStateKeptAfterBuild() throws Exception {
    BlazeRuntime runtime = createRuntime();
    optionsParser.parse("--nokeep_state_after_build");
    CommandEnvironment env = createCommandEnvironment(runtime);
    CommonCommandOptions options = Options.getDefaults(CommonCommandOptions.class);
    runtime.beforeCommand(env, options);

    ImmutableList<IdleTask> gcIdleTasks =
        env.getIdleTasks().stream()
            .filter(t -> t instanceof GcAndInternerShrinkingIdleTask)
            .collect(toImmutableList());
    assertThat(gcIdleTasks).hasSize(1);
    var idleTask = (GcAndInternerShrinkingIdleTask) gcIdleTasks.get(0);
    assertThat(idleTask.delay()).isEqualTo(Duration.ZERO);
  }

  @Test
  public void addsGcAndInternerShrinkingIdleTask_stateKeptAfterBuild() throws Exception {
    BlazeRuntime runtime = createRuntime();
    optionsParser.parse("--keep_state_after_build");
    CommandEnvironment env = createCommandEnvironment(runtime);
    env.getOptions().getOptions(KeepStateAfterBuildOption.class).setKeepStateAfterBuild(true);
    CommonCommandOptions options = Options.getDefaults(CommonCommandOptions.class);

    runtime.beforeCommand(env, options);

    ImmutableList<IdleTask> gcIdleTasks =
        env.getIdleTasks().stream()
            .filter(t -> t instanceof GcAndInternerShrinkingIdleTask)
            .collect(toImmutableList());
    assertThat(gcIdleTasks).hasSize(1);
    var idleTask = (GcAndInternerShrinkingIdleTask) gcIdleTasks.get(0);
    assertThat(idleTask.delay()).isGreaterThan(Duration.ZERO);
  }

  @Test
  public void runfilesTreeUpdater_distinctAcrossCommandsSharingWorkspace() throws Exception {
    // Two commands sharing one workspace must still get distinct updaters. This pins the command
    // scope against a potential regression of scoping the updater to the long-lived workspace:
    // its dedup map never evicts, so a reused updater would skip restaging a tree that changed
    // between commands.
    BlazeRuntime runtime =
        createRuntime(
            ImmutableList.of(
                new BlazeModule() {
                  @Override
                  public void initializeRuleClasses(ConfiguredRuleClassProvider.Builder builder) {
                    builder.setRunfilesPrefix("_main");
                  }
                }),
            ImmutableList.of());
    BlazeWorkspace workspace =
        runtime.initWorkspace(blazeDirectories, BinTools.empty(blazeDirectories));
    CommandEnvironment firstEnv = createCommandEnvironment(runtime, workspace);
    CommandEnvironment secondEnv = createCommandEnvironment(runtime, workspace);

    RunfilesTreeUpdater firstUpdater = firstEnv.getRunfilesTreeUpdater();
    assertThat(firstUpdater).isNotNull();
    assertThat(firstEnv.getRunfilesTreeUpdater()).isSameInstanceAs(firstUpdater);
    assertThat(secondEnv.getRunfilesTreeUpdater()).isNotSameInstanceAs(firstUpdater);
  }

  @Test
  public void doesNotAddInstallBaseGcIdleTaskWhenDisabled() throws Exception {
    BlazeRuntime runtime = createRuntime();
    CommandEnvironment env = createCommandEnvironment(runtime);
    CommonCommandOptions options = Options.getDefaults(CommonCommandOptions.class);
    options.setInstallBaseGcMaxAge(Duration.ZERO);

    runtime.beforeCommand(env, options);

    ImmutableList<IdleTask> gcIdleTasks =
        env.getIdleTasks().stream()
            .filter(t -> t instanceof InstallBaseGarbageCollectorIdleTask)
            .collect(toImmutableList());
    assertThat(gcIdleTasks).isEmpty();
  }

  @Test
  public void addsInstallBaseGcIdleTaskWhenEnabled() throws Exception {
    BlazeRuntime runtime = createRuntime();
    CommandEnvironment env = createCommandEnvironment(runtime);
    CommonCommandOptions options = Options.getDefaults(CommonCommandOptions.class);
    options.setInstallBaseGcMaxAge(Duration.ofDays(365));

    runtime.beforeCommand(env, options);

    ImmutableList<IdleTask> gcIdleTasks =
        env.getIdleTasks().stream()
            .filter(t -> t instanceof InstallBaseGarbageCollectorIdleTask)
            .collect(toImmutableList());
    assertThat(gcIdleTasks).hasSize(1);
    var idleTask = (InstallBaseGarbageCollectorIdleTask) gcIdleTasks.get(0);
    assertThat(idleTask.delay()).isEqualTo(Duration.ZERO);
    assertThat(idleTask.getGarbageCollector().getRoot())
        .isEqualTo(blazeDirectories.getInstallBase().getParentDirectory());
    assertThat(idleTask.getGarbageCollector().getOwnInstallBase())
        .isEqualTo(blazeDirectories.getInstallBase());
    assertThat(idleTask.getGarbageCollector().getMaxAge()).isEqualTo(Duration.ofDays(365));
  }

  @Test
  public void doesNotAddActionCacheGcIdleTaskWhenDisabled() throws Exception {
    BlazeRuntime runtime = createRuntime();
    CommandEnvironment env = createCommandEnvironment(runtime);
    CommonCommandOptions options = Options.getDefaults(CommonCommandOptions.class);
    options.setActionCacheGcMaxAge(Duration.ZERO);
    options.setActionCacheGcIdleDelay(Duration.ofMinutes(5));
    options.setActionCacheGcThreshold(10);

    runtime.beforeCommand(env, options);

    ImmutableList<IdleTask> gcIdleTasks =
        env.getIdleTasks().stream()
            .filter(t -> t instanceof ActionCacheGarbageCollectorIdleTask)
            .collect(toImmutableList());
    assertThat(gcIdleTasks).isEmpty();
  }

  @Test
  public void addsActionCacheGcIdleTaskWhenEnabled() throws Exception {
    BlazeRuntime runtime = createRuntime();
    CommandEnvironment env = createCommandEnvironment(runtime);
    CommonCommandOptions options = Options.getDefaults(CommonCommandOptions.class);
    options.setActionCacheGcMaxAge(Duration.ofDays(7));
    options.setActionCacheGcIdleDelay(Duration.ofMinutes(5));
    options.setActionCacheGcThreshold(10);

    runtime.beforeCommand(env, options);

    ImmutableList<IdleTask> gcIdleTasks =
        env.getIdleTasks().stream()
            .filter(t -> t instanceof ActionCacheGarbageCollectorIdleTask)
            .collect(toImmutableList());
    assertThat(gcIdleTasks).hasSize(1);
    var idleTask = (ActionCacheGarbageCollectorIdleTask) gcIdleTasks.get(0);
    assertThat(idleTask.delay()).isEqualTo(Duration.ofMinutes(5));
    assertThat(idleTask.getThreshold()).isEqualTo(0.1f);
    assertThat(idleTask.getMaxAge()).isEqualTo(Duration.ofDays(7));
  }

  @Test
  public void addsIdleTasksFromModules() throws Exception {
    BlazeRuntime runtime = createRuntime();
    CommandEnvironment env = createCommandEnvironment(runtime);
    IdleTask fooTask =
        new IdleTask() {
          @Override
          public String displayName() {
            return "foo";
          }

          @Override
          public void run() {}
        };
    IdleTask barTask =
        new IdleTask() {
          @Override
          public String displayName() {
            return "bar";
          }

          @Override
          public void run() {}
        };
    env.addIdleTask(fooTask);
    env.addIdleTask(barTask);
    assertThat(
            runtime
                .afterCommand(
                    /* forceKeepStateForTesting= */ false, env, BlazeCommandResult.success())
                .getIdleTasks())
        .containsAtLeast(fooTask, barTask);
  }

  @Test
  public void addsCommandsFromModules() throws Exception {
    BlazeRuntime runtime =
        createRuntime(
            ImmutableList.of(new FooCommandModule(), new BarCommandModule()), ImmutableList.of());

    assertThat(runtime.getCommandMap().keySet()).containsExactly("foo", "bar").inOrder();
    assertThat(runtime.getCommandMap().get("foo")).isInstanceOf(FooCommandModule.FooCommand.class);
    assertThat(runtime.getCommandMap().get("bar")).isInstanceOf(BarCommandModule.BarCommand.class);
  }

  @Test
  public void returnsBothModulesAndServicesAsOptionsSuppliers() throws Exception {
    var module = new BlazeModule() {};
    var service = new BlazeService() {};

    BlazeRuntime runtime = createRuntime(ImmutableList.of(module), ImmutableList.of(service));

    assertThat(runtime.getOptionsSuppliers()).containsAtLeast(module, service);
  }

  @Test
  public void deleteOutputBase_doesNotDeleteWhenWorkspaceIsNull() throws Exception {
    Path outputBase = blazeDirectories.getOutputBase();
    outputBase.createDirectoryAndParents();
    Path dummyFile = outputBase.getChild("dummy.txt");
    FileSystemUtils.writeIsoLatin1(dummyFile, "content");

    BlazeRuntime uninitializedRuntime = createRuntime();
    assertThat(uninitializedRuntime.getWorkspace()).isNull();

    uninitializedRuntime.deleteOutputBase();

    assertThat(outputBase.exists()).isTrue();
    assertThat(dummyFile.exists()).isTrue();
  }

  @Test
  public void deleteOutputBase_deletesDirectoryAndContents() throws Exception {
    Path outputBase = blazeDirectories.getOutputBase();
    outputBase.createDirectoryAndParents();
    Path dummyFile = outputBase.getChild("dummy.txt");
    FileSystemUtils.writeIsoLatin1(dummyFile, "content");
    Path subDir = outputBase.getChild("subdir");
    subDir.createDirectoryAndParents();
    FileSystemUtils.writeIsoLatin1(subDir.getChild("nested.txt"), "nested");

    BlazeRuntime runtime = createRuntime();
    runtime.initWorkspace(blazeDirectories, BinTools.empty(blazeDirectories));

    runtime.deleteOutputBase();

    assertThat(outputBase.exists()).isFalse();
  }

  @Test
  public void deleteOutputBase_removesDefaultConvenienceSymlinksWhenCommandRanWithoutOption()
      throws Exception {
    Path workspaceDir = blazeDirectories.getWorkspace();
    workspaceDir.createDirectoryAndParents();
    Path outputBase = blazeDirectories.getOutputBase();
    outputBase.createDirectoryAndParents();
    FileSystemUtils.writeIsoLatin1(outputBase.getChild("dummy.txt"), "content");

    BlazeRuntime runtime = createRuntime();
    CommandEnvironment env = createCommandEnvironment(runtime);
    runtime.beforeCommand(env, optionsParser.getOptions(CommonCommandOptions.class));

    Path symlink = workspaceDir.getChild(runtime.getProductName() + "-bin");
    symlink.createSymbolicLink(outputBase);
    assertThat(symlink.isSymbolicLink()).isTrue();

    runtime.deleteOutputBase();

    assertThat(outputBase.exists()).isFalse();
    assertThat(symlink.exists(Symlinks.NOFOLLOW)).isFalse();
  }

  @Test
  public void deleteOutputBase_removesCustomConvenienceSymlinksFromLastCommand() throws Exception {
    Path workspaceDir = blazeDirectories.getWorkspace();
    workspaceDir.createDirectoryAndParents();
    Path outputBase = blazeDirectories.getOutputBase();
    outputBase.createDirectoryAndParents();
    FileSystemUtils.writeIsoLatin1(outputBase.getChild("dummy.txt"), "content");

    BlazeRuntime runtime = createRuntime();
    BlazeWorkspace workspace =
        runtime.initWorkspace(blazeDirectories, BinTools.empty(blazeDirectories));
    optionsParser.parse("--symlink_prefix=custom-");
    CommandEnvironment buildEnv = createCommandEnvironment(runtime, workspace);
    runtime.beforeCommand(buildEnv, optionsParser.getOptions(CommonCommandOptions.class));

    // Simulate a subsequent non-build command (e.g. 'shutdown' or 'version') whose options do not
    // include BuildRequestOptions.
    OptionsParser nonBuildParser =
        OptionsParser.builder()
            .optionsClasses(
                ImmutableList.of(
                    CommonCommandOptions.class,
                    KeepStateAfterBuildOption.class,
                    ClientOptions.class))
            .build();
    CommandEnvironment shutdownEnv = createCommandEnvironment(runtime, workspace, nonBuildParser);
    runtime.beforeCommand(shutdownEnv, nonBuildParser.getOptions(CommonCommandOptions.class));

    Path customSymlink = workspaceDir.getChild("custom-bin");
    customSymlink.createSymbolicLink(outputBase);
    assertThat(customSymlink.isSymbolicLink()).isTrue();

    runtime.deleteOutputBase();

    assertThat(outputBase.exists()).isFalse();
    assertThat(customSymlink.exists(Symlinks.NOFOLLOW)).isFalse();
  }

  @Test
  public void deleteOutputBase_doesNotRemoveConvenienceSymlinksWhenNoCommandRan() throws Exception {
    Path workspaceDir = blazeDirectories.getWorkspace();
    workspaceDir.createDirectoryAndParents();
    Path outputBase = blazeDirectories.getOutputBase();
    outputBase.createDirectoryAndParents();
    FileSystemUtils.writeIsoLatin1(outputBase.getChild("dummy.txt"), "content");

    BlazeRuntime runtime = createRuntime();
    runtime.initWorkspace(blazeDirectories, BinTools.empty(blazeDirectories));

    Path symlink = workspaceDir.getChild(runtime.getProductName() + "-bin");
    symlink.createSymbolicLink(outputBase);
    assertThat(symlink.isSymbolicLink()).isTrue();

    runtime.deleteOutputBase();

    assertThat(outputBase.exists()).isFalse();
    assertThat(symlink.isSymbolicLink()).isTrue();
  }

  @Test
  public void deleteOutputBase_deletesReadOnlySubdirectory() throws Exception {
    Path outputBase = blazeDirectories.getOutputBase();
    Path subDir = outputBase.getChild("readonly");
    subDir.createDirectoryAndParents();
    FileSystemUtils.writeIsoLatin1(subDir.getChild("file.txt"), "content");
    subDir.setWritable(false);

    BlazeRuntime runtime = createRuntime();
    runtime.initWorkspace(blazeDirectories, BinTools.empty(blazeDirectories));

    runtime.deleteOutputBase();

    assertThat(outputBase.exists()).isFalse();
  }

  @Test
  public void deleteOutputBase_refusesToDeleteRootDirectory() throws Exception {
    Path rootDir = fs.getPath("/");
    ServerDirectories rootServerDirs =
        new ServerDirectories(fs.getPath("/install"), rootDir, fs.getPath("/output_user"));
    BlazeDirectories rootBlazeDirs =
        new BlazeDirectories(rootServerDirs, fs.getPath("/workspace"), "blaze");
    BlazeRuntime rootRuntime = createRuntime(rootServerDirs);
    rootRuntime.initWorkspace(rootBlazeDirs, BinTools.empty(rootBlazeDirs));

    rootRuntime.deleteOutputBase();

    assertThat(rootDir.exists()).isTrue();
  }

  @Test
  public void deleteOutputBase_handlesLongAndHyphenatedBasename() throws Exception {
    String longHyphenatedName = "-" + "a".repeat(150);
    Path longOb = fs.getPath("/output_user/" + longHyphenatedName);
    longOb.createDirectoryAndParents();
    FileSystemUtils.writeIsoLatin1(longOb.getChild("dummy.txt"), "content");

    ServerDirectories longServerDirs =
        new ServerDirectories(fs.getPath("/install"), longOb, fs.getPath("/output_user"));
    BlazeDirectories longBlazeDirs =
        new BlazeDirectories(longServerDirs, fs.getPath("/workspace"), "blaze");
    BlazeRuntime longRuntime = createRuntime(longServerDirs);
    longRuntime.initWorkspace(longBlazeDirs, BinTools.empty(longBlazeDirs));

    longRuntime.deleteOutputBase();

    assertThat(longOb.exists()).isFalse();
  }

  @Test
  public void deleteOutputBase_doesNotCloseStandardOutputInTest() throws Exception {
    Path outputBase = blazeDirectories.getOutputBase();
    outputBase.createDirectoryAndParents();
    BlazeRuntime runtime = createRuntime();
    runtime.initWorkspace(blazeDirectories, BinTools.empty(blazeDirectories));

    assertThat(FileDescriptor.out.valid()).isTrue();
    runtime.deleteOutputBase();
    assertThat(FileDescriptor.out.valid()).isTrue();
  }

  private BlazeRuntime createRuntime() throws Exception {
    return createRuntime(ImmutableList.of(), ImmutableList.of());
  }

  private BlazeRuntime createRuntime(ServerDirectories directories) throws Exception {
    return createRuntime(directories, ImmutableList.of(), ImmutableList.of());
  }

  private BlazeRuntime createRuntime(Iterable<BlazeModule> modules, Iterable<BlazeService> services)
      throws Exception {
    return createRuntime(serverDirectories, modules, services);
  }

  private BlazeRuntime createRuntime(
      ServerDirectories directories, Iterable<BlazeModule> modules, Iterable<BlazeService> services)
      throws Exception {
    var builder =
        new BlazeRuntime.Builder()
            .setFileSystem(fs)
            .setProductName("foo product")
            .setServerDirectories(directories)
            .setStartupOptionsProvider(mock(OptionsParsingResult.class))
            .addBlazeService(new CompressionServiceImpl());
    for (var module : modules) {
      builder.addBlazeModule(module);
    }
    for (var service : services) {
      builder.addBlazeService(service);
    }
    return builder.build();
  }

  private CommandEnvironment createCommandEnvironment(BlazeRuntime runtime) throws Exception {
    BlazeWorkspace workspace =
        runtime.initWorkspace(blazeDirectories, BinTools.empty(blazeDirectories));
    return createCommandEnvironment(runtime, workspace);
  }

  private CommandEnvironment createCommandEnvironment(
      BlazeRuntime runtime, BlazeWorkspace workspace) {
    return createCommandEnvironment(runtime, workspace, optionsParser);
  }

  private CommandEnvironment createCommandEnvironment(
      BlazeRuntime runtime, BlazeWorkspace workspace, OptionsParsingResult options) {
    return new CommandEnvironment(
        runtime,
        workspace,
        mock(EventBus.class),
        commandThread,
        VersionCommand.class.getAnnotation(Command.class),
        options,
        InvocationPolicy.getDefaultInstance(),
        /* packageLocator= */ null,
        SyscallCache.NO_CACHE,
        QuiescingExecutorsImpl.forTesting(),
        /* warnings= */ ImmutableList.of(),
        /* waitTimeInMs= */ 0L,
        /* commandStartTime= */ 0L,
        /* idleTaskResultsFromPreviousIdlePeriod= */ ImmutableList.of(),
        /* shutdownReasonConsumer= */ shutdownReason::set,
        /* commandExtensions= */ ImmutableList.of(),
        NO_OP_COMMAND_EXTENSION_REPORTER,
        /* attemptNumber= */ 1,
        /* buildRequestIdOverride= */ null,
        ConfigFlagDefinitions.NONE,
        new ResourceManager());
  }

  private static class FooCommandModule extends BlazeModule {
    @Command(name = "foo", shortDescription = "", help = "")
    private static class FooCommand implements BlazeCommand {

      @Override
      public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
        return null;
      }
    }

    @Override
    public void serverInit(OptionsParsingResult startupOptions, ServerBuilder builder) {
      builder.addCommands(new FooCommand());
    }
  }

  private static class BarCommandModule extends BlazeModule {
    @Command(name = "bar", shortDescription = "", help = "")
    private static class BarCommand implements BlazeCommand {

      @Override
      public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
        return null;
      }
    }

    @Override
    public void serverInit(OptionsParsingResult startupOptions, ServerBuilder builder) {
      builder.addCommands(new BarCommand());
    }
  }
}
