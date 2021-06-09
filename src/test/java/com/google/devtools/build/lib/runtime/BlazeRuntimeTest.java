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

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.verify;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.runtime.commands.VersionCommand;
import com.google.devtools.build.lib.server.FailureDetails.Crash;
import com.google.devtools.build.lib.server.FailureDetails.Crash.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.protobuf.BytesValue;
import com.google.protobuf.StringValue;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for {@link BlazeRuntime} static methods. */
@RunWith(JUnit4.class)
public class BlazeRuntimeTest {

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
        .isEqualTo(Arrays.asList("--install_base=/foo --host_jvm_args=-Xmx1B"));
    assertThat(options.getOtherArgs()).isEqualTo(Arrays.asList("build", "//foo:bar", "--nobuild"));
  }

  // A regression test to make sure that the 'no' prefix is handled correctly.
  @Test
  public void optionSplittingNoPrefix() {
    BlazeRuntime.CommandLineOptions options =
        BlazeRuntime.splitStartupOptions(ImmutableList.of(), "--nobatch", "build");
    assertThat(options.getStartupArgs()).isEqualTo(Arrays.asList("--nobatch"));
    assertThat(options.getOtherArgs()).isEqualTo(Arrays.asList("build"));
  }

  private static final ImmutableList<Class<? extends OptionsBase>> COMMAND_ENV_REQUIRED_OPTIONS =
      ImmutableList.of(CommonCommandOptions.class, ClientOptions.class);

  @Test
  public void crashTest() throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    ServerDirectories serverDirectories =
        new ServerDirectories(
            fs.getPath("/install"), fs.getPath("/output"), fs.getPath("/output_user"));
    BlazeRuntime runtime =
        new BlazeRuntime.Builder()
            .setFileSystem(fs)
            .setProductName("bazel")
            .setServerDirectories(serverDirectories)
            .setStartupOptionsProvider(Mockito.mock(OptionsParsingResult.class))
            .build();
    BlazeDirectories directories =
        new BlazeDirectories(
            serverDirectories, fs.getPath("/workspace"), fs.getPath("/system_javabase"), "blaze");
    BlazeWorkspace workspace = runtime.initWorkspace(directories, BinTools.empty(directories));
    EventBus eventBus = Mockito.mock(EventBus.class);
    OptionsParser options =
        OptionsParser.builder().optionsClasses(COMMAND_ENV_REQUIRED_OPTIONS).build();
    Thread commandThread = Mockito.mock(Thread.class);
    CommandEnvironment env =
        new CommandEnvironment(
            runtime,
            workspace,
            eventBus,
            commandThread,
            VersionCommand.class.getAnnotation(Command.class),
            options,
            ImmutableList.of(),
            0L,
            0L,
            ImmutableList.of());
    runtime.beforeCommand(env, options.getOptions(CommonCommandOptions.class));
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
    assertThat(runtime.afterCommand(env, mainThreadCrash).getDetailedExitCode()).isEqualTo(oom);
    // Confirm that runtime interrupted the command thread.
    verify(commandThread).interrupt();
  }

  @Test
  public void resultExtensions() throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    ServerDirectories serverDirectories =
        new ServerDirectories(
            fs.getPath("/install"), fs.getPath("/output"), fs.getPath("/output_user"));
    BlazeRuntime runtime =
        new BlazeRuntime.Builder()
            .setFileSystem(fs)
            .setProductName("bazel")
            .setServerDirectories(serverDirectories)
            .setStartupOptionsProvider(Mockito.mock(OptionsParsingResult.class))
            .build();
    BlazeDirectories directories =
        new BlazeDirectories(
            serverDirectories, fs.getPath("/workspace"), fs.getPath("/system_javabase"), "blaze");
    BlazeWorkspace workspace = runtime.initWorkspace(directories, BinTools.empty(directories));
    CommandEnvironment env =
        new CommandEnvironment(
            runtime,
            workspace,
            Mockito.mock(EventBus.class),
            Thread.currentThread(),
            VersionCommand.class.getAnnotation(Command.class),
            OptionsParser.builder().optionsClasses(COMMAND_ENV_REQUIRED_OPTIONS).build(),
            ImmutableList.of(),
            0L,
            0L,
            ImmutableList.of());
    Any anyFoo = Any.pack(StringValue.of("foo"));
    Any anyBar = Any.pack(BytesValue.of(ByteString.copyFromUtf8("bar")));
    env.addResponseExtensions(ImmutableList.of(anyFoo, anyBar));
    assertThat(runtime.afterCommand(env, BlazeCommandResult.success()).getResponseExtensions())
        .containsExactly(anyFoo, anyBar);
  }

  @Test
  public void addsCommandsFromModules() throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    ServerDirectories serverDirectories =
        new ServerDirectories(
            fs.getPath("/install"), fs.getPath("/output"), fs.getPath("/output_user"));
    BlazeRuntime runtime =
        new BlazeRuntime.Builder()
            .addBlazeModule(new FooCommandModule())
            .addBlazeModule(new BarCommandModule())
            .setFileSystem(fs)
            .setProductName("bazel")
            .setServerDirectories(serverDirectories)
            .setStartupOptionsProvider(Mockito.mock(OptionsParsingResult.class))
            .build();

    assertThat(runtime.getCommandMap().keySet()).containsExactly("foo", "bar").inOrder();
    assertThat(runtime.getCommandMap().get("foo")).isInstanceOf(FooCommandModule.FooCommand.class);
    assertThat(runtime.getCommandMap().get("bar")).isInstanceOf(BarCommandModule.BarCommand.class);
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
