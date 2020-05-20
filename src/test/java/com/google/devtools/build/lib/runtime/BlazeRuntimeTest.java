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

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.runtime.commands.VersionCommand;
import com.google.devtools.build.lib.server.FailureDetails.Crash;
import com.google.devtools.build.lib.server.FailureDetails.Crash.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for {@link BlazeRuntime} static methods. */
@RunWith(JUnit4.class)
public class BlazeRuntimeTest {
  @Test
  public void requestLogStringParsing() {
    assertThat(BlazeRuntime.getRequestLogString(ImmutableList.of("--client_env=A=B")))
        .isEqualTo("[--client_env=A=B]");
    assertThat(BlazeRuntime.getRequestLogString(ImmutableList.of("--client_env=BROKEN")))
        .isEqualTo("[--client_env=BROKEN]");
    assertThat(BlazeRuntime.getRequestLogString(ImmutableList.of("--client_env=auth=notprinted")))
        .isEqualTo("[--client_env=auth=__private_value_removed__]");
    assertThat(
            BlazeRuntime.getRequestLogString(ImmutableList.of("--client_env=MY_COOKIE=notprinted")))
        .isEqualTo("[--client_env=MY_COOKIE=__private_value_removed__]");
    assertThat(
            BlazeRuntime.getRequestLogString(
                ImmutableList.of("--client_env=dont_paSS_ME=notprinted")))
        .isEqualTo("[--client_env=dont_paSS_ME=__private_value_removed__]");
    assertThat(BlazeRuntime.getRequestLogString(ImmutableList.of("--client_env=ok=COOKIE")))
        .isEqualTo("[--client_env=ok=COOKIE]");
    assertThat(BlazeRuntime.getRequestLogString(
        ImmutableList.of("--client_env=foo=bar", "--client_env=pass=notprinted")))
            .isEqualTo("[--client_env=foo=bar, --client_env=pass=__private_value_removed__]");

    List<String> complexCommandLine = ImmutableList.of(
        "blaze",
        "build",
        "--client_env=FOO=BAR",
        "--client_env=FOOPASS=mypassword",
        "--package_path=./MY_PASSWORD/foo",
        "--client_env=SOMEAuThCode=something");
    assertThat(BlazeRuntime.getRequestLogString(complexCommandLine)).isEqualTo(
        "[blaze, build, --client_env=FOO=BAR, --client_env=FOOPASS=__private_value_removed__, "
            + "--package_path=./MY_PASSWORD/foo, "
            + "--client_env=SOMEAuThCode=__private_value_removed__]");
  }

  @Test
  public void optionSplitting() throws Exception {
    BlazeRuntime.CommandLineOptions options =
        BlazeRuntime.splitStartupOptions(
            ImmutableList.<BlazeModule>of(),
            "--install_base=/foo --host_jvm_args=-Xmx1B", "build", "//foo:bar", "--nobuild");
    assertThat(options.getStartupArgs())
        .isEqualTo(Arrays.asList("--install_base=/foo --host_jvm_args=-Xmx1B"));
    assertThat(options.getOtherArgs()).isEqualTo(Arrays.asList("build", "//foo:bar", "--nobuild"));
  }

  // A regression test to make sure that the 'no' prefix is handled correctly.
  @Test
  public void optionSplittingNoPrefix() throws Exception {
    BlazeRuntime.CommandLineOptions options = BlazeRuntime.splitStartupOptions(
        ImmutableList.<BlazeModule>of(), "--nobatch", "build");
    assertThat(options.getStartupArgs()).isEqualTo(Arrays.asList("--nobatch"));
    assertThat(options.getOtherArgs()).isEqualTo(Arrays.asList("build"));
  }

  private static final ImmutableList<Class<? extends OptionsBase>> COMMAND_ENV_REQUIRED_OPTIONS =
      ImmutableList.of(CommonCommandOptions.class, ClientOptions.class);

  @Test
  public void crashTest() throws Exception {
    FileSystem fs = new InMemoryFileSystem();
    ServerDirectories serverDirectories =
        new ServerDirectories(
            fs.getPath("/install"), fs.getPath("/output"), fs.getPath("/output_user"));
    BlazeRuntime runtime =
        new BlazeRuntime.Builder()
            .addBlazeModule(
                new BlazeModule() {
                  @Override
                  public BuildOptions getDefaultBuildOptions(BlazeRuntime runtime) {
                    return BuildOptions.builder().build();
                  }
                })
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
    CommandEnvironment env =
        new CommandEnvironment(
            runtime,
            workspace,
            eventBus,
            Thread.currentThread(),
            VersionCommand.class.getAnnotation(Command.class),
            options,
            ImmutableList.of(),
            0L,
            0L);
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
  }
}
