// Copyright 2007 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.util.io.CommandExtensionReporter.NO_OP_COMMAND_EXTENSION_REPORTER;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Range;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.common.eventbus.SubscriberExceptionContext;
import com.google.common.eventbus.SubscriberExceptionHandler;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.bazel.rules.BazelRulesModule;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.compress.CompressionServiceImpl;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link BlazeWorkspace}. */
@RunWith(JUnit4.class)
public class BlazeWorkspaceTest {
  @Command(name = "build", shortDescription = "", help = "")
  private static final class FakeBuildCommand {}

  private static final ImmutableList<Class<? extends OptionsBase>> COMMAND_ENV_REQUIRED_OPTIONS =
      ImmutableList.of(
          CommonCommandOptions.class, KeepStateAfterBuildOption.class, ClientOptions.class);

  private Scratch scratch = new Scratch();
  private ManualClock clock = new ManualClock();
  private RecordingExceptionHandler exceptionHandler = new RecordingExceptionHandler();
  private List<String> warnings = new ArrayList<>();

  private Path getWorkspace() throws IOException {
    return scratch.dir(LabelConstants.MAIN_REPO_DIR_NAME);
  }

  private Path getInstallBase() throws IOException {
    return scratch.dir("install_base");
  }

  private Path getOutputBase() throws IOException {
    return scratch.dir("output_base");
  }

  private BlazeWorkspace newWorkspace() throws Exception {
    String productName = TestConstants.PRODUCT_NAME;
    // no ConfiguredTargetFactory for testing Runtime basics
    ServerDirectories serverDirectories =
        new ServerDirectories(getInstallBase(), getOutputBase(), getInstallBase());
    BlazeRuntime runtime =
        new BlazeRuntime.Builder()
            .setProductName(productName)
            .setFileSystem(scratch.getFileSystem())
            .setServerDirectories(serverDirectories)
            .setClock(clock)
            .addBlazeService(new CompressionServiceImpl())
            .addBlazeModule(new BazelRulesModule())
            .setEventBusExceptionHandler(exceptionHandler)
            .setStartupOptionsProvider(
                OptionsParser.builder().optionsClasses(BlazeServerStartupOptions.class).build())
            .build();
    runtime.initWorkspace(
        new BlazeDirectories(serverDirectories, getWorkspace(), productName), /* binTools= */ null);
    return runtime.getWorkspace();
  }

  private CommandEnvironment initCommand(String... args) throws Exception {
    OptionsParser optionsParser =
        OptionsParser.builder().optionsClasses(COMMAND_ENV_REQUIRED_OPTIONS).build();
    optionsParser.parse(args);
    warnings = new ArrayList<>();
    return newWorkspace()
        .initCommand(
            FakeBuildCommand.class.getAnnotation(Command.class),
            optionsParser,
            InvocationPolicy.getDefaultInstance(),
            warnings,
            /* waitTimeInMs= */ 0,
            /* commandStartTime= */ 0,
            /* idleTaskResultsFromPreviousIdlePeriod= */ ImmutableList.of(),
            /* shutdownReasonConsumer= */ s -> {},
            /* commandExtensions= */ ImmutableList.of(),
            NO_OP_COMMAND_EXTENSION_REPORTER,
            /* attemptNumber= */ 1,
            /* buildRequestIdOverride= */ null,
            ConfigFlagDefinitions.NONE);
  }

  @Test
  public void testLastExecutionTimeGoingForwardSetsRange() throws Exception {
    BlazeWorkspace underTest = newWorkspace();
    clock.advanceMillis(1234);
    underTest.recordLastExecutionTime(/* commandStartTime= */ 1000);

    assertThat(underTest.getLastExecutionTimeRange())
        .isEqualTo(Range.closed(1_000_000_000L, 1_234_000_000L));
  }

  @Test
  public void testLastExecutionTimeGoingBackwardNullsValue() throws Exception {
    BlazeWorkspace underTest = newWorkspace();
    clock.advanceMillis(1000);
    underTest.recordLastExecutionTime(/* commandStartTime= */ 1234);

    assertThat(underTest.getLastExecutionTimeRange()).isNull();
  }

  @Test
  public void testRetainsOutputBase() throws Exception {
    assertThat(initCommand().getOutputBase()).isEqualTo(getOutputBase());
  }

  @Test
  public void testCreatesOutputBaseDoNotBuildHereFiles() throws Exception {
    runDoNotBuildHereFileTest(
        newWorkspace().getOutputBase().getChild("DO_NOT_BUILD_HERE"), getWorkspace());
  }

  @Test
  public void testCreatesExecrootDoNotBuildHereFiles() throws Exception {
    runDoNotBuildHereFileTest(
        newWorkspace().getOutputBase().getRelative("execroot/DO_NOT_BUILD_HERE"), getWorkspace());
  }

  private static void runDoNotBuildHereFileTest(Path path, Path workspace) throws IOException {
    assertThat(path.exists()).isTrue();
    // Containing the workspace path isn't an advertised feature, but it's been Hyrum's-lawed, so at
    // least make sure we don't break it unintentionally.
    assertThat(FileSystemUtils.readContentAsLatin1(path.getInputStream()))
        .isEqualTo(workspace.getPathString());
  }

  @Test
  public void testRetainsWorkspace() throws Exception {
    assertThat(initCommand().getWorkspace()).isEqualTo(getWorkspace());
  }

  @Test
  public void testRelativeWorkingDirectoryStartsEmpty() throws Exception {
    assertThat(initCommand().getRelativeWorkingDirectory().getPathString()).isEmpty();
  }

  @Test
  public void testBeforeCommandUpdatesRelativeWorkingDirectory() throws Exception {
    Command command = FakeBuildCommand.class.getAnnotation(Command.class);
    OptionsParser optionsParser =
        OptionsParser.builder().optionsClasses(COMMAND_ENV_REQUIRED_OPTIONS).build();
    optionsParser.parse("--client_cwd=" + getWorkspace().getRelative("foo").getPathString());
    BlazeWorkspace workspace = newWorkspace();
    CommandEnvironment env =
        workspace.initCommand(
            command,
            optionsParser,
            InvocationPolicy.getDefaultInstance(),
            /* warnings= */ new ArrayList<>(),
            /* waitTimeInMs= */ 0,
            /* commandStartTime= */ 0,
            /* idleTaskResultsFromPreviousIdlePeriod= */ ImmutableList.of(),
            /* shutdownReasonConsumer= */ s -> {},
            /* commandExtensions= */ ImmutableList.of(),
            NO_OP_COMMAND_EXTENSION_REPORTER,
            /* attemptNumber= */ 1,
            /* buildRequestIdOverride= */ null,
            ConfigFlagDefinitions.NONE);
    env.beforeCommand(InvocationPolicy.getDefaultInstance());
    assertThat(env.getRelativeWorkingDirectory().getPathString()).isEqualTo("foo");
  }

  @Test
  public void testExecRootIsSubdirOfOutputBase() throws Exception {
    CommandEnvironment env = initCommand();
    assertThat(env.getExecRoot())
        .isEqualTo(getOutputBase().getRelative("execroot/" + LabelConstants.MAIN_REPO_DIR_NAME));
  }

  @Test
  public void testUpdateClientEnv() throws Exception {
    CommandEnvironment env = initCommand("--client_env=V1=val1", "--client_env=V2=val2");
    assertThat(env.getClientEnv()).containsExactly("V1", "val1", "V2", "val2");
  }

  @Test
  public void testClientCwd() throws Exception {
    Command command = FakeBuildCommand.class.getAnnotation(Command.class);
    OptionsParser optionsParser =
        OptionsParser.builder().optionsClasses(COMMAND_ENV_REQUIRED_OPTIONS).build();
    optionsParser.parse("--client_cwd=/");
    AbruptExitException e =
        assertThrows(
            AbruptExitException.class,
            () ->
                newWorkspace()
                    .initCommand(
                        command,
                        optionsParser,
                        InvocationPolicy.getDefaultInstance(),
                        warnings,
                        /* waitTimeInMs= */ 0,
                        /* commandStartTime= */ 0,
                        /* idleTaskResultsFromPreviousIdlePeriod= */ ImmutableList.of(),
                        /* shutdownReasonConsumer= */ s -> {},
                        /* commandExtensions= */ ImmutableList.of(),
                        NO_OP_COMMAND_EXTENSION_REPORTER,
                        /* attemptNumber= */ 1,
                        /* buildRequestIdOverride= */ null,
                        ConfigFlagDefinitions.NONE)
                    .beforeCommand(InvocationPolicy.getDefaultInstance()));
    assertThat(e.getDetailedExitCode().getExitCode()).isEqualTo(ExitCode.COMMAND_LINE_ERROR);
  }

  // We expect that command_id and build_request_id get set by invocation policy, but calling
  // initCommand bypasses the application of invocation policy, which occurs earlier in the
  // BlazeCommandDispatcher. We rely on other tests to check that invocation policy enforcement
  // works as intended, this test checks that the final value is correctly interpreted.
  @Test
  public void testCommandId() throws Exception {
    UUID uuid = UUID.randomUUID();

    // Test that we accept a UUID from the environment
    CommandEnvironment env = initCommand("--client_env=BAZEL_INTERNAL_INVOCATION_ID=" + uuid);
    assertThat(env.getCommandId()).isEqualTo(uuid);
    assertThat(warnings)
        .containsExactly(
            "BAZEL_INTERNAL_INVOCATION_ID is set. This will soon be deprecated in favor of "
                + "--invocation_id. Please switch to using the flag.");

    // Test that we accept a UUID from the command argument
    env = initCommand("--invocation_id=" + uuid);
    assertThat(env.getCommandId()).isEqualTo(uuid);
    assertThat(warnings).isEmpty();

    // Test that the command argument wins over the environment variable.
    UUID otherUuid = UUID.randomUUID();
    env =
        initCommand(
            "--client_env=BAZEL_INTERNAL_INVOCATION_ID=" + otherUuid, "--invocation_id=" + uuid);
    assertThat(env.getCommandId()).isEqualTo(uuid);
    assertThat(warnings).isEmpty();
  }

  @Test
  public void testBuildRequestId() throws Exception {
    UUID uuid = UUID.randomUUID();

    // Test that we accept optionally prefixed UUID from the command argument
    CommandEnvironment env = initCommand("--build_request_id=" + uuid);
    assertThat(env.getBuildRequestId()).isEqualTo(uuid.toString());

    String prefixedUuid = "prefix-" + uuid;
    env = initCommand("--build_request_id=" + prefixedUuid);
    assertThat(env.getBuildRequestId()).isEqualTo(prefixedUuid);
  }

  @Test
  public void testInvalidBuildRequestIDs() throws Exception {
    UUID uuid = UUID.randomUUID();
    String[] invalidUUIDs = {
      "notAUUID",
      uuid.toString() + "-post",
      uuid.toString().replace('-', 'x'),
      uuid.toString().substring(2),
      "foo-" + uuid.toString().substring(2),
      "foo-" + uuid.toString().replace('-', 'x'),
      "foo-" + uuid.toString() + "-post"
    };
    for (String invalidUUID : invalidUUIDs) {
      OptionsParsingException e =
          assertThrows(
              invalidUUID + " should have failed to parse as a UUID",
              OptionsParsingException.class,
              () -> initCommand("--build_request_id=" + invalidUUID));
      assertThat(e).hasMessageThat().contains(invalidUUID);
      assertThat(e).hasMessageThat().contains("does not end in a valid UUID");
    }
  }

  @Test
  public void testEventBusThrowsAnException() throws Exception {
    EventBus eventBus = initCommand().getEventBus();
    final RuntimeException runtimeException = new RuntimeException();
    Object subscriber =
        new Object() {
          @Subscribe
          public void throwException(@SuppressWarnings("unused") String message) {
            throw runtimeException;
          }
        };
    eventBus.register(subscriber);
    final String message = "foo";
    eventBus.post(message);
    assertThat(exceptionHandler.exception).isNotNull();
    assertThat(exceptionHandler.exception).isEqualTo(runtimeException);
    assertThat(exceptionHandler.context.getSubscriber()).isEqualTo(subscriber);
    assertThat(exceptionHandler.context.getEvent()).isEqualTo(message);
    assertThat(exceptionHandler.context.getEventBus()).isEqualTo(eventBus);
  }

  private static final class RecordingExceptionHandler implements SubscriberExceptionHandler {
    private Throwable exception;
    private SubscriberExceptionContext context;

    @Override
    public void handleException(Throwable exception, SubscriberExceptionContext context) {
      this.exception = exception;
      this.context = context;
    }
  }
}
