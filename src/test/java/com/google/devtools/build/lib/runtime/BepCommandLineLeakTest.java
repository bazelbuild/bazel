// Copyright 2025 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.buildeventservice.BuildEventServiceOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEvent;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.runtime.CommandLineEvent.CanonicalCommandLineEvent;
import com.google.devtools.build.lib.runtime.CommandLineEvent.OriginalCommandLineEvent;
import com.google.devtools.build.lib.runtime.proto.CommandLineOuterClass.CommandLine;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.DisallowValues;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.FlagPolicy;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.SetValue;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.SetValue.Behavior;
import com.google.devtools.common.options.CommandNameCache;
import com.google.devtools.common.options.InvocationPolicyEnforcer;
import com.google.devtools.common.options.OptionPriority.PriorityCategory;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test that options with FULLY_REDACTED_IN_LOGS are redacted in BEP command line events. */
@RunWith(JUnit4.class)
public final class BepCommandLineLeakTest {

  private static final String BES_BEARER =
      "ya29" // make sure these two strings are not concatenated, as it trips git-secrets
          + ".BES_BEARER_TOKEN_xxxxxxxxxxxxxxxx";
  private static final String BES_HEADER_ARG = "--bes_header=Authorization=Bearer " + BES_BEARER;

  private static final String REMOTE_BEARER = "REMOTE_EXEC_TOKEN_zzzzzzzzzzzzzzzz";
  private static final String REMOTE_HEADER_ARG =
      "--remote_header=Authorization=Bearer " + REMOTE_BEARER;

  private static OptionsParser newCommandParser() throws Exception {
    OptionsParser parser =
        OptionsParser.builder()
            .optionsClasses(BuildEventServiceOptions.class, RemoteOptions.class)
            .build();
    parser.parse(
        PriorityCategory.COMMAND_LINE,
        /* source= */ "command line options",
        ImmutableList.of(BES_HEADER_ARG, REMOTE_HEADER_ARG));
    return parser;
  }

  @Test
  public void structuredCommandLine_redactsHeaders() throws Exception {
    OptionsParser cmd = newCommandParser();
    OptionsParser startup =
        OptionsParser.builder().optionsClasses(BlazeServerStartupOptions.class).build();

    CommandLine original =
        new OriginalCommandLineEvent(
                "testblaze",
                startup,
                "build",
                ImmutableList.of("//some:target"),
                false,
                cmd.asListOfExplicitOptions(),
                ImmutableSortedMap.of(),
                ImmutableSet.of(),
                Optional.of(ImmutableList.of()))
            .asStreamProto(null)
            .getStructuredCommandLine();

    CommandLine canonical =
        new CanonicalCommandLineEvent(
                "testblaze",
                startup,
                "build",
                ImmutableList.of("//some:target"),
                false,
                ImmutableSortedMap.of(),
                ImmutableSortedMap.of(),
                ImmutableSet.of(),
                cmd.asListOfCanonicalOptions(),
                false)
            .asStreamProto(null)
            .getStructuredCommandLine();

    for (CommandLine line : ImmutableList.of(original, canonical)) {
      assertThat(line.toString()).doesNotContain(BES_BEARER);
      assertThat(line.toString()).doesNotContain(REMOTE_BEARER);
      assertThat(line.toString()).contains("--bes_header=<REDACTED>");
      assertThat(line.toString()).contains("--remote_header=<REDACTED>");
    }
  }

  @Test
  public void unstructuredCommandLine_redactsHeaders() {
    ImmutableList<String> rawArgs =
        ImmutableList.of("build", BES_HEADER_ARG, REMOTE_HEADER_ARG, "//some:target");
    BuildEvent proto =
        new OriginalUnstructuredCommandLineEvent(SafeRequestLogging.redactArguments(rawArgs))
            .asStreamProto(null);

    String unstructured = proto.getUnstructuredCommandLine().toString();
    assertThat(unstructured).doesNotContain(BES_BEARER);
    assertThat(unstructured).doesNotContain(REMOTE_BEARER);
    assertThat(unstructured).contains("--bes_header=<REDACTED>");
    assertThat(unstructured).contains("--remote_header=<REDACTED>");
  }

  @Test
  public void invocationPolicy_redactsHeadersInLogs() throws Exception {
    String secretPolicyValue = "Authorization=Bearer secret_policy_token";
    InvocationPolicy invocationPolicy =
        InvocationPolicy.newBuilder()
            .addFlagPolicies(
                FlagPolicy.newBuilder()
                    .setFlagName("bes_header")
                    .setSetValue(
                        SetValue.newBuilder()
                            .setBehavior(Behavior.FINAL_VALUE_IGNORE_OVERRIDES)
                            .addFlagValue(secretPolicyValue)))
            .build();

    CommandNameCache.CommandNameCacheInstance.INSTANCE.setCommandNameCache(ImmutableSet::of);
    try {
      OptionsParser parser = newCommandParser();
      InvocationPolicyEnforcer enforcer =
          new InvocationPolicyEnforcer(invocationPolicy, Level.INFO, parser.getConversionContext());

      Logger enforcerLogger = Logger.getLogger(InvocationPolicyEnforcer.class.getName());
      Level oldLevel = enforcerLogger.getLevel();
      enforcerLogger.setLevel(Level.INFO);
      List<String> logMessages = new ArrayList<>();
      Handler handler =
          new Handler() {
            @Override
            public void publish(LogRecord record) {
              logMessages.add(record.getMessage());
            }

            @Override
            public void flush() {}

            @Override
            public void close() {}
          };
      enforcerLogger.addHandler(handler);
      try {
        enforcer.enforce(parser, "build", ImmutableList.builder());
      } finally {
        enforcerLogger.setLevel(oldLevel);
        enforcerLogger.removeHandler(handler);
      }

      assertThat(logMessages).isNotEmpty();
      String allLogs = String.join("\n", logMessages);
      assertThat(allLogs).doesNotContain(BES_BEARER);
      assertThat(allLogs).doesNotContain(secretPolicyValue);
      assertThat(allLogs).contains("<REDACTED>");
    } finally {
      CommandNameCache.CommandNameCacheInstance.INSTANCE.setCommandNameCache(null);
    }
  }

  @Test
  public void invocationPolicy_disallowValues_redactsHeadersInException() throws Exception {
    InvocationPolicy invocationPolicy =
        InvocationPolicy.newBuilder()
            .addFlagPolicies(
                FlagPolicy.newBuilder()
                    .setFlagName("bes_header")
                    .setDisallowValues(
                        DisallowValues.newBuilder()
                            .addDisallowedValues("Authorization=Bearer " + BES_BEARER)))
            .build();

    CommandNameCache.CommandNameCacheInstance.INSTANCE.setCommandNameCache(ImmutableSet::of);
    try {
      OptionsParser parser = newCommandParser();
      InvocationPolicyEnforcer enforcer =
          new InvocationPolicyEnforcer(invocationPolicy, Level.INFO, parser.getConversionContext());

      OptionsParsingException e =
          assertThrows(
              OptionsParsingException.class,
              () -> enforcer.enforce(parser, "build", ImmutableList.builder()));
      assertThat(e.getMessage()).doesNotContain(BES_BEARER);
      assertThat(e.getMessage()).contains("<REDACTED>");
    } finally {
      CommandNameCache.CommandNameCacheInstance.INSTANCE.setCommandNameCache(null);
    }
  }

  @Test
  public void invocationPolicy_throwOnOverride_redactsHeadersInException() throws Exception {
    String secretPolicyValue = "Authorization=Bearer secret_policy_token";
    InvocationPolicy invocationPolicy =
        InvocationPolicy.newBuilder()
            .addFlagPolicies(
                FlagPolicy.newBuilder()
                    .setFlagName("bes_header")
                    .setSetValue(
                        SetValue.newBuilder()
                            .setBehavior(Behavior.FINAL_VALUE_THROW_ON_OVERRIDE)
                            .addFlagValue(secretPolicyValue)))
            .build();

    CommandNameCache.CommandNameCacheInstance.INSTANCE.setCommandNameCache(ImmutableSet::of);
    try {
      OptionsParser parser = newCommandParser();
      InvocationPolicyEnforcer enforcer =
          new InvocationPolicyEnforcer(invocationPolicy, Level.INFO, parser.getConversionContext());

      OptionsParsingException e =
          assertThrows(
              OptionsParsingException.class,
              () -> enforcer.enforce(parser, "build", ImmutableList.builder()));
      assertThat(e.getMessage()).doesNotContain(secretPolicyValue);
      assertThat(e.getMessage()).contains("<REDACTED>");
    } finally {
      CommandNameCache.CommandNameCacheInstance.INSTANCE.setCommandNameCache(null);
    }
  }
}
