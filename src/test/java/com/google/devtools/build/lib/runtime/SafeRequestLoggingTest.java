// Copyright 2020 The Bazel Authors. All rights reserved.
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
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SafeRequestLogging}. */
@RunWith(JUnit4.class)
public class SafeRequestLoggingTest {

  @After
  public void resetRedactedNames() {
    SafeRequestLogging.setRedactedClientEnvNames(ImmutableList.of());
  }

  @Test
  public void testGetRequestLogStringPassesThroughNonSensitiveClientEnv() {
    assertThat(
            SafeRequestLogging.getRequestLogString(
                ImmutableList.of("--client_env=A=B", "--client_env=C=D")))
        .isEqualTo("[--client_env=A=B, --client_env=C=D]");
  }

  @Test
  public void testGetRequestLogStringToleratesNonsensicalClientEnv() {
    // Client env is key=value pairs, no '=' is silly, but shouldn't break anything.
    assertThat(SafeRequestLogging.getRequestLogString(ImmutableList.of("--client_env=BROKEN")))
        .isEqualTo("[--client_env=BROKEN]");
  }

  @Test
  public void testGetRequestLogStringStripsApparentAuthValues() {
    assertThat(
            SafeRequestLogging.getRequestLogString(
                ImmutableList.of("--client_env=auth=notprinted", "--client_env=other=isprinted")))
        .isEqualTo("[--client_env=auth=__private_value_removed__, --client_env=other=isprinted]");
  }

  @Test
  public void testGetRequestLogStringStripsApparentCookieValues() {
    assertThat(
            SafeRequestLogging.getRequestLogString(
                ImmutableList.of(
                    "--client_env=MY_COOKIE=notprinted", "--client_env=other=isprinted")))
        .isEqualTo(
            "[--client_env=MY_COOKIE=__private_value_removed__, --client_env=other=isprinted]");
  }

  @Test
  public void testGetRequestLogStringStripsApparentPasswordValues() {
    assertThat(
            SafeRequestLogging.getRequestLogString(
                ImmutableList.of(
                    "--client_env=dont_paSS_ME=notprinted", "--client_env=other=isprinted")))
        .isEqualTo(
            "[--client_env=dont_paSS_ME=__private_value_removed__, --client_env=other=isprinted]");
  }

  @Test
  public void testGetRequestLogStringStripsApparentTokenValues() {
    assertThat(
            SafeRequestLogging.getRequestLogString(
                ImmutableList.of(
                    "--client_env=service_ToKEn=notprinted", "--client_env=other=isprinted")))
        .isEqualTo(
            "[--client_env=service_ToKEn=__private_value_removed__, --client_env=other=isprinted]");
  }

  @Test
  public void testGetRequestLogStringStripsApparentApiKeyValues() {
    assertThat(
            SafeRequestLogging.getRequestLogString(
                ImmutableList.of(
                    "--client_env=MY_API_KEY=notprinted", "--client_env=other=isprinted")))
        .isEqualTo(
            "[--client_env=MY_API_KEY=__private_value_removed__, --client_env=other=isprinted]");
  }

  @Test
  public void testGetRequestLogStringStripsApparentCredentialValues() {
    assertThat(
            SafeRequestLogging.getRequestLogString(
                ImmutableList.of(
                    "--client_env=COURSIER_CREDENTIALS=notprinted",
                    "--client_env=other=isprinted")))
        .isEqualTo(
            "[--client_env=COURSIER_CREDENTIALS=__private_value_removed__,"
                + " --client_env=other=isprinted]");
  }

  @Test
  public void testGetRequestLogStringStripsApparentSecretValues() {
    assertThat(
            SafeRequestLogging.getRequestLogString(
                ImmutableList.of(
                    "--client_env=my_SeCrEt=notprinted", "--client_env=other=isprinted")))
        .isEqualTo(
            "[--client_env=my_SeCrEt=__private_value_removed__, --client_env=other=isprinted]");
  }

  @Test
  public void testGetRequestLogIgnoresSensitiveTermsInValues() {
    assertThat(SafeRequestLogging.getRequestLogString(ImmutableList.of("--client_env=ok=COOKIE")))
        .isEqualTo("[--client_env=ok=COOKIE]");
  }

  @Test
  public void testGetRequestLogForStandardCommandLine() {
    ImmutableList<String> complexCommandLine =
        ImmutableList.of(
            "blaze",
            "build",
            "--client_env=FOO=BAR",
            "--client_env=FOOPASS=mypassword",
            "--package_path=./MY_PASSWORD/foo",
            "--client_env=SOMEAuThCode=something");
    assertThat(SafeRequestLogging.getRequestLogString(complexCommandLine))
        .isEqualTo(
            "[blaze, build, --client_env=FOO=BAR, --client_env=FOOPASS=__private_value_removed__, "
                + "--package_path=./MY_PASSWORD/foo, "
                + "--client_env=SOMEAuThCode=__private_value_removed__]");
  }

  @Test
  public void testGetRequestLogStringStripsCredentialHeaderOptions() {
    assertThat(
            SafeRequestLogging.getRequestLogString(
                ImmutableList.of(
                    "blaze",
                    "build",
                    "--bes_header=Authorization=Bearer123",
                    "--remote_header",
                    "Authorization=Bearer456",
                    "--some_other_flag")))
        .isEqualTo(
            "[blaze, build, --bes_header=<REDACTED>, --remote_header, <REDACTED>, "
                + "--some_other_flag]");
  }

  @Test
  public void testRedactArgumentsRedactsCredentials() {
    ImmutableList<String> args =
        ImmutableList.of(
            "blaze",
            "build",
            "--bes_header=Auth=123",
            "--remote_header",
            "Auth=456",
            "--some_other_flag");
    assertThat(SafeRequestLogging.redactArguments(args))
        .containsExactly(
            "blaze",
            "build",
            "--bes_header=<REDACTED>",
            "--remote_header",
            "<REDACTED>",
            "--some_other_flag")
        .inOrder();
  }

  @Test
  public void testRedactArgumentsRedactsInvocationPolicy() {
    ImmutableList<String> args =
        ImmutableList.of(
            "blaze",
            "--invocation_policy=flag_policies:{flag_name:'bes_header'}",
            "--batch",
            "--invocation_policy",
            "flag_policies:{flag_name:'bes_header'}",
            "build");
    assertThat(SafeRequestLogging.redactArguments(args))
        .containsExactly(
            "blaze",
            "--invocation_policy=<REDACTED>",
            "--batch",
            "--invocation_policy",
            "<REDACTED>",
            "build")
        .inOrder();
  }

  @Test
  public void setRedactedClientEnvNames_exactNameIsRedacted_wholeNameOnly() {
    SafeRequestLogging.setRedactedClientEnvNames(ImmutableList.of("MY_INTERNAL_SIGNER"));
    assertThat(
            SafeRequestLogging.getRequestLogString(
                ImmutableList.of(
                    "--client_env=MY_INTERNAL_SIGNER=notprinted",
                    "--client_env=MY_INTERNAL_SIGNER_2=isprinted",
                    "--client_env=X_MY_INTERNAL_SIGNER=isprinted")))
        .isEqualTo(
            "[--client_env=MY_INTERNAL_SIGNER=__private_value_removed__,"
                + " --client_env=MY_INTERNAL_SIGNER_2=isprinted,"
                + " --client_env=X_MY_INTERNAL_SIGNER=isprinted]");
  }

  @Test
  public void setRedactedClientEnvNames_isCaseInsensitive() {
    SafeRequestLogging.setRedactedClientEnvNames(ImmutableList.of("my_internal_signer"));
    assertThat(
            SafeRequestLogging.getRequestLogString(
                ImmutableList.of("--client_env=MY_INTERNAL_SIGNER=notprinted")))
        .isEqualTo("[--client_env=MY_INTERNAL_SIGNER=__private_value_removed__]");
  }

  @Test
  public void setRedactedClientEnvNames_starMatchesAnyRunWithinName() {
    SafeRequestLogging.setRedactedClientEnvNames(ImmutableList.of("GLYD_*", "*_SIGNING_*"));
    assertThat(
            SafeRequestLogging.getRequestLogString(
                ImmutableList.of(
                    "--client_env=GLYD_X=notprinted",
                    "--client_env=GLYD_X=a=b",
                    "--client_env=PGP_SIGNING_PWD=notprinted",
                    "--client_env=XGLYD_X=isprinted",
                    "--client_env=OTHER=GLYD_X")))
        .isEqualTo(
            "[--client_env=GLYD_X=__private_value_removed__,"
                + " --client_env=GLYD_X=__private_value_removed__,"
                + " --client_env=PGP_SIGNING_PWD=__private_value_removed__,"
                + " --client_env=XGLYD_X=isprinted,"
                + " --client_env=OTHER=GLYD_X]");
  }

  @Test
  public void setRedactedClientEnvNames_bareStarRedactsEveryVariable() {
    SafeRequestLogging.setRedactedClientEnvNames(ImmutableList.of("*"));
    assertThat(
            SafeRequestLogging.getRequestLogString(
                ImmutableList.of("--client_env=A=notprinted", "--client_env=B=notprinted")))
        .isEqualTo(
            "[--client_env=A=__private_value_removed__,"
                + " --client_env=B=__private_value_removed__]");
  }

  @Test
  public void setRedactedClientEnvNames_regexMetacharactersAreLiteral() {
    SafeRequestLogging.setRedactedClientEnvNames(ImmutableList.of("A.B", "C[0]"));
    assertThat(
            SafeRequestLogging.getRequestLogString(
                ImmutableList.of(
                    "--client_env=A.B=notprinted",
                    "--client_env=AXB=isprinted",
                    "--client_env=C[0]=notprinted",
                    "--client_env=C0=isprinted")))
        .isEqualTo(
            "[--client_env=A.B=__private_value_removed__,"
                + " --client_env=AXB=isprinted,"
                + " --client_env=C[0]=__private_value_removed__,"
                + " --client_env=C0=isprinted]");
  }

  @Test
  public void setRedactedClientEnvNames_builtInHeuristicStillApplies() {
    SafeRequestLogging.setRedactedClientEnvNames(ImmutableList.of("MY_INTERNAL_SIGNER"));
    assertThat(
            SafeRequestLogging.getRequestLogString(
                ImmutableList.of("--client_env=MY_TOKEN=notprinted")))
        .isEqualTo("[--client_env=MY_TOKEN=__private_value_removed__]");
  }

  @Test
  public void setRedactedClientEnvNames_emptyListRestoresDefault() {
    SafeRequestLogging.setRedactedClientEnvNames(ImmutableList.of("MY_INTERNAL_SIGNER"));
    SafeRequestLogging.setRedactedClientEnvNames(ImmutableList.of());
    assertThat(
            SafeRequestLogging.getRequestLogString(
                ImmutableList.of("--client_env=MY_INTERNAL_SIGNER=isprinted")))
        .isEqualTo("[--client_env=MY_INTERNAL_SIGNER=isprinted]");
  }

  @Test
  public void setRedactedClientEnvNames_onlyAffectsClientEnvArgs() {
    SafeRequestLogging.setRedactedClientEnvNames(ImmutableList.of("MY_INTERNAL_SIGNER"));
    assertThat(
            SafeRequestLogging.getRequestLogString(
                ImmutableList.of("--package_path=MY_INTERNAL_SIGNER=x", "MY_INTERNAL_SIGNER=y")))
        .isEqualTo("[--package_path=MY_INTERNAL_SIGNER=x, MY_INTERNAL_SIGNER=y]");
  }
}
