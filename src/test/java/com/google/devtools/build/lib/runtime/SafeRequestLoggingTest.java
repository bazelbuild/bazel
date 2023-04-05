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
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SafeRequestLogging}. */
@RunWith(JUnit4.class)
public class SafeRequestLoggingTest {

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
  public void testGetRequestLogIgnoresSensitiveTermsInValues() {
    assertThat(SafeRequestLogging.getRequestLogString(ImmutableList.of("--client_env=ok=COOKIE")))
        .isEqualTo("[--client_env=ok=COOKIE]");
  }

  @Test
  public void testGetRequestLogForStandardCommandLine() {
    List<String> complexCommandLine = ImmutableList.of(
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
}
