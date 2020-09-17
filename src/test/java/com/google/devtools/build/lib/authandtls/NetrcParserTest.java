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
package com.google.devtools.build.lib.authandtls;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.authandtls.Netrc.Credential;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link NetrcParser}.
 */
@RunWith(JUnit4.class)
public class NetrcParserTest {

  private final static String fooMachine = "foo.example.org";
  private final static Credential fooCredential = Credential.builder("foo.example.org")
      .setLogin("foouser")
      .setPassword("foopass").build();
  private final static String barMachine = "bar.example.org";
  private final static Credential barCredential = Credential.builder("bar.example.org")
      .setLogin("baruser")
      .setPassword("barpass").build();
  private final static Credential defaultCredential = Credential.builder("default")
      .setLogin("defaultuser")
      .setPassword("defaultpass").build();

  @Test
  public void shouldWorkOnEmptyContent() throws IOException {
    String content = "";
    InputStream inputStream = new ByteArrayInputStream(content.getBytes(StandardCharsets.UTF_8));

    Netrc netrc = NetrcParser.parseAndClose(inputStream);

    assertThat(netrc.getDefaultCredential()).isNull();
    assertThat(netrc.getCredentials()).isEmpty();
  }

  @Test
  public void shouldWorkOnEmptyContentWithWhitespaces() throws IOException {
    String content = "\t \n   \r\n  \n";
    InputStream inputStream = new ByteArrayInputStream(content.getBytes(StandardCharsets.UTF_8));

    Netrc netrc = NetrcParser.parseAndClose(inputStream);

    assertThat(netrc.getDefaultCredential()).isNull();
    assertThat(netrc.getCredentials()).isEmpty();
  }

  @Test
  public void shouldWorkWithMultipleMachines() throws IOException {
    String content =
        "machine " + fooMachine + " login " + fooCredential.login() + " password " + fooCredential
            .password() + "\n"
            + "machine " + barMachine + " login " + barCredential.login() + " password "
            + barCredential.password();
    InputStream inputStream = new ByteArrayInputStream(content.getBytes(StandardCharsets.UTF_8));

    Netrc netrc = NetrcParser.parseAndClose(inputStream);

    ImmutableMap<String, Credential> expectedCredentials = ImmutableMap.of(
        fooMachine,
        fooCredential,
        barMachine,
        barCredential
    );
    assertThat(netrc.getDefaultCredential()).isNull();
    assertThat(netrc.getCredentials()).isEqualTo(expectedCredentials);
    assertThat(netrc.getCredential(fooMachine)).isEqualTo(fooCredential);
    assertThat(netrc.getCredential(barMachine)).isEqualTo(barCredential);
  }

  @Test
  public void shouldWorkWithCorrectButBadlyFormattedContent() throws IOException {
    String content =
        "machine " + fooMachine + "\r\n   login " + fooCredential.login()
            + "\n\t\t\t password \t\t\n" + fooCredential
            .password() + "\n";
    InputStream inputStream = new ByteArrayInputStream(content.getBytes(StandardCharsets.UTF_8));

    Netrc netrc = NetrcParser.parseAndClose(inputStream);

    ImmutableMap<String, Credential> expectedCredentials = ImmutableMap.of(
        fooMachine,
        fooCredential
    );
    assertThat(netrc.getDefaultCredential()).isNull();
    assertThat(netrc.getCredentials()).isEqualTo(expectedCredentials);
    assertThat(netrc.getCredential(fooMachine)).isEqualTo(fooCredential);
  }

  @Test
  public void shouldSkipMacdef() throws IOException {
    String content =
        "macdef init\n"
            + "\tcd /pub\n"
            + "\tmget *\n"
            + "\tquit";
    InputStream inputStream = new ByteArrayInputStream(content.getBytes(StandardCharsets.UTF_8));

    Netrc netrc = NetrcParser.parseAndClose(inputStream);

    assertThat(netrc.getDefaultCredential()).isNull();
    assertThat(netrc.getCredentials()).isEmpty();
  }

  @Test
  public void shouldWorkWithMixOfMachinesAndMacdef() throws IOException {
    String content =
        "machine " + fooMachine + " login " + fooCredential.login() + " password " + fooCredential
            .password() + "\n"
            + "macdef init\n"
            + "\tcd /pub\n"
            + "\tmget *\n"
            + "\tquit\n"
            + "\n"
            + "machine " + barMachine + " login " + barCredential.login() + " password "
            + barCredential.password();
    InputStream inputStream = new ByteArrayInputStream(content.getBytes(StandardCharsets.UTF_8));

    Netrc netrc = NetrcParser.parseAndClose(inputStream);

    ImmutableMap<String, Credential> expectedCredentials = ImmutableMap.of(
        fooMachine,
        fooCredential,
        barMachine,
        barCredential
    );
    assertThat(netrc.getDefaultCredential()).isNull();
    assertThat(netrc.getCredentials()).isEqualTo(expectedCredentials);
    assertThat(netrc.getCredential(fooMachine)).isEqualTo(fooCredential);
    assertThat(netrc.getCredential(barMachine)).isEqualTo(barCredential);
  }

  @Test
  public void shouldOverrideDuplicatedMachineFields() throws IOException {
    String content =
        "machine " + fooMachine + " login overridden_user login " + fooCredential.login()
            + " password overridden_pass password " + fooCredential
            .password();
    InputStream inputStream = new ByteArrayInputStream(content.getBytes(StandardCharsets.UTF_8));

    Netrc netrc = NetrcParser.parseAndClose(inputStream);

    ImmutableMap<String, Credential> expectedCredentials = ImmutableMap.of(
        fooMachine,
        fooCredential
    );
    assertThat(netrc.getDefaultCredential()).isNull();
    assertThat(netrc.getCredentials()).isEqualTo(expectedCredentials);
    assertThat(netrc.getCredential(fooMachine)).isEqualTo(fooCredential);
  }

  @Test
  public void shouldIgnoreMachinesAfterDefault() throws IOException {
    String content =
        "machine " + fooMachine + " login " + fooCredential.login() + " password " + fooCredential
            .password() + "\n"
            + "default login " + defaultCredential.login() + " password " + defaultCredential
            .password() + "\n"
            + "machine " + barMachine + " login " + barCredential.login() + " password "
            + barCredential.password();
    InputStream inputStream = new ByteArrayInputStream(content.getBytes(StandardCharsets.UTF_8));

    Netrc netrc = NetrcParser.parseAndClose(inputStream);

    ImmutableMap<String, Credential> expectedCredentials = ImmutableMap.of(
        fooMachine,
        fooCredential
    );
    assertThat(netrc.getDefaultCredential()).isEqualTo(defaultCredential);
    assertThat(netrc.getCredentials()).isEqualTo(expectedCredentials);
    assertThat(netrc.getCredential(fooMachine)).isEqualTo(fooCredential);
  }

  @Test
  public void shouldFailWithBadStartingContent() {
    String content = "this is not netrc syntax";
    InputStream inputStream = new ByteArrayInputStream(content.getBytes(StandardCharsets.UTF_8));

    assertThrows(IOException.class, () -> {
      NetrcParser.parseAndClose(inputStream);
    });
  }

  @Test
  public void shouldFailWithBadMachine() {
    String content = "machine this is not netrc syntax";
    InputStream inputStream = new ByteArrayInputStream(content.getBytes(StandardCharsets.UTF_8));

    assertThrows(IOException.class, () -> {
      NetrcParser.parseAndClose(inputStream);
    });
  }

  @Test
  public void shouldFailWithBadDefault() {
    String content = "default this is not netrc syntax";
    InputStream inputStream = new ByteArrayInputStream(content.getBytes(StandardCharsets.UTF_8));

    assertThrows(IOException.class, () -> {
      NetrcParser.parseAndClose(inputStream);
    });
  }
}
