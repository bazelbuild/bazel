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
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.authandtls.Netrc.Credential;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link NetrcParser}. */
@RunWith(JUnit4.class)
public class NetrcParserTest {

  private static final String FOO_MACHINE = "foo.example.org";
  private static final Credential FOO_CREDENTIAL =
      Credential.builder(FOO_MACHINE).setLogin("foouser").setPassword("foopass").build();
  private static final String BAR_MACHINE = "bar.example.org";
  private static final Credential BAR_CREDENTIAL =
      Credential.builder(BAR_MACHINE).setLogin("baruser").setPassword("barpass").build();
  private static final Credential DEFAULT_CREDENTIAL =
      Credential.builder("default").setLogin("defaultuser").setPassword("defaultpass").build();

  public static InputStream newInputStreamWithContent(String content) {
    return new ByteArrayInputStream(content.getBytes(UTF_8));
  }

  @Test
  public void parseAndClose_emptyContent_returnEmpty() throws IOException {
    InputStream inputStream = newInputStreamWithContent("");

    Netrc netrc = NetrcParser.parseAndClose(inputStream);

    assertThat(netrc.defaultCredential()).isNull();
    assertThat(netrc.credentials()).isEmpty();
  }

  @Test
  public void parseAndClose_emptyContentWithWhitespaces_returnEmpty() throws IOException {
    String content = "\t \n   \r\n  \n";
    InputStream inputStream = newInputStreamWithContent(content);

    Netrc netrc = NetrcParser.parseAndClose(inputStream);

    assertThat(netrc.defaultCredential()).isNull();
    assertThat(netrc.credentials()).isEmpty();
  }

  @Test
  public void parseAndClose_multipleMachines_returnMatched() throws IOException {
    String content =
        "machine "
            + FOO_MACHINE
            + " login "
            + FOO_CREDENTIAL.login()
            + " password "
            + FOO_CREDENTIAL.password()
            + "\n"
            + "machine "
            + BAR_MACHINE
            + " login "
            + BAR_CREDENTIAL.login()
            + " password "
            + BAR_CREDENTIAL.password();
    InputStream inputStream = newInputStreamWithContent(content);

    Netrc netrc = NetrcParser.parseAndClose(inputStream);

    assertThat(netrc.defaultCredential()).isNull();
    assertThat(netrc.credentials())
        .containsExactly(FOO_MACHINE, FOO_CREDENTIAL, BAR_MACHINE, BAR_CREDENTIAL);
  }

  @Test
  public void parseAndClose_correctButBadlyFormattedContent_returnMatched() throws IOException {
    String content =
        "machine "
            + FOO_MACHINE
            + "\r\n   login "
            + FOO_CREDENTIAL.login()
            + "\n\t\t\t password \t\t\n"
            + FOO_CREDENTIAL.password()
            + "\n";
    InputStream inputStream = newInputStreamWithContent(content);

    Netrc netrc = NetrcParser.parseAndClose(inputStream);

    assertThat(netrc.defaultCredential()).isNull();
    assertThat(netrc.credentials()).containsExactly(FOO_MACHINE, FOO_CREDENTIAL);
  }

  @Test
  public void parseAndClose_macdefOnly_returnEmpty() throws IOException {
    String content = "macdef init\n" + "\tcd /pub\n" + "\tmget *\n" + "\tquit";
    InputStream inputStream = newInputStreamWithContent(content);

    Netrc netrc = NetrcParser.parseAndClose(inputStream);

    assertThat(netrc.defaultCredential()).isNull();
    assertThat(netrc.credentials()).isEmpty();
  }

  @Test
  public void parseAndClose_mixOfMachinesAndMacdef_skipMacdefAndReturnMatched() throws IOException {
    String content =
        "machine "
            + FOO_MACHINE
            + " login "
            + FOO_CREDENTIAL.login()
            + " password "
            + FOO_CREDENTIAL.password()
            + "\n"
            + "macdef init\n"
            + "\tcd /pub\n"
            + "\tmget *\n"
            + "\tquit\n"
            + "\n"
            + "machine "
            + BAR_MACHINE
            + " login "
            + BAR_CREDENTIAL.login()
            + " password "
            + BAR_CREDENTIAL.password();
    InputStream inputStream = newInputStreamWithContent(content);

    Netrc netrc = NetrcParser.parseAndClose(inputStream);

    assertThat(netrc.defaultCredential()).isNull();
    assertThat(netrc.credentials())
        .containsExactly(FOO_MACHINE, FOO_CREDENTIAL, BAR_MACHINE, BAR_CREDENTIAL);
  }

  @Test
  public void parseAndClose_macdefWithCommentsInBetween_skipMacdefAndReturnMatched()
      throws IOException {
    String content =
        "machine "
            + FOO_MACHINE
            + " login "
            + FOO_CREDENTIAL.login()
            + " password "
            + FOO_CREDENTIAL.password()
            + "\n"
            + "macdef init\n"
            + "# this is comment\n"
            + "\tcd /pub\n"
            + "\tmget *\n"
            + "\tquit\n"
            + "# this is comment\n"
            + "\n"
            + "machine "
            + BAR_MACHINE
            + " login "
            + BAR_CREDENTIAL.login()
            + " password "
            + BAR_CREDENTIAL.password();
    InputStream inputStream = newInputStreamWithContent(content);

    Netrc netrc = NetrcParser.parseAndClose(inputStream);

    assertThat(netrc.defaultCredential()).isNull();
    assertThat(netrc.credentials())
        .containsExactly(FOO_MACHINE, FOO_CREDENTIAL, BAR_MACHINE, BAR_CREDENTIAL);
  }

  @Test
  public void parseAndClose_duplicatedMachineFields_override() throws IOException {
    String content =
        "machine "
            + FOO_MACHINE
            + " login overridden_user login "
            + FOO_CREDENTIAL.login()
            + " password overridden_pass password "
            + FOO_CREDENTIAL.password();
    InputStream inputStream = newInputStreamWithContent(content);

    Netrc netrc = NetrcParser.parseAndClose(inputStream);

    assertThat(netrc.defaultCredential()).isNull();
    assertThat(netrc.credentials()).containsExactly(FOO_MACHINE, FOO_CREDENTIAL);
  }

  @Test
  public void parseAndClose_machinesAfterDefault_ignore() throws IOException {
    String content =
        "machine "
            + FOO_MACHINE
            + " login "
            + FOO_CREDENTIAL.login()
            + " password "
            + FOO_CREDENTIAL.password()
            + "\n"
            + "default login "
            + DEFAULT_CREDENTIAL.login()
            + " password "
            + DEFAULT_CREDENTIAL.password()
            + "\n"
            + "machine "
            + BAR_MACHINE
            + " login "
            + BAR_CREDENTIAL.login()
            + " password "
            + BAR_CREDENTIAL.password();
    InputStream inputStream = newInputStreamWithContent(content);

    Netrc netrc = NetrcParser.parseAndClose(inputStream);

    assertThat(netrc.defaultCredential()).isEqualTo(DEFAULT_CREDENTIAL);
    assertThat(netrc.credentials()).containsExactly(FOO_MACHINE, FOO_CREDENTIAL);
  }

  @Test
  public void parseAndClose_badStartingContent_fail() {
    String content = "this is not netrc syntax";
    InputStream inputStream = newInputStreamWithContent(content);

    assertThrows(
        IOException.class,
        () -> {
          NetrcParser.parseAndClose(inputStream);
        });
  }

  @Test
  public void parseAndClose_badMachine_fail() {
    String content = "machine this is not netrc syntax";
    InputStream inputStream = newInputStreamWithContent(content);

    assertThrows(
        IOException.class,
        () -> {
          NetrcParser.parseAndClose(inputStream);
        });
  }

  @Test
  public void parseAndClose_badDefault_fail() {
    String content = "default this is not netrc syntax";
    InputStream inputStream = newInputStreamWithContent(content);

    assertThrows(
        IOException.class,
        () -> {
          NetrcParser.parseAndClose(inputStream);
        });
  }

  @Test
  public void parseAndClose_commentOnly_returnEmpty() throws IOException {
    String content = "# this is comment";
    InputStream inputStream = newInputStreamWithContent(content);

    Netrc netrc = NetrcParser.parseAndClose(inputStream);

    assertThat(netrc.defaultCredential()).isNull();
    assertThat(netrc.credentials()).isEmpty();
  }

  @Test
  public void credential_shouldNotLeakPassword() {
    assertThat(FOO_CREDENTIAL.toString()).doesNotContain(FOO_CREDENTIAL.password());
  }
}
