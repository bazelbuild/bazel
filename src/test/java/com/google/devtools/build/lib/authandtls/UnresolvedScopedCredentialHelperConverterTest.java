// Copyright 2022 The Bazel Authors. All rights reserved.
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
import static com.google.common.truth.Truth8.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions.UnresolvedScopedCredentialHelper;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions.UnresolvedScopedCredentialHelperConverter;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link UnresolvedScopedCredentialHelperConverter}. */
@RunWith(JUnit4.class)
public class UnresolvedScopedCredentialHelperConverterTest {
  @Test
  public void convertAbsolutePath() throws Exception {
    UnresolvedScopedCredentialHelper helper1 =
        UnresolvedScopedCredentialHelperConverter.INSTANCE.convert("/absolute/path");
    assertThat(helper1.getScope()).isEmpty();
    assertThat(helper1.getPath()).isEqualTo("/absolute/path");

    UnresolvedScopedCredentialHelper helper2 =
        UnresolvedScopedCredentialHelperConverter.INSTANCE.convert("example.com=/absolute/path");
    assertThat(helper2.getScope()).hasValue("example.com");
    assertThat(helper2.getPath()).isEqualTo("/absolute/path");

    UnresolvedScopedCredentialHelper helper3 =
        UnresolvedScopedCredentialHelperConverter.INSTANCE.convert("*.example.com=/absolute/path");
    assertThat(helper3.getScope()).hasValue("*.example.com");
    assertThat(helper3.getPath()).isEqualTo("/absolute/path");
  }

  @Test
  public void convertRootRelativePath() throws Exception {
    UnresolvedScopedCredentialHelper helper1 =
        UnresolvedScopedCredentialHelperConverter.INSTANCE.convert("%workspace%/path");
    assertThat(helper1.getScope()).isEmpty();
    assertThat(helper1.getPath()).isEqualTo("%workspace%/path");

    UnresolvedScopedCredentialHelper helper2 =
        UnresolvedScopedCredentialHelperConverter.INSTANCE.convert("example.com=%workspace%/path");
    assertThat(helper2.getScope()).hasValue("example.com");
    assertThat(helper2.getPath()).isEqualTo("%workspace%/path");

    UnresolvedScopedCredentialHelper helper3 =
        UnresolvedScopedCredentialHelperConverter.INSTANCE.convert(
            "*.example.com=%workspace%/path");
    assertThat(helper3.getScope()).hasValue("*.example.com");
    assertThat(helper3.getPath()).isEqualTo("%workspace%/path");
  }

  @Test
  public void convertPathLookup() throws Exception {
    UnresolvedScopedCredentialHelper helper1 =
        UnresolvedScopedCredentialHelperConverter.INSTANCE.convert("foo");
    assertThat(helper1.getScope()).isEmpty();
    assertThat(helper1.getPath()).isEqualTo("foo");

    UnresolvedScopedCredentialHelper helper2 =
        UnresolvedScopedCredentialHelperConverter.INSTANCE.convert("example.com=foo");
    assertThat(helper2.getScope()).hasValue("example.com");
    assertThat(helper2.getPath()).isEqualTo("foo");

    UnresolvedScopedCredentialHelper helper3 =
        UnresolvedScopedCredentialHelperConverter.INSTANCE.convert("*.example.com=foo");
    assertThat(helper3.getScope()).hasValue("*.example.com");
    assertThat(helper3.getPath()).isEqualTo("foo");
  }

  @Test
  public void emptyPath() {
    assertThrows(
        OptionsParsingException.class,
        () -> UnresolvedScopedCredentialHelperConverter.INSTANCE.convert(""));
    assertThrows(
        OptionsParsingException.class,
        () -> UnresolvedScopedCredentialHelperConverter.INSTANCE.convert("foo="));
    assertThrows(
        OptionsParsingException.class,
        () -> UnresolvedScopedCredentialHelperConverter.INSTANCE.convert("="));
  }

  @Test
  public void emptyScope() {
    assertThrows(
        OptionsParsingException.class,
        () -> UnresolvedScopedCredentialHelperConverter.INSTANCE.convert("=/foo"));
    assertThrows(
        OptionsParsingException.class,
        () -> UnresolvedScopedCredentialHelperConverter.INSTANCE.convert("="));
  }
}
