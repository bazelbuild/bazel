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

package com.google.devtools.build.lib.authandtls.credentialhelper;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth8.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.OutputStream;
import java.net.URI;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CredentialHelperProvider}. */
@RunWith(JUnit4.class)
public class CredentialHelperProviderTest {
  private static final PathFragment DEFAULT_HELPER_PATH =
      PathFragment.create("/path/to/default/helper");
  private static final PathFragment EXAMPLE_COM_HELPER_PATH =
      PathFragment.create("/path/to/example/com/helper");
  private static final PathFragment EXAMPLE_COM_WILDCARD_HELPER_PATH =
      PathFragment.create("/path/to/example/com/wildcard/helper");
  private static final PathFragment SUB_EXAMPLE_COM_WILDCARD_HELPER_PATH =
      PathFragment.create("/path/to/sub/example/com/wildcard/helper");

  private final FileSystem fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);

  @Before
  public void setUp() throws Exception {
    setUpHelper(fileSystem.getPath(DEFAULT_HELPER_PATH));
    setUpHelper(fileSystem.getPath(EXAMPLE_COM_HELPER_PATH));
    setUpHelper(fileSystem.getPath(EXAMPLE_COM_WILDCARD_HELPER_PATH));
    setUpHelper(fileSystem.getPath(SUB_EXAMPLE_COM_WILDCARD_HELPER_PATH));
  }

  private void setUpHelper(Path path) throws Exception {
    Preconditions.checkNotNull(path);

    path.getParentDirectory().createDirectoryAndParents();
    try (OutputStream stream = path.getOutputStream()) {
      // Just create an empty file, nothing to do.
    }
    path.setExecutable(true);
  }

  @Test
  public void noHelpersConfigured() {
    CredentialHelperProvider provider = CredentialHelperProvider.builder().build();

    assertThat(provider.findCredentialHelper(URI.create("http://example.com/foo"))).isEmpty();
    assertThat(provider.findCredentialHelper(URI.create("https://example.com/foo"))).isEmpty();
    assertThat(provider.findCredentialHelper(URI.create("grpc://example.com/foo"))).isEmpty();
    assertThat(provider.findCredentialHelper(URI.create("grpcs://example.com/foo"))).isEmpty();
    assertThat(provider.findCredentialHelper(URI.create("custom://example.com/foo"))).isEmpty();

    assertThat(provider.findCredentialHelper(URI.create("https://subdomain.example.com/bar")))
        .isEmpty();
    assertThat(provider.findCredentialHelper(URI.create("https://other-domain.com"))).isEmpty();
  }

  private void assertInvalidPattern(String pattern) {
    Preconditions.checkNotNull(pattern);

    IllegalArgumentException exception =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                CredentialHelperProvider.builder()
                    .add(pattern, fileSystem.getPath(DEFAULT_HELPER_PATH)));
    assertThat(exception).hasMessageThat().contains(pattern);
  }

  @Test
  public void invalidPattern() throws Exception {
    assertInvalidPattern("foo.*.example.com");
    assertInvalidPattern("*.foo.*.example.com");
    assertInvalidPattern("*-foo.example.com");
    assertInvalidPattern("example.*");
    assertInvalidPattern("*.example.*");

    // Punycode
    assertInvalidPattern("foo.*.münchen.de");
    assertInvalidPattern(".*.münchen.de");
    assertInvalidPattern("foo-*.münchen.de");
  }

  @Test
  public void uriWithoutHostComponent() throws Exception {
    Path helper = fileSystem.getPath(EXAMPLE_COM_HELPER_PATH);
    CredentialHelperProvider provider =
        CredentialHelperProvider.builder().add("example.com", helper).build();

    assertThat(provider.findCredentialHelper(URI.create("unix:///path/to/socket"))).isEmpty();
  }

  @Test
  public void addNonExecutableDefaultHelper() throws Exception {
    Path helper = fileSystem.getPath("/path/to/non/executable");
    setUpHelper(helper);
    helper.setExecutable(false);
    CredentialHelperProvider.Builder provider = CredentialHelperProvider.builder();

    IllegalArgumentException exception =
        assertThrows(IllegalArgumentException.class, () -> provider.add(helper));
    assertThat(exception).hasMessageThat().contains("is not executable");
  }

  @Test
  public void onlyDefaultHelper() throws Exception {
    Path helper = fileSystem.getPath(DEFAULT_HELPER_PATH);
    CredentialHelperProvider provider = CredentialHelperProvider.builder().add(helper).build();

    assertThat(provider.findCredentialHelper(URI.create("http://example.com/foo")).get().getPath())
        .isEqualTo(helper);
    assertThat(provider.findCredentialHelper(URI.create("https://example.com/foo")).get().getPath())
        .isEqualTo(helper);
    assertThat(provider.findCredentialHelper(URI.create("grpc://example.com/foo")).get().getPath())
        .isEqualTo(helper);
    assertThat(provider.findCredentialHelper(URI.create("grpcs://example.com/foo")).get().getPath())
        .isEqualTo(helper);
    assertThat(provider.findCredentialHelper(URI.create("unix:///tmp/grpc.sock")).get().getPath())
        .isEqualTo(helper);
    assertThat(
            provider.findCredentialHelper(URI.create("custom://example.com/foo")).get().getPath())
        .isEqualTo(helper);

    assertThat(
            provider
                .findCredentialHelper(URI.create("https://subdomain.example.com/bar"))
                .get()
                .getPath())
        .isEqualTo(helper);
    assertThat(
            provider.findCredentialHelper(URI.create("https://other-domain.com")).get().getPath())
        .isEqualTo(helper);
  }

  @Test
  public void withHostHelpersAndDefaultFallback() throws Exception {
    Path defaultHelper = fileSystem.getPath(DEFAULT_HELPER_PATH);
    Path exampleComHelper = fileSystem.getPath(EXAMPLE_COM_HELPER_PATH);
    CredentialHelperProvider provider =
        CredentialHelperProvider.builder()
            .add(defaultHelper)
            .add("example.com", exampleComHelper)
            .build();

    assertThat(provider.findCredentialHelper(URI.create("http://example.com/foo")).get().getPath())
        .isEqualTo(exampleComHelper);
    assertThat(provider.findCredentialHelper(URI.create("https://example.com/foo")).get().getPath())
        .isEqualTo(exampleComHelper);
    assertThat(provider.findCredentialHelper(URI.create("grpc://example.com/foo")).get().getPath())
        .isEqualTo(exampleComHelper);
    assertThat(provider.findCredentialHelper(URI.create("grpcs://example.com/foo")).get().getPath())
        .isEqualTo(exampleComHelper);
    assertThat(
            provider.findCredentialHelper(URI.create("custom://example.com/foo")).get().getPath())
        .isEqualTo(exampleComHelper);

    assertThat(
            provider
                .findCredentialHelper(URI.create("https://subdomain.example.com/bar"))
                .get()
                .getPath())
        .isEqualTo(defaultHelper);
    assertThat(
            provider.findCredentialHelper(URI.create("https://other-domain.com")).get().getPath())
        .isEqualTo(defaultHelper);
    assertThat(provider.findCredentialHelper(URI.create("unix:///tmp/grpc.sock")).get().getPath())
        .isEqualTo(defaultHelper);
  }

  @Test
  public void wildcardMatching() throws Exception {
    Path defaultHelper = fileSystem.getPath(DEFAULT_HELPER_PATH);
    Path exampleComWildcardHelper = fileSystem.getPath(EXAMPLE_COM_WILDCARD_HELPER_PATH);
    CredentialHelperProvider provider =
        CredentialHelperProvider.builder()
            .add(defaultHelper)
            .add("*.example.com", exampleComWildcardHelper)
            .build();

    assertThat(provider.findCredentialHelper(URI.create("http://example.com/foo")).get().getPath())
        .isEqualTo(exampleComWildcardHelper);
    assertThat(provider.findCredentialHelper(URI.create("https://example.com/foo")).get().getPath())
        .isEqualTo(exampleComWildcardHelper);
    assertThat(provider.findCredentialHelper(URI.create("grpc://example.com/foo")).get().getPath())
        .isEqualTo(exampleComWildcardHelper);
    assertThat(provider.findCredentialHelper(URI.create("grpcs://example.com/foo")).get().getPath())
        .isEqualTo(exampleComWildcardHelper);
    assertThat(
            provider.findCredentialHelper(URI.create("custom://example.com/foo")).get().getPath())
        .isEqualTo(exampleComWildcardHelper);

    assertThat(
            provider
                .findCredentialHelper(URI.create("https://subdomain.example.com/bar"))
                .get()
                .getPath())
        .isEqualTo(exampleComWildcardHelper);
    assertThat(
            provider
                .findCredentialHelper(URI.create("https://subdomain2.example.com/bar"))
                .get()
                .getPath())
        .isEqualTo(exampleComWildcardHelper);
    assertThat(
            provider
                .findCredentialHelper(URI.create("https://sub.subdomain.example.com/bar"))
                .get()
                .getPath())
        .isEqualTo(exampleComWildcardHelper);
    assertThat(
            provider
                .findCredentialHelper(URI.create("https://subdomain.example.com/bar"))
                .get()
                .getPath())
        .isEqualTo(exampleComWildcardHelper);

    assertThat(
            provider.findCredentialHelper(URI.create("https://other-domain.com")).get().getPath())
        .isEqualTo(defaultHelper);

    assertThat(provider.findCredentialHelper(URI.create("unix:///tmp/grpc.sock")).get().getPath())
        .isEqualTo(defaultHelper);
  }

  @Test
  public void preferExactMatchOverWildcardMatching() throws Exception {
    Path defaultHelper = fileSystem.getPath(DEFAULT_HELPER_PATH);
    Path exampleComHelper = fileSystem.getPath(EXAMPLE_COM_HELPER_PATH);
    Path exampleComWildcardHelper = fileSystem.getPath(EXAMPLE_COM_WILDCARD_HELPER_PATH);
    CredentialHelperProvider provider =
        CredentialHelperProvider.builder()
            .add(defaultHelper)
            .add("example.com", exampleComHelper)
            .add("*.example.com", exampleComWildcardHelper)
            .build();

    assertThat(provider.findCredentialHelper(URI.create("http://example.com/foo")).get().getPath())
        .isEqualTo(exampleComHelper);
    assertThat(provider.findCredentialHelper(URI.create("https://example.com/foo")).get().getPath())
        .isEqualTo(exampleComHelper);
    assertThat(provider.findCredentialHelper(URI.create("grpc://example.com/foo")).get().getPath())
        .isEqualTo(exampleComHelper);
    assertThat(provider.findCredentialHelper(URI.create("grpcs://example.com/foo")).get().getPath())
        .isEqualTo(exampleComHelper);
    assertThat(
            provider.findCredentialHelper(URI.create("custom://example.com/foo")).get().getPath())
        .isEqualTo(exampleComHelper);

    assertThat(
            provider
                .findCredentialHelper(URI.create("https://subdomain.example.com/bar"))
                .get()
                .getPath())
        .isEqualTo(exampleComWildcardHelper);
    assertThat(
            provider
                .findCredentialHelper(URI.create("https://subdomain2.example.com/bar"))
                .get()
                .getPath())
        .isEqualTo(exampleComWildcardHelper);
    assertThat(
            provider
                .findCredentialHelper(URI.create("https://sub.subdomain.example.com/bar"))
                .get()
                .getPath())
        .isEqualTo(exampleComWildcardHelper);
    assertThat(
            provider
                .findCredentialHelper(URI.create("https://subdomain.example.com/bar"))
                .get()
                .getPath())
        .isEqualTo(exampleComWildcardHelper);

    assertThat(
            provider.findCredentialHelper(URI.create("https://other-domain.com")).get().getPath())
        .isEqualTo(defaultHelper);
  }

  @Test
  public void preferMostSpecificWildcardMatch() throws Exception {
    Path exampleComWildcardHelper = fileSystem.getPath(EXAMPLE_COM_WILDCARD_HELPER_PATH);
    Path subExampleComWildcardHelper = fileSystem.getPath(SUB_EXAMPLE_COM_WILDCARD_HELPER_PATH);
    CredentialHelperProvider provider =
        CredentialHelperProvider.builder()
            .add("*.example.com", exampleComWildcardHelper)
            .add("*.sub.example.com", subExampleComWildcardHelper)
            .build();

    assertThat(provider.findCredentialHelper(URI.create("https://example.com/bar")).get().getPath())
        .isEqualTo(exampleComWildcardHelper);
    assertThat(
            provider
                .findCredentialHelper(URI.create("https://foo.example.com/bar"))
                .get()
                .getPath())
        .isEqualTo(exampleComWildcardHelper);
    assertThat(
            provider
                .findCredentialHelper(URI.create("https://sub.example.com/bar"))
                .get()
                .getPath())
        .isEqualTo(subExampleComWildcardHelper);
    assertThat(
            provider
                .findCredentialHelper(URI.create("https://foo.sub.example.com/bar"))
                .get()
                .getPath())
        .isEqualTo(subExampleComWildcardHelper);
  }

  @Test
  public void punycodeMatching() throws Exception {
    Path defaultHelper = fileSystem.getPath(DEFAULT_HELPER_PATH);
    Path specificHelper = fileSystem.getPath(EXAMPLE_COM_HELPER_PATH);
    Path subdomainHelper = fileSystem.getPath(EXAMPLE_COM_HELPER_PATH);

    CredentialHelperProvider provider =
        CredentialHelperProvider.builder()
            .add(defaultHelper)
            .add("münchen.de", specificHelper)
            .add("*.köln.de", subdomainHelper)
            .build();

    // münchen.de
    assertThat(
            provider.findCredentialHelper(URI.create("http://xn--mnchen-3ya.de")).get().getPath())
        .isEqualTo(specificHelper);
    assertThat(
            provider
                .findCredentialHelper(URI.create("http://foo.xn--mnchen-3ya.de"))
                .get()
                .getPath())
        .isEqualTo(defaultHelper);
    assertThat(provider.findCredentialHelper(URI.create("http://muenchen.de")).get().getPath())
        .isEqualTo(defaultHelper);

    // köln.de
    assertThat(provider.findCredentialHelper(URI.create("http://xn--kln-sna.de")).get().getPath())
        .isEqualTo(subdomainHelper);
    assertThat(
            provider.findCredentialHelper(URI.create("http://foo.xn--kln-sna.de")).get().getPath())
        .isEqualTo(subdomainHelper);
    assertThat(
            provider.findCredentialHelper(URI.create("http://bar.xn--kln-sna.de")).get().getPath())
        .isEqualTo(subdomainHelper);
    assertThat(provider.findCredentialHelper(URI.create("http://koeln.de")).get().getPath())
        .isEqualTo(defaultHelper);

    // småland.se
    assertThat(
            provider.findCredentialHelper(URI.create("http://xn--smland-jua.se")).get().getPath())
        .isEqualTo(defaultHelper);
  }

  @Test
  public void parentDomain() {
    assertThat(CredentialHelperProvider.parentDomain("com")).isEmpty();

    assertThat(CredentialHelperProvider.parentDomain("foo.example.com")).hasValue("example.com");
    assertThat(CredentialHelperProvider.parentDomain("example.com")).hasValue("com");

    // Punycode URIs (münchen.de).
    assertThat(CredentialHelperProvider.parentDomain("foo.xn--mnchen-3ya.de"))
        .hasValue("xn--mnchen-3ya.de");
    assertThat(CredentialHelperProvider.parentDomain("bar.foo.xn--mnchen-3ya.de"))
        .hasValue("foo.xn--mnchen-3ya.de");
    assertThat(CredentialHelperProvider.parentDomain("xn--mnchen-3ya.de")).hasValue("de");
  }
}
