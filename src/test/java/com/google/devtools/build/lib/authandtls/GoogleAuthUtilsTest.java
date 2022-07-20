package com.google.devtools.build.lib.authandtls;

import static com.google.common.truth.Truth8.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.auth.Credentials;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.truth.Truth;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.net.URI;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class GoogleAuthUtilsTest {
  @Test
  public void testNetrc_emptyEnv_shouldIgnore() throws Exception {
    Map<String, String> clientEnv = ImmutableMap.of();
    FileSystem fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);

    assertThat(GoogleAuthUtils.newCredentialsFromNetrc(clientEnv, fileSystem)).isEmpty();
  }

  @Test
  public void testNetrc_netrcNotExist_shouldIgnore() throws Exception {
    String home = "/home/foo";
    Map<String, String> clientEnv = ImmutableMap.of("HOME", home);
    FileSystem fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);

    assertThat(GoogleAuthUtils.newCredentialsFromNetrc(clientEnv, fileSystem)).isEmpty();
  }

  @Test
  public void testNetrc_netrcExist_shouldUse() throws Exception {
    String home = "/home/foo";
    Map<String, String> clientEnv = ImmutableMap.of("HOME", home);
    FileSystem fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);
    Scratch scratch = new Scratch(fileSystem);
    scratch.file(home + "/.netrc", "machine foo.example.org login foouser password foopass");

    Optional<Credentials> credentials = GoogleAuthUtils.newCredentialsFromNetrc(clientEnv, fileSystem);

    assertThat(credentials).isPresent();
    assertRequestMetadata(
        credentials.get().getRequestMetadata(URI.create("https://foo.example.org")),
        "foouser",
        "foopass");
  }

  @Test
  public void testNetrc_netrcFromNetrcEnvExist_shouldUse() throws Exception {
    String home = "/home/foo";
    String netrc = "/.netrc";
    Map<String, String> clientEnv = ImmutableMap.of("HOME", home, "NETRC", netrc);
    FileSystem fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);
    Scratch scratch = new Scratch(fileSystem);
    scratch.file(home + "/.netrc", "machine foo.example.org login foouser password foopass");
    scratch.file(netrc, "machine foo.example.org login baruser password barpass");

    Optional<Credentials> credentials = GoogleAuthUtils.newCredentialsFromNetrc(clientEnv, fileSystem);

    assertThat(credentials).isPresent();
    assertRequestMetadata(
        credentials.get().getRequestMetadata(URI.create("https://foo.example.org")),
        "baruser",
        "barpass");
  }

  @Test
  public void testNetrc_netrcFromNetrcEnvNotExist_shouldIgnore() throws Exception {
    String home = "/home/foo";
    String netrc = "/.netrc";
    Map<String, String> clientEnv = ImmutableMap.of("HOME", home, "NETRC", netrc);
    FileSystem fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);
    Scratch scratch = new Scratch(fileSystem);
    scratch.file(home + "/.netrc", "machine foo.example.org login foouser password foopass");

    assertThat(GoogleAuthUtils.newCredentialsFromNetrc(clientEnv, fileSystem)).isEmpty();
  }

  private static void assertRequestMetadata(
      Map<String, List<String>> requestMetadata, String username, String password) {
    Truth.assertThat(requestMetadata.keySet()).containsExactly("Authorization");
    Truth.assertThat(Iterables.getOnlyElement(requestMetadata.values()))
        .containsExactly(BasicHttpAuthenticationEncoder.encode(username, password, UTF_8));
  }
}
