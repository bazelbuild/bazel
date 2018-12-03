package com.google.devtools.build.lib.bazel.repository.downloader;

import org.junit.Test;

import static com.google.common.truth.Truth.assertThat;

public class AuthorizationHeaderProviderTest {

    AuthorizationHeaderProvider authorizationHeaderProvider = new AuthorizationHeaderProvider();

    @Test
    public void valid_githubToken_isOk() {
        String token = authorizationHeaderProvider.getAuthorizationHeaderValue(
                "github",
                        new CredentialsProvider.Credentials(
                                "user", "123"));
        assertThat(token).isEqualTo("token 123");
    }

    @Test
    public void valid_defaultToken_isOk() {
        String token = authorizationHeaderProvider.getAuthorizationHeaderValue(
                "notgithub",
                        new CredentialsProvider.Credentials(
                                "user", "123"));
        assertThat(token).isEqualTo("123");
    }

    @Test
    public void noUser_githubHost_returns_emptyToken() {
        String token = authorizationHeaderProvider.getAuthorizationHeaderValue(
                "github", null);
        assertThat(token).isEqualTo("");
    }

    @Test
    public void noUser_defaultHost_returns_emptyToken() {
        String token = authorizationHeaderProvider.getAuthorizationHeaderValue(
                "nogithub", null);
        assertThat(token).isEqualTo("");
    }
}