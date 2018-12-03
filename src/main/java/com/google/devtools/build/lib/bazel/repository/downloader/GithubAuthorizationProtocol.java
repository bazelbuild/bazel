package com.google.devtools.build.lib.bazel.repository.downloader;

public class GithubAuthorizationProtocol implements AuthorizationProtocol {

    @Override
    public String getAuthProtocol(CredentialsProvider.Credentials credentials) {
        if (credentials != null) {
            String token = credentials.getPassword().orElse("");
            return token.isEmpty() ? "" : "token " + token;
        }
        return "";
    }
}
