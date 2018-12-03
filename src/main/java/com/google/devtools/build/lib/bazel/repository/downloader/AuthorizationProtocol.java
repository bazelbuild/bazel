package com.google.devtools.build.lib.bazel.repository.downloader;

public interface AuthorizationProtocol {

    String getAuthProtocol(CredentialsProvider.Credentials credentials);

}
