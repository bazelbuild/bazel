package com.google.devtools.build.lib.bazel.repository.downloader;

public class DefaultAuthorizationProtocol implements AuthorizationProtocol {

    @Override
    public String getAuthProtocol(CredentialsProvider.Credentials credentials) {
        if(credentials != null){
            return credentials.getPassword().orElse("");
        }
        return "";
    }
}
