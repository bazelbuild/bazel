package com.google.devtools.build.lib.bazel.repository.downloader;

import java.util.HashMap;
import java.util.Map;

public class AuthorizationHeaderProvider {

    private static final Map<String, AuthorizationProtocol> TYPE_TO_PROTOCOL = new HashMap<>();

    public AuthorizationHeaderProvider() {
        TYPE_TO_PROTOCOL.put("github", new GithubAuthorizationProtocol());
    }

    public String getAuthorizationHeaderValue(String type, CredentialsProvider.Credentials credentials){
        return TYPE_TO_PROTOCOL.getOrDefault(type, new DefaultAuthorizationProtocol()).getAuthProtocol(credentials);
    }
}
