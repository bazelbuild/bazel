package com.google.devtools.build.lib.bazel.repository.downloader;

public class NetrcCredentialsProviderException extends RuntimeException {

    NetrcCredentialsProviderException(String message) {
        super(message);
    }

    NetrcCredentialsProviderException(String message, Throwable t) {
        super(message, t);
    }

}
