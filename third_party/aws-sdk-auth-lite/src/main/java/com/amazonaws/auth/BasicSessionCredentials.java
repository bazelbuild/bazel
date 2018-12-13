/*
 * Copyright 2011-2018 Amazon Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *    http://aws.amazon.com/apache2.0
 *
 * This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and
 * limitations under the License.
 */
package com.amazonaws.auth;

/**
 * Simple session credentials with keys and session token.
 */
public class BasicSessionCredentials implements AWSSessionCredentials {

    private final String awsAccessKey;
    private final String awsSecretKey;
    private final String sessionToken;
    
    public BasicSessionCredentials(String awsAccessKey, String awsSecretKey, String sessionToken) {
        this.awsAccessKey = awsAccessKey;
        this.awsSecretKey = awsSecretKey;
        this.sessionToken = sessionToken;
    }

    public String getAWSAccessKeyId() {
        return awsAccessKey;
    }

    public String getAWSSecretKey() {
        return awsSecretKey;
    }

    public String getSessionToken() {
        return sessionToken;
    }

}
