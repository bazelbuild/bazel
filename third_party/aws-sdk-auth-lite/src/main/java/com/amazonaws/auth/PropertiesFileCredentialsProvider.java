/*
 * Copyright 2010-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *  http://aws.amazon.com/apache2.0
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */
package com.amazonaws.auth;

import com.amazonaws.SdkClientException;

import java.io.File;
import java.io.IOException;

/**
 * {@link AWSCredentialsProvider} implementation that loads AWS security
 * credentials from a properties file provided on initialization.
 * <p>
 * The AWS access key ID is expected to be in the <code>accessKey</code>
 * property and the AWS secret key is expected to be in the
 * <code>secretKey</code> property.
 */
public class PropertiesFileCredentialsProvider implements
        AWSCredentialsProvider {

    private final String credentialsFilePath;

    /**
     * Creates a new PropertiesFileCredentialsProvider that will attempt to load
     * a custom file from the path specified to read AWS security credentials.
     * 
     * @param credentialsFilePath
     *            The custom classpath resource path to a properties file from
     *            which the AWS security credentials should be loaded.
     * 
     *            For example,
     *            <ul>
     *            <li>/etc/somewhere/credentials.properties</li>
     *            </ul>
     */
    public PropertiesFileCredentialsProvider(String credentialsFilePath) {
        if (credentialsFilePath == null)
            throw new IllegalArgumentException(
                    "Credentials file path cannot be null");
        this.credentialsFilePath = credentialsFilePath;
    }

    public AWSCredentials getCredentials() {
        try {
            return new PropertiesCredentials(new File(this.credentialsFilePath));
        } catch (IOException e) {
            throw new SdkClientException(
                    "Unable to load AWS credentials from the "
                            + credentialsFilePath + " file", e);
        }
    }

    public void refresh() {
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "(" + credentialsFilePath + ")";
    }
}