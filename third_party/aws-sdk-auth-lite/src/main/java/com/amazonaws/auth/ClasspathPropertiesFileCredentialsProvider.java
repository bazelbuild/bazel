/*
 * Copyright 2012-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import java.io.IOException;
import java.io.InputStream;

import com.amazonaws.SdkClientException;

/**
 * {@link AWSCredentialsProvider} implementation that loads AWS security
 * credentials from a properties file on the classpath. The default
 * constructor creates a credentials provider that loads the credentials
 * from a file named <code>AwsCredentials.properties</code> on the
 * classpath, but which file to use from the classpath can also be controled
 * through the one-argument constructor.
 * <p>
 * The AWS access key ID is expected to be in the <code>accessKey</code>
 * property and the AWS secret key is expected to be in the
 * <code>secretKey</code> property.
 */
public class ClasspathPropertiesFileCredentialsProvider implements AWSCredentialsProvider {

    /** The name of the properties file to check for credentials */
    private static String DEFAULT_PROPERTIES_FILE = "AwsCredentials.properties";

    private final String credentialsFilePath;

    /**
     * Creates a new ClasspathPropertiesFileCredentialsProvider that will
     * attempt to load the <code>AwsCredentials.properties</code> file from
     * the classpath to read AWS security credentials.
     */
    public ClasspathPropertiesFileCredentialsProvider() {
        this(DEFAULT_PROPERTIES_FILE);
    }

    /**
     * Creates a new ClasspathPropertiesFileCredentialsProvider that will
     * attempt to load a custom file from the classpath to read AWS security
     * credentials.
     *
     * @param credentialsFilePath
     *            The custom classpath resource path to a properties file
     *            from which the AWS security credentials should be loaded.
     *
     *            For example,
     *            <ul>
     *              <li>com/mycompany/credentials.properties</li>
     *              <li>beta-credentials.properties</li>
     *              <li>AwsCredentials.properties</li>
     *            </ul>
     */
    public ClasspathPropertiesFileCredentialsProvider(String credentialsFilePath) {
        if (credentialsFilePath == null)
            throw new IllegalArgumentException("Credentials file path cannot be null");

        // Make sure the path is absolute
        if (!credentialsFilePath.startsWith("/")) {
            this.credentialsFilePath = "/" + credentialsFilePath;
        } else {
            this.credentialsFilePath = credentialsFilePath;
        }
    }

    public AWSCredentials getCredentials() {
        InputStream inputStream = getClass().getResourceAsStream(credentialsFilePath);
        if (inputStream == null) {
            throw new SdkClientException("Unable to load AWS credentials from the " + credentialsFilePath + " file on the classpath");
        }

        try {
            return new PropertiesCredentials(inputStream);
        } catch (IOException e) {
            throw new SdkClientException("Unable to load AWS credentials from the " + credentialsFilePath + " file on the classpath", e);
        }
    }

    public void refresh() {}

    @Override
    public String toString() {
        return getClass().getSimpleName() + "(" + credentialsFilePath + ")";
    }
}