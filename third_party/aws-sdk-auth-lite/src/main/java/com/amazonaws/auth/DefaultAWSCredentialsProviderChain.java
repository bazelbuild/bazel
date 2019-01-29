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

import com.amazonaws.auth.profile.ProfileCredentialsProvider;

/**
 * AWS credentials provider chain that looks for credentials in this order:
 * <ul>
 *   <li>Environment Variables -
 *      <code>AWS_ACCESS_KEY_ID</code> and <code>AWS_SECRET_ACCESS_KEY</code>
 *      (RECOMMENDED since they are recognized by all the AWS SDKs and CLI except for .NET),
 *      or <code>AWS_ACCESS_KEY</code> and <code>AWS_SECRET_KEY</code> (only recognized by Java SDK)
 *   </li>
 *   <li>Java System Properties - aws.accessKeyId and aws.secretKey</li>
 *   <li>Credential profiles file at the default location (~/.aws/credentials) shared by all AWS SDKs and the AWS CLI</li>
 *   <li>Credentials delivered through the Amazon EC2 container service if AWS_CONTAINER_CREDENTIALS_RELATIVE_URI" environment variable is set
 *   and security manager has permission to access the variable,</li>
 *   <li>Instance profile credentials delivered through the Amazon EC2 metadata service</li>
 * </ul>
 *
 * @see EnvironmentVariableCredentialsProvider
 * @see SystemPropertiesCredentialsProvider
 * @see ProfileCredentialsProvider
 * @see EC2ContainerCredentialsProviderWrapper
 */
public class DefaultAWSCredentialsProviderChain extends AWSCredentialsProviderChain {

    private static final DefaultAWSCredentialsProviderChain INSTANCE
        = new DefaultAWSCredentialsProviderChain();

    public DefaultAWSCredentialsProviderChain() {
        super(new EnvironmentVariableCredentialsProvider(),
              new SystemPropertiesCredentialsProvider(),
              new ProfileCredentialsProvider(),
              new EC2ContainerCredentialsProviderWrapper());
    }

    public static DefaultAWSCredentialsProviderChain getInstance() {
        return INSTANCE;
    }
}
