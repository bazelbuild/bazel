/*
 * Copyright 2011-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import static com.amazonaws.auth.ContainerCredentialsProvider.CONTAINER_CREDENTIALS_FULL_URI;
import static com.amazonaws.auth.ContainerCredentialsProvider.ECS_CONTAINER_CREDENTIALS_PATH;

import com.amazonaws.auth.ContainerCredentialsProvider.ECSCredentialsEndpointProvider;
import com.amazonaws.auth.ContainerCredentialsProvider.FullUriCredentialsEndpointProvider;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * <p>
 * {@link AWSCredentialsProvider} that loads credentials from an Amazon Container (e.g. EC2)
 *
 * Credentials are solved in the following order:
 * <ol>
 *     <li>
 *         If environment variable "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI" is
 *         set (typically on EC2) it is used to hit the metadata service at the following endpoint: http://169.254.170.2
 *     </li>
 *     <li>
 *         If environment variable "AWS_CONTAINER_CREDENTIALS_FULL_URI" is
 *         set it is used to hit a metadata service at that URI. <br/> Optionally an authorization token can be included
 *         in the "Authorization" header of the request by setting the "AWS_CONTAINER_AUTHORIZATION_TOKEN" environment variable.
 *     </li>
 *     <li>
 *         If neither of the above environment variables are specified credentials are attempted to be loaded from Amazon EC2
 *         Instance Metadata Service using the {@link InstanceProfileCredentialsProvider}.
 *     </li>
 * </ol>
 */
public class EC2ContainerCredentialsProviderWrapper implements AWSCredentialsProvider {

    private static final Log LOG = LogFactory.getLog(EC2ContainerCredentialsProviderWrapper.class);

    private final AWSCredentialsProvider provider;

    public EC2ContainerCredentialsProviderWrapper() {
        provider = initializeProvider();
    }

    private AWSCredentialsProvider initializeProvider() {
        try {
            if (System.getenv(ECS_CONTAINER_CREDENTIALS_PATH) != null) {
                return new ContainerCredentialsProvider(new ECSCredentialsEndpointProvider());
            }
            if (System.getenv(CONTAINER_CREDENTIALS_FULL_URI) != null) {
                return new ContainerCredentialsProvider(new FullUriCredentialsEndpointProvider());
            }
            return InstanceProfileCredentialsProvider.getInstance();
        } catch (SecurityException securityException) {
            LOG.debug("Security manager did not allow access to the ECS credentials environment variable " + ECS_CONTAINER_CREDENTIALS_PATH +
                "or the container full URI environment variable " + CONTAINER_CREDENTIALS_FULL_URI
                        + ". Please provide access to this environment variable if you want to load credentials from ECS Container.");
            return InstanceProfileCredentialsProvider.getInstance();
        }
    }

    @Override
    public AWSCredentials getCredentials() {
        return provider.getCredentials();
    }

    @Override
    public void refresh() {
       provider.refresh();
    }
}
