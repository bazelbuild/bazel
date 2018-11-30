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

import com.amazonaws.SdkClientException;
import com.amazonaws.internal.CredentialsEndpointProvider;
import com.amazonaws.retry.internal.CredentialsEndpointRetryPolicy;
import com.amazonaws.util.CollectionUtils;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * <p>
 * {@link AWSCredentialsProvider} implementation that loads credentials
 * from an Amazon Elastic Container.
 * </p>
 * <p>
 * By default, the URI path is retrieved from the environment variable
 * "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI" in the container's environment.
 * </p>
 */
public class ContainerCredentialsProvider implements AWSCredentialsProvider {

    /** Environment variable to get the Amazon ECS credentials resource path. */
    static final String ECS_CONTAINER_CREDENTIALS_PATH = "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI";

    /** Environment variable to get the full URI for a credentials path */
    static final String CONTAINER_CREDENTIALS_FULL_URI = "AWS_CONTAINER_CREDENTIALS_FULL_URI";

    static final String CONTAINER_AUTHORIZATION_TOKEN = "AWS_CONTAINER_AUTHORIZATION_TOKEN";

    private static final Set<String> ALLOWED_FULL_URI_HOSTS = allowedHosts();

    /** Default endpoint to retreive the Amazon ECS Credentials. */
    private static final String ECS_CREDENTIALS_ENDPOINT = "http://169.254.170.2";

    private final EC2CredentialsFetcher credentialsFetcher;

    /**
     * @deprecated use {@link #ContainerCredentialsProvider(CredentialsEndpointProvider)}
     */
    @Deprecated
    public ContainerCredentialsProvider() {
        this(new ECSCredentialsEndpointProvider());
    }

    public ContainerCredentialsProvider(CredentialsEndpointProvider credentialsEndpointProvider) {
        this.credentialsFetcher = new EC2CredentialsFetcher(credentialsEndpointProvider);
    }

    @Override
    public AWSCredentials getCredentials() {
        return credentialsFetcher.getCredentials();
    }

    @Override
    public void refresh() {
        credentialsFetcher.refresh();
    }

    public Date getCredentialsExpiration() {
        return credentialsFetcher.getCredentialsExpiration();
    }


    static class ECSCredentialsEndpointProvider extends CredentialsEndpointProvider {
        @Override
        public URI getCredentialsEndpoint() throws URISyntaxException {
            String path = System.getenv(ECS_CONTAINER_CREDENTIALS_PATH);
            if (path == null) {
                throw new SdkClientException(
                        "The environment variable " + ECS_CONTAINER_CREDENTIALS_PATH + " is empty");
            }

            return new URI(ECS_CREDENTIALS_ENDPOINT + path);
        }
        @Override
        public CredentialsEndpointRetryPolicy getRetryPolicy() {
            return ContainerCredentialsRetryPolicy.getInstance();
        }

    }

    /**
     * A URI resolver that uses environment variable {@value CONTAINER_CREDENTIALS_FULL_URI} as the URI
     * for the metadata service.
     * Optionally an authorization token can be provided using the {@value CONTAINER_AUTHORIZATION_TOKEN} environment variable.
     */
    static class FullUriCredentialsEndpointProvider extends CredentialsEndpointProvider {

        @Override
        public URI getCredentialsEndpoint() throws URISyntaxException {
            String fullUri = System.getenv(CONTAINER_CREDENTIALS_FULL_URI);
            if (fullUri == null || fullUri.length() == 0) {
                throw new SdkClientException("The environment variable " + CONTAINER_CREDENTIALS_FULL_URI + " is empty");
            }

            URI uri = new URI(fullUri);

            if (!ALLOWED_FULL_URI_HOSTS.contains(uri.getHost())) {
                throw new SdkClientException("The full URI (" + uri + ") contained withing environment variable " +
                    CONTAINER_CREDENTIALS_FULL_URI + " has an invalid host. Host can only be one of [" +
                    CollectionUtils.join(ALLOWED_FULL_URI_HOSTS, ", ") + "]");
            }

            return uri;
        }

        @Override
        public Map<String, String> getHeaders() {
            if (System.getenv(CONTAINER_AUTHORIZATION_TOKEN) != null) {
                return Collections.singletonMap("Authorization", System.getenv(CONTAINER_AUTHORIZATION_TOKEN));
            }
            return new HashMap<String, String>();
        }
    }

    private static Set<String> allowedHosts() {
        HashSet<String> hosts = new HashSet<String>();
        hosts.add("127.0.0.1");
        hosts.add("localhost");
        return Collections.unmodifiableSet(hosts);
    }

}
