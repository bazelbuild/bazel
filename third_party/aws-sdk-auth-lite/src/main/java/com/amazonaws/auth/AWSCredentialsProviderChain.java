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

import java.util.LinkedList;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.amazonaws.SdkClientException;

/**
 * {@link AWSCredentialsProvider} implementation that chains together multiple
 * credentials providers. When a caller first requests credentials from this provider,
 * it calls all the providers in the chain, in the original order specified,
 * until one can provide credentials, and then returns those credentials. If all
 * of the credential providers in the chain have been called, and none of them
 * can provide credentials, then this class will throw an exception indicated
 * that no credentials are available.
 * <p>
 * By default, this class will remember the first credentials provider in the chain
 * that was able to provide credentials, and will continue to use that provider when
 * credentials are requested in the future, instead of traversing the chain each time.
 * This behavior can be controlled through the {@link #setReuseLastProvider(boolean)} method.
 */
public class AWSCredentialsProviderChain implements AWSCredentialsProvider {

    private static final Log log = LogFactory.getLog(AWSCredentialsProviderChain.class);

    private final List<AWSCredentialsProvider> credentialsProviders =
            new LinkedList<AWSCredentialsProvider>();

    private boolean reuseLastProvider = true;
    private AWSCredentialsProvider lastUsedProvider;

    /**
     * Constructs a new AWSCredentialsProviderChain with the specified credential providers. When
     * credentials are requested from this provider, it will call each of these credential providers
     * in the same order specified here until one of them returns AWS security credentials.
     *
     * @param credentialsProviders
     *            The chain of credentials providers.
     */
    public AWSCredentialsProviderChain(List<? extends AWSCredentialsProvider> credentialsProviders) {
        if (credentialsProviders == null || credentialsProviders.size() == 0) {
            throw new IllegalArgumentException("No credential providers specified");
        }
        this.credentialsProviders.addAll(credentialsProviders);
    }

    /**
     * Constructs a new AWSCredentialsProviderChain with the specified credential providers. When
     * credentials are requested from this provider, it will call each of these credential providers
     * in the same order specified here until one of them returns AWS security credentials.
     *
     * @param credentialsProviders
     *            The chain of credentials providers.
     */
    public AWSCredentialsProviderChain(AWSCredentialsProvider... credentialsProviders) {
        if (credentialsProviders == null || credentialsProviders.length == 0) {
            throw new IllegalArgumentException("No credential providers specified");
        }

        for (AWSCredentialsProvider provider : credentialsProviders) {
            this.credentialsProviders.add(provider);
        }
    }

    /**
     * Returns true if this chain will reuse the last successful credentials
     * provider for future credentials requests, otherwise, false if it will
     * search through the chain each time.
     *
     * @return True if this chain will reuse the last successful credentials
     *         provider for future credentials requests.
     */
    public boolean getReuseLastProvider() {
        return reuseLastProvider;
    }

    /**
     * Enables or disables caching of the last successful credentials provider
     * in this chain. Reusing the last successful credentials provider will
     * typically return credentials faster than searching through the chain.
     *
     * @param b
     *            Whether to enable or disable reusing the last successful
     *            credentials provider for future credentials requests instead
     *            of searching through the whole chain.
     */
    public void setReuseLastProvider(boolean b) {
        this.reuseLastProvider = b;
    }

    public AWSCredentials getCredentials() {
        if (reuseLastProvider && lastUsedProvider != null) {
            return lastUsedProvider.getCredentials();
        }

        for (AWSCredentialsProvider provider : credentialsProviders) {
            try {
                AWSCredentials credentials = provider.getCredentials();

                if (credentials.getAWSAccessKeyId() != null &&
                    credentials.getAWSSecretKey() != null) {
                    log.debug("Loading credentials from " + provider.toString());

                    lastUsedProvider = provider;
                    return credentials;
                }
            } catch (Exception e) {
                // Ignore any exceptions and move onto the next provider
                log.debug("Unable to load credentials from " + provider.toString() +
                          ": " + e.getMessage());
            }
        }

        throw new SdkClientException("Unable to load AWS credentials from any provider in the chain");
    }

    public void refresh() {
        for (AWSCredentialsProvider provider : credentialsProviders) {
            provider.refresh();
        }
    }
}
