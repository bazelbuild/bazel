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

import com.amazonaws.AmazonClientException;
import com.amazonaws.SDKGlobalConfiguration;
import com.amazonaws.SdkClientException;
import com.amazonaws.internal.CredentialsEndpointProvider;
import com.amazonaws.internal.EC2CredentialsUtils;
import com.amazonaws.util.EC2MetadataUtils;
import java.io.Closeable;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * Credentials provider implementation that loads credentials from the Amazon EC2 Instance Metadata Service.
 *
 * <p>When using {@link InstanceProfileCredentialsProvider} with asynchronous refreshing it is
 * <b>strongly</b> recommended to explicitly call {@link #close()} to release the async thread.</p>
 */
public class InstanceProfileCredentialsProvider implements AWSCredentialsProvider, Closeable {

    private static final Log LOG = LogFactory.getLog(InstanceProfileCredentialsProvider.class);

    /**
     * The wait time, after which the background thread initiates a refresh to
     * load latest credentials if needed.
     */
    private static final int ASYNC_REFRESH_INTERVAL_TIME_MINUTES = 1;

    /**
     * The default InstanceProfileCredentialsProvider that can be shared by
     * multiple CredentialsProvider instance threads to shrink the amount of
     * requests to EC2 metadata service.
     */
    private static final InstanceProfileCredentialsProvider INSTANCE = new InstanceProfileCredentialsProvider();

    private final EC2CredentialsFetcher credentialsFetcher;

    /**
     * The executor service used for refreshing the credentials in the
     * background.
     */
    private volatile ScheduledExecutorService executor;

    private volatile boolean shouldRefresh = false;

    /**
     * @deprecated for the singleton method {@link #getInstance()}.
     */
    @Deprecated
    public InstanceProfileCredentialsProvider() {
        this(false);
    }

    /**
     * Spins up a new thread to refresh the credentials asynchronously if
     * refreshCredentialsAsync is set to true, otherwise the credentials will be
     * refreshed from the instance metadata service synchronously,
     *
     * <p>It is <b>strongly</b> recommended to reuse instances of this credentials provider, especially
     * when async refreshing is used since a background thread is created.</p>
     *
     * @param refreshCredentialsAsync
     *            true if credentials needs to be refreshed asynchronously else
     *            false.
     */
    public InstanceProfileCredentialsProvider(boolean refreshCredentialsAsync) {
        this(refreshCredentialsAsync, true);
    }

    /**
     * Spins up a new thread to refresh the credentials asynchronously.
     *
     * <p>It is <b>strongly</b> recommended to reuse instances of this credentials provider, especially
     * when async refreshing is used since a background thread is created.</p>
     *
     * @param eagerlyRefreshCredentialsAsync
     *            when set to false will not attempt to refresh credentials asynchronously
     *            until after a call has been made to {@link #getCredentials()} - ensures that
     *            {@link EC2CredentialsFetcher#getCredentials()} is only hit when this CredentialProvider is actually required
     */
    public static InstanceProfileCredentialsProvider createAsyncRefreshingProvider(final boolean eagerlyRefreshCredentialsAsync) {
        return new InstanceProfileCredentialsProvider(true, eagerlyRefreshCredentialsAsync);
    }

    private InstanceProfileCredentialsProvider(boolean refreshCredentialsAsync, final boolean eagerlyRefreshCredentialsAsync) {

        credentialsFetcher = new EC2CredentialsFetcher(new InstanceMetadataCredentialsEndpointProvider());

        if (!SDKGlobalConfiguration.isEc2MetadataDisabled()) {
            if (refreshCredentialsAsync) {
                executor = Executors.newScheduledThreadPool(1);
                executor.scheduleWithFixedDelay(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            if (shouldRefresh) credentialsFetcher.getCredentials();
                        } catch (AmazonClientException ace) {
                            handleError(ace);
                        } catch (RuntimeException re) {
                            handleError(re);
                        }
                    }
                }, 0, ASYNC_REFRESH_INTERVAL_TIME_MINUTES, TimeUnit.MINUTES);
            }
        }
    }

    /**
     * Returns a singleton {@link InstanceProfileCredentialsProvider} that does not refresh credentials asynchronously.
     *
     * <p>
     * See {@link #InstanceProfileCredentialsProvider(boolean)} or {@link #createAsyncRefreshingProvider(boolean)} for
     * asynchronous credentials refreshing.
     * </p>
     */
    public static InstanceProfileCredentialsProvider getInstance() {
        return INSTANCE;
    }

    private void handleError(Throwable t) {
        refresh();
        LOG.error(t.getMessage(), t);
    }

    @Override
    protected void finalize() throws Throwable {
        if (executor != null) {
            executor.shutdownNow();
        }
    }


    /**
     * {@inheritDoc}
     *
     * @throws AmazonClientException if {@link SDKGlobalConfiguration#isEc2MetadataDisabled()} is true
     */
    @Override
    public AWSCredentials getCredentials() {
        if (SDKGlobalConfiguration.isEc2MetadataDisabled()) {
            throw new AmazonClientException("AWS_EC2_METADATA_DISABLED is set to true, not loading credentials from EC2 Instance "
                                         + "Metadata service");
        }
        AWSCredentials creds = credentialsFetcher.getCredentials();
        shouldRefresh = true;
        return creds;
    }

    @Override
    public void refresh() {
        if (credentialsFetcher != null) {
            credentialsFetcher.refresh();
        }
    }

    @Override
    public void close() throws IOException {
        if (executor != null) {
            executor.shutdownNow();
            executor = null;
        }
    }

    private static class InstanceMetadataCredentialsEndpointProvider extends CredentialsEndpointProvider {
        @Override
        public URI getCredentialsEndpoint() throws URISyntaxException, IOException {
            String host = EC2MetadataUtils.getHostAddressForEC2MetadataService();

            String securityCredentialsList = EC2CredentialsUtils.getInstance().readResource(new URI(host + EC2MetadataUtils.SECURITY_CREDENTIALS_RESOURCE));
            String[] securityCredentials = securityCredentialsList.trim().split("\n");
            if (securityCredentials.length == 0) {
                throw new SdkClientException("Unable to load credentials path");
            }

            return new URI(host + EC2MetadataUtils.SECURITY_CREDENTIALS_RESOURCE + securityCredentials[0]);
        }
    }
}
