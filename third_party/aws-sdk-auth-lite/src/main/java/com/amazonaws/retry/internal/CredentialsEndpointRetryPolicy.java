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
package com.amazonaws.retry.internal;

import com.amazonaws.annotation.SdkInternalApi;
import com.amazonaws.auth.ContainerCredentialsProvider;
import com.amazonaws.auth.InstanceProfileCredentialsProvider;

/**
 * Custom retry policy for credentials providers ({@link InstanceProfileCredentialsProvider},
 * {@link ContainerCredentialsProvider}) that retrieve credentials from a local endpoint in EC2 host.
 *
 * Internal use only.
 */
@SdkInternalApi
public interface CredentialsEndpointRetryPolicy {

    public static final CredentialsEndpointRetryPolicy NO_RETRY = new CredentialsEndpointRetryPolicy() {

        @Override
        public boolean shouldRetry(int retriesAttempted, CredentialsEndpointRetryParameters retryParams) {
            return false;
        }
    };

    /**
     * Returns whether a failed request should be retried.
     *
     * @param retriesAttempted
     *            The number of times the current request has been
     *            attempted.
     *
     * @return True if the failed request should be retried.
     */
    boolean shouldRetry(int retriesAttempted, CredentialsEndpointRetryParameters retryParams);

}
