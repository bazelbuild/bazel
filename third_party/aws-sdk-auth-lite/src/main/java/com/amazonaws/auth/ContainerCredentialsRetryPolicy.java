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

import java.io.IOException;

import com.amazonaws.annotation.SdkInternalApi;
import com.amazonaws.retry.internal.CredentialsEndpointRetryParameters;
import com.amazonaws.retry.internal.CredentialsEndpointRetryPolicy;

@SdkInternalApi
class ContainerCredentialsRetryPolicy implements CredentialsEndpointRetryPolicy {

    /** Max number of times a request is retried before failing. */
    private static final int MAX_RETRIES = 5;

    private static ContainerCredentialsRetryPolicy instance;

    private ContainerCredentialsRetryPolicy() {

    }

    public static ContainerCredentialsRetryPolicy getInstance() {
        if (instance == null) {
            instance = new ContainerCredentialsRetryPolicy();
        }
        return instance;
    }

    @Override
    public boolean shouldRetry(int retriesAttempted, CredentialsEndpointRetryParameters retryParams) {
        if (retriesAttempted >= MAX_RETRIES) {
            return false;
        }

        Integer statusCode = retryParams.getStatusCode();
        if (statusCode != null && statusCode >= 500 && statusCode < 600) {
            return true;
        }

        if (retryParams.getException() != null && retryParams.getException() instanceof IOException) {
            return true;
        }

        return false;
    }

}
