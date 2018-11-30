/*
 * Copyright 2011-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package com.amazonaws.auth.profile.internal;

import com.amazonaws.SdkClientException;
import com.amazonaws.annotation.Immutable;
import com.amazonaws.annotation.SdkInternalApi;
import com.amazonaws.auth.AWSCredentials;
import com.amazonaws.auth.AWSCredentialsProvider;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.auth.BasicSessionCredentials;
import com.amazonaws.internal.StaticCredentialsProvider;
import com.amazonaws.util.StringUtils;

/**
 * Serves credentials defined in a {@link BasicProfile}. Does validation that both access key and
 * secret key exists and are non empty.
 */
@SdkInternalApi
@Immutable
public class ProfileStaticCredentialsProvider implements AWSCredentialsProvider {

    private final BasicProfile profile;
    private final AWSCredentialsProvider credentialsProvider;

    public ProfileStaticCredentialsProvider(BasicProfile profile) {
        this.profile = profile;
        this.credentialsProvider = new StaticCredentialsProvider(fromStaticCredentials());
    }

    @Override
    public AWSCredentials getCredentials() {
        return credentialsProvider.getCredentials();
    }

    @Override
    public void refresh() {
        // No Op
    }

    private AWSCredentials fromStaticCredentials() {
        if (StringUtils.isNullOrEmpty(profile.getAwsAccessIdKey())) {
            throw new SdkClientException(String.format(
                    "Unable to load credentials into profile [%s]: AWS Access Key ID is not specified.",
                    profile.getProfileName()));
        }
        if (StringUtils.isNullOrEmpty(profile.getAwsSecretAccessKey())) {
            throw new SdkClientException(String.format(
                    "Unable to load credentials into profile [%s]: AWS Secret Access Key is not specified.",
                    profile.getAwsSecretAccessKey()));
        }

        if (profile.getAwsSessionToken() == null) {
            return new BasicAWSCredentials(profile.getAwsAccessIdKey(),
                                           profile.getAwsSecretAccessKey());
        } else {
            if (profile.getAwsSessionToken().isEmpty()) {
                throw new SdkClientException(String.format(
                        "Unable to load credentials into profile [%s]: AWS Session Token is empty.",
                        profile.getProfileName()));
            }

            return new BasicSessionCredentials(profile.getAwsAccessIdKey(),
                                               profile.getAwsSecretAccessKey(),
                                               profile.getAwsSessionToken());
        }
    }

}
