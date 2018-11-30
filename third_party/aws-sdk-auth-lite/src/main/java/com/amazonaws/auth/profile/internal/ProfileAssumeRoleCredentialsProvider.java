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
import com.amazonaws.auth.profile.internal.securitytoken.ProfileCredentialsService;
import com.amazonaws.auth.profile.internal.securitytoken.RoleInfo;
import com.amazonaws.util.StringUtils;

/**
 * Serves assume role credentials defined in a {@link BasicProfile}. If a profile defines the
 * role_arn property then the profile is treated as an assume role profile. Does basic validation
 * that the role exists and the source (long lived) credentials are valid.
 */
@SdkInternalApi
@Immutable
public class ProfileAssumeRoleCredentialsProvider implements AWSCredentialsProvider {


    private final AllProfiles allProfiles;
    private final BasicProfile profile;
    private final ProfileCredentialsService profileCredentialsService;
    private final AWSCredentialsProvider assumeRoleCredentialsProvider;

    public ProfileAssumeRoleCredentialsProvider(ProfileCredentialsService profileCredentialsService,
                                                AllProfiles allProfiles, BasicProfile profile) {
        this.allProfiles = allProfiles;
        this.profile = profile;
        this.profileCredentialsService = profileCredentialsService;
        this.assumeRoleCredentialsProvider = fromAssumeRole();
    }

    @Override
    public AWSCredentials getCredentials() {
        return assumeRoleCredentialsProvider.getCredentials();
    }

    @Override
    public void refresh() {
    }

    private AWSCredentialsProvider fromAssumeRole() {
        if (StringUtils.isNullOrEmpty(profile.getRoleSourceProfile())) {
            throw new SdkClientException(String.format(
                    "Unable to load credentials from profile [%s]: Source profile name is not specified",
                    profile.getProfileName()));
        }

        final BasicProfile sourceProfile = allProfiles
                .getProfile(this.profile.getRoleSourceProfile());
        if (sourceProfile == null) {
            throw new SdkClientException(String.format(
                    "Unable to load source profile [%s]: Source profile was not found [%s]",
                    profile.getProfileName(), profile.getRoleSourceProfile()));
        }
        AWSCredentials sourceCredentials = new ProfileStaticCredentialsProvider(sourceProfile)
                .getCredentials();


        final String roleSessionName = (this.profile.getRoleSessionName() == null) ?
                "aws-sdk-java-" + System.currentTimeMillis() : this.profile.getRoleSessionName();

        RoleInfo roleInfo = new RoleInfo().withRoleArn(this.profile.getRoleArn())
                .withRoleSessionName(roleSessionName)
                .withExternalId(this.profile.getRoleExternalId())
                .withLongLivedCredentials(sourceCredentials);
        return profileCredentialsService.getAssumeRoleCredentialsProvider(roleInfo);
    }
}
