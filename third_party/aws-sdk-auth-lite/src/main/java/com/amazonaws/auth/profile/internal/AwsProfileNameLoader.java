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

import com.amazonaws.annotation.Immutable;
import com.amazonaws.annotation.SdkInternalApi;
import com.amazonaws.util.StringUtils;

/**
 * Loads profile name from the usual places or uses the default profile name.
 */
@SdkInternalApi
@Immutable
public class AwsProfileNameLoader {

    /**
     * Name of the default profile as specified in the configuration file.
     */
    public static final String DEFAULT_PROFILE_NAME = "default";

    /**
     * Environment variable name for overriding the default AWS profile
     */
    public static final String AWS_PROFILE_ENVIRONMENT_VARIABLE = "AWS_PROFILE";

    /**
     * System property name for overriding the default AWS profile
     */
    public static final String AWS_PROFILE_SYSTEM_PROPERTY = "aws.profile";

    public static final AwsProfileNameLoader INSTANCE = new AwsProfileNameLoader();

    private AwsProfileNameLoader() {
    }

    /**
     * TODO The order would make more sense as System Property, Environment Variable, Default
     * Profile name but we have to keep the current order for backwards compatiblity. Consider
     * changing this in a future major version.
     */
    public final String loadProfileName() {
        final String profileEnvVarOverride = getEnvProfileName();
        if (!StringUtils.isNullOrEmpty(profileEnvVarOverride)) {
            return profileEnvVarOverride;
        } else {
            final String profileSysPropOverride = getSysPropertyProfileName();
            if (!StringUtils.isNullOrEmpty(profileSysPropOverride)) {
                return profileSysPropOverride;
            } else {
                return DEFAULT_PROFILE_NAME;
            }
        }
    }

    private String getSysPropertyProfileName() {
        return StringUtils.trim(System.getProperty(AWS_PROFILE_SYSTEM_PROPERTY));
    }

    private String getEnvProfileName() {
        return StringUtils.trim(System.getenv(AWS_PROFILE_ENVIRONMENT_VARIABLE));
    }
}
