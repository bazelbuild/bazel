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
package com.amazonaws.profile.path.config;

import com.amazonaws.SDKGlobalConfiguration;
import com.amazonaws.annotation.SdkInternalApi;
import com.amazonaws.profile.path.AwsProfileFileLocationProvider;

import java.io.File;

/**
 * If the {@value SDKGlobalConfiguration#AWS_CONFIG_FILE_ENV_VAR} environment variable is set then we source
 * the config file from the location specified.
 */
@SdkInternalApi
public class ConfigEnvVarOverrideLocationProvider implements AwsProfileFileLocationProvider {

    @Override
    public File getLocation() {
        String overrideLocation = System.getenv(SDKGlobalConfiguration.AWS_CONFIG_FILE_ENV_VAR);
        if (overrideLocation != null) {
            return new File(overrideLocation);
        }
        return null;
    }
}
