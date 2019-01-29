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
package com.amazonaws.profile.path.cred;

import com.amazonaws.annotation.SdkInternalApi;
import com.amazonaws.profile.path.AwsProfileFileLocationProvider;

import java.io.File;

/**
 * If {@value #CREDENTIAL_PROFILES_FILE_ENVIRONMENT_VARIABLE} environment variable is set then the
 * shared credentials file is source from it's location.
 */
@SdkInternalApi
public class CredentialsEnvVarOverrideLocationProvider implements AwsProfileFileLocationProvider {

    private static final String CREDENTIAL_PROFILES_FILE_ENVIRONMENT_VARIABLE = "AWS_CREDENTIAL_PROFILES_FILE";

    @Override
    public File getLocation() {
        String credentialProfilesFileOverride = System
                .getenv(CREDENTIAL_PROFILES_FILE_ENVIRONMENT_VARIABLE);
        if (credentialProfilesFileOverride == null) {
            return null;
        } else {
            return new File(credentialProfilesFileOverride);
        }
    }
}
