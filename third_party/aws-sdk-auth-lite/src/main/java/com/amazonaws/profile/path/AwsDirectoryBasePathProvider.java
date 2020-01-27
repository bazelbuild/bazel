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
package com.amazonaws.profile.path;

import com.amazonaws.SdkClientException;
import com.amazonaws.annotation.SdkInternalApi;

import java.io.File;

/**
 * Base provider for all location providers that source a file from the ~/.aws directory.
 */
@SdkInternalApi
public abstract class AwsDirectoryBasePathProvider implements AwsProfileFileLocationProvider {

    /**
     * @return File of ~/.aws directory.
     */
    protected final File getAwsDirectory() {
        return new File(getHomeDirectory(), ".aws");
    }

    private String getHomeDirectory() {
        String userHome = System.getProperty("user.home");
        if (userHome == null) {
            throw new SdkClientException(
                    "Unable to load AWS profiles: " + "'user.home' System property is not set.");
        }
        return userHome;
    }
}
