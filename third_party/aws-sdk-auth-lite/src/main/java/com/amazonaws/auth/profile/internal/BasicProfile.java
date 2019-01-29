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

import java.util.Collections;
import java.util.Map;

/**
 * Represents a CLI style config profile with a name and simple properties. Provides convenient
 * access to the properties the Java SDK deals with and also raw access to all properties.
 */
@Immutable
@SdkInternalApi
public class BasicProfile {


    private final String profileName;
    private final Map<String, String> properties;

    public BasicProfile(String profileName,
                        Map<String, String> properties) {
        this.profileName = profileName;
        this.properties = properties;
    }

    /**
     * @return The name of this profile.
     */
    public String getProfileName() {
        return profileName;
    }

    /**
     * Returns a map of profile properties included in this Profile instance. The returned
     * properties corresponds to how this profile is described in the credential profiles file,
     * i.e., profiles with basic credentials consist of two properties {"aws_access_key_id",
     * "aws_secret_access_key"} and profiles with session credentials have three properties, with an
     * additional "aws_session_token" property.
     */
    public Map<String, String> getProperties() {
        return Collections.unmodifiableMap(properties);
    }

    /**
     * Returns the value of a specific property that is included in this Profile instance.
     *
     * @see BasicProfile#getProperties()
     */
    public String getPropertyValue(String propertyName) {
        return getProperties().get(propertyName);
    }

    public String getAwsAccessIdKey() {
        return getPropertyValue(ProfileKeyConstants.AWS_ACCESS_KEY_ID);
    }

    public String getAwsSecretAccessKey() {
        return getPropertyValue(ProfileKeyConstants.AWS_SECRET_ACCESS_KEY);
    }

    public String getAwsSessionToken() {
        return getPropertyValue(ProfileKeyConstants.AWS_SESSION_TOKEN);
    }

    public String getRoleArn() {
        return getPropertyValue(ProfileKeyConstants.ROLE_ARN);
    }

    public String getRoleSourceProfile() {
        return getPropertyValue(ProfileKeyConstants.SOURCE_PROFILE);
    }

    public String getRoleSessionName() {
        return getPropertyValue(ProfileKeyConstants.ROLE_SESSION_NAME);
    }

    public String getRoleExternalId() {
        return getPropertyValue(ProfileKeyConstants.EXTERNAL_ID);
    }

    public String getRegion() {
        return getPropertyValue(ProfileKeyConstants.REGION);
    }

    public boolean isRoleBasedProfile() {
        return getRoleArn() != null;
    }
}
