/*
 * Copyright 2013-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package com.amazonaws.util;

import static com.amazonaws.SDKGlobalConfiguration.EC2_METADATA_SERVICE_OVERRIDE_SYSTEM_PROPERTY;

import com.amazonaws.SDKGlobalConfiguration;

/**
 * Utility class for retrieving Amazon EC2 instance metadata.<br>
 * You can use the data to build more generic AMIs that can be modified by
 * configuration files supplied at launch time. For example, if you run web
 * servers for various small businesses, they can all use the same AMI and
 * retrieve their content from the Amazon S3 bucket you specify at launch. To
 * add a new customer at any time, simply create a bucket for the customer, add
 * their content, and launch your AMI.<br>
 *
 * <p>
 * You can disable the use of the EC2 Instance meta data service by either setting the
 * {@link SDKGlobalConfiguration#AWS_EC2_METADATA_DISABLED_ENV_VAR} or
 * {@link SDKGlobalConfiguration#AWS_EC2_METADATA_DISABLED_SYSTEM_PROPERTY} to 'true'(not case sensitive).
 *
 * More information about Amazon EC2 Metadata
 *
 * @see <a
 *      href="http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AESDG-chapter-instancedata.html">Amazon
 *      EC2 User Guide: Instance Metadata</a>
 */
public class EC2MetadataUtils {

    /** Default endpoint for the Amazon EC2 Instance Metadata Service. */
    private static final String EC2_METADATA_SERVICE_URL = "http://169.254.169.254";

    /** Default resource path for credentials in the Amazon EC2 Instance Metadata Service. */
    public static final String SECURITY_CREDENTIALS_RESOURCE = "/latest/meta-data/iam/security-credentials/";

    /**
     * Returns the host address of the Amazon EC2 Instance Metadata Service.
     */
    public static String getHostAddressForEC2MetadataService() {
        String host = System.getProperty(EC2_METADATA_SERVICE_OVERRIDE_SYSTEM_PROPERTY);
        return host != null ? host : EC2_METADATA_SERVICE_URL;
    }

}
