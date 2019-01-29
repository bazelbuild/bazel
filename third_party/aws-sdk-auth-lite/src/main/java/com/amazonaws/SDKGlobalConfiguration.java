/*
 * Copyright 2010-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package com.amazonaws;

/**
 * SDKGlobalConfiguration is to configure any global settings
 */
public class SDKGlobalConfiguration {
    /////////////////////// System Properties ///////////////////////

    /** System property name for the AWS access key ID */
    public static final String ACCESS_KEY_SYSTEM_PROPERTY = "aws.accessKeyId";

    /** System property name for the AWS secret key */
    public  static final String SECRET_KEY_SYSTEM_PROPERTY = "aws.secretKey";

    /**
     * System property name for the AWS session token
     */
    public static final String SESSION_TOKEN_SYSTEM_PROPERTY = "aws.sessionToken";

    /**
     * System property for overriding the Amazon EC2 Instance Metadata Service
     * endpoint.
     */
    public static final String EC2_METADATA_SERVICE_OVERRIDE_SYSTEM_PROPERTY =
        "com.amazonaws.sdk.ec2MetadataServiceEndpointOverride";

    /**
     * By default, the AmazonS3Client will continue to use the legacy
     * S3Signer to authenticate requests it makes to S3 in regions that
     * support the older protocol. Setting this property to anything other
     * than null will cause the client to upgrade to Signature Version 4
     * whenever it has been configured with an explicit region (which is a
     * required parameter for Signature Version 4). The client will continue
     * to use the older signature protocol when not configured with a region
     * to avoid breaking existing applications.
     * <p>
     * Signature Version 4 is more secure than the legacy S3Signer, but
     * requires calculating a SHA-256 hash of the entire request body which
     * can be expensive for large upload requests.
     */
    @Deprecated
    public static final String ENABLE_S3_SIGV4_SYSTEM_PROPERTY =
        "com.amazonaws.services.s3.enableV4";

    /**
     * Like {@link #ENABLE_S3_SIGV4_SYSTEM_PROPERTY}, but causes the client to
     * always use Signature Version 4, assuming a region of
     * &quot;us-east-1&quot; if no explicit region has been configured. This
     * guarantees that the more secure authentication protocol will be used,
     * but will cause authentication failures in code that accesses buckets in
     * regions other than US Standard without explicitly configuring a region.
     */
    @Deprecated
    public static final String ENFORCE_S3_SIGV4_SYSTEM_PROPERTY =
        "com.amazonaws.services.s3.enforceV4";

    /**
     * @deprecated with {@link AmazonWebServiceRequest#getRequestClientOptions()}
     * and {@link RequestClientOptions#setReadLimit(int)}.
     * <p>
     * The default size of the buffer when uploading data from a stream. A
     * buffer of this size will be created and filled with the first bytes from
     * a stream being uploaded so that any transmit errors that occur in that
     * section of the data can be automatically retried without the caller's
     * intervention.
     * <p>
     * If not set, the default value of 128 KB will be used.
     */
    @Deprecated
    public static final String DEFAULT_S3_STREAM_BUFFER_SIZE =
        "com.amazonaws.sdk.s3.defaultStreamBufferSize";

    /**
     * @deprecated by {@link #DEFAULT_METRICS_SYSTEM_PROPERTY}.
     *
     * Internal system property to enable timing info collection.
     */
    @Deprecated
    public static final String PROFILING_SYSTEM_PROPERTY =
        "com.amazonaws.sdk.enableRuntimeProfiling";

    /////////////////////// Environment Variables ///////////////////////
    /** Environment variable name for the AWS access key ID */
    public static final String ACCESS_KEY_ENV_VAR = "AWS_ACCESS_KEY_ID";

    /** Alternate environment variable name for the AWS access key ID */
    public static final String ALTERNATE_ACCESS_KEY_ENV_VAR = "AWS_ACCESS_KEY";

    /** Environment variable name for the AWS secret key */
    public static final String SECRET_KEY_ENV_VAR = "AWS_SECRET_KEY";

    /** Alternate environment variable name for the AWS secret key */
    public static final String ALTERNATE_SECRET_KEY_ENV_VAR = "AWS_SECRET_ACCESS_KEY";

    /** Environment variable name for the AWS session token */
    public static final String AWS_SESSION_TOKEN_ENV_VAR = "AWS_SESSION_TOKEN";

    /**
     * Environment variable to set an alternate path to the shared config file (default path is
     * ~/.aws/config).
     */
    public static final String AWS_CONFIG_FILE_ENV_VAR = "AWS_CONFIG_FILE";

    /**
     * Environment variable to disable loading credentials or regions from EC2 Metadata instance service.
     */
    public static final String AWS_EC2_METADATA_DISABLED_ENV_VAR = "AWS_EC2_METADATA_DISABLED";

    /**
     * System property to disable loading credentials or regions from EC2 Metadata instance service.
     */
    public static final String AWS_EC2_METADATA_DISABLED_SYSTEM_PROPERTY = "com.amazonaws.sdk.disableEc2Metadata";


    /**
     * @deprecated by {@link SDKGlobalTime#setGlobalTimeOffset(int)}
     */
    @Deprecated
    public  static void setGlobalTimeOffset(int timeOffset) {
        SDKGlobalTime.setGlobalTimeOffset(timeOffset);
    }

    /**
     * @deprecated by {@link SDKGlobalTime#getGlobalTimeOffset()}
     */
    @Deprecated
    public static int getGlobalTimeOffset() {
        return SDKGlobalTime.getGlobalTimeOffset();
    }

    public static boolean isEc2MetadataDisabled() {
        return isPropertyTrue(System.getProperty(AWS_EC2_METADATA_DISABLED_SYSTEM_PROPERTY)) ||
               isPropertyTrue(System.getenv(AWS_EC2_METADATA_DISABLED_ENV_VAR));
    }

    private static boolean isPropertyTrue(final String property) {
        if (property != null && property.equalsIgnoreCase("true")) {
            return true;
        }
        return false;
    }
}
