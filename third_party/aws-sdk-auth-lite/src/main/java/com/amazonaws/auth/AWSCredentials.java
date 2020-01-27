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
package com.amazonaws.auth;

/**
 * Provides access to the AWS credentials used for accessing AWS services: AWS
 * access key ID and secret access key. These credentials are used to securely
 * sign requests to AWS services.
 * <p>
 * A basic implementation of this interface is provided in
 * {@link BasicAWSCredentials}, but callers are free to provide their own
 * implementation, for example, to load AWS credentials from an encrypted file.
 * <p>
 * For more details on AWS access keys, see: <a href="http://docs.amazonwebservices.com/AWSSecurityCredentials/1.0/AboutAWSCredentials.html#AccessKeys"
 * >http://docs.amazonwebservices.com/AWSSecurityCredentials/1.0/
 * AboutAWSCredentials.html#AccessKeys</a>
 */
public interface AWSCredentials {

    /**
     * Returns the AWS access key ID for this credentials object. 
     * 
     * @return The AWS access key ID for this credentials object. 
     */
    public String getAWSAccessKeyId();

    /**
     * Returns the AWS secret access key for this credentials object.
     * 
     * @return The AWS secret access key for this credentials object.
     */
    public String getAWSSecretKey();

}
