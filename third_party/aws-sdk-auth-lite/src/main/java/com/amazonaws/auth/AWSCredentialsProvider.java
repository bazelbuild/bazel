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
package com.amazonaws.auth;

/**
 * Interface for providing AWS credentials. Implementations are free to use any
 * strategy for providing AWS credentials, such as simply providing static
 * credentials that don't change, or more complicated implementations, such as
 * integrating with existing key management systems.
 */
public interface AWSCredentialsProvider {

    /**
     * Returns AWSCredentials which the caller can use to authorize an AWS request.
     * Each implementation of AWSCredentialsProvider can chose its own strategy for
     * loading credentials.  For example, an implementation might load credentials
     * from an existing key management system, or load new credentials when
     * credentials are rotated.
     *
     * @return AWSCredentials which the caller can use to authorize an AWS request.
     */
    public AWSCredentials getCredentials();

    /**
     * Forces this credentials provider to refresh its credentials. For many
     * implementations of credentials provider, this method may simply be a
     * no-op, such as any credentials provider implementation that vends
     * static/non-changing credentials. For other implementations that vend
     * different credentials through out their lifetime, this method should
     * force the credentials provider to refresh its credentials.
     */
    public void refresh();

}
