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
 * Base exception class for any errors that occur while attempting to use an AWS
 * client from AWS SDK for Java to make service calls to Amazon Web Services.
 *
 * Error responses from services will be handled as AmazonServiceExceptions.
 * This class is primarily for errors that occur when unable to get a response
 * from a service, or when the client is unable to parse the response from a
 * service. For example, if a caller tries to use a client to make a service
 * call, but no network connection is present, an AmazonClientException will be
 * thrown to indicate that the client wasn't able to successfully make the
 * service call, and no information from the service is available.
 *
 * Note : If the SDK is able to parse the response; but doesn't recognize the
 * error code from the service, an AmazonServiceException is thrown
 *
 * Callers should typically deal with exceptions through AmazonServiceException,
 * which represent error responses returned by services. AmazonServiceException
 * has much more information available for callers to appropriately deal with
 * different types of errors that can occur.
 *
 * @see AmazonServiceException
 */
public class AmazonClientException extends SdkBaseException {
    private static final long serialVersionUID = 1L;

    /**
     * Creates a new AmazonClientException with the specified message, and root
     * cause.
     * 
     * @param message
     *            An error message describing why this exception was thrown.
     * @param t
     *            The underlying cause of this exception.
     */
    public AmazonClientException(String message, Throwable t) {
        super(message, t);
    }

    /**
     * Creates a new AmazonClientException with the specified message.
     * 
     * @param message
     *            An error message describing why this exception was thrown.
     */
    public AmazonClientException(String message) {
        super(message);
    }

    public AmazonClientException(Throwable t) {
        super(t);
    }

}
