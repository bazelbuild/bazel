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
 * Extension of SdkClientException that represents an error response returned
 * by an Amazon web service. Receiving an exception of this type indicates that
 * the caller's request was correctly transmitted to the service, but for some
 * reason, the service was not able to process it, and returned an error
 * response instead.
 * <p>
 * AmazonServiceException provides callers several pieces of
 * information that can be used to obtain more information about the error and
 * why it occurred. In particular, the errorType field can be used to determine
 * if the caller's request was invalid, or the service encountered an error on
 * the server side while processing it.
 */
public class AmazonServiceException extends SdkClientException {
    private static final long serialVersionUID = 1L;

    /**
     * The unique AWS identifier for the service request the caller made. The
     * AWS request ID can uniquely identify the AWS request, and is used for
     * reporting an error to AWS support team.
     */
    private String requestId;

    /**
     * The AWS error code represented by this exception (ex:
     * InvalidParameterValue).
     */
    private String errorCode;

    /**
     * The error message as returned by the service.
     */
    private String errorMessage;

    /** The HTTP status code that was returned with this error */
    private int statusCode;

    /**
     * The name of the Amazon service that sent this error response.
     */
    private String serviceName;

    /**
     * Constructs a new AmazonServiceException with the specified message.
     *
     * @param errorMessage
     *            An error message describing what went wrong.
     */
    public AmazonServiceException(String errorMessage) {
        super((String)null);
        this.errorMessage = errorMessage;
    }

    /**
     * Constructs a new AmazonServiceException with the specified message and
     * exception indicating the root cause.
     *
     * @param errorMessage
     *            An error message describing what went wrong.
     * @param cause
     *            The root exception that caused this exception to be thrown.
     */
    public AmazonServiceException(String errorMessage, Exception cause) {
        super(null, cause);
        this.errorMessage = errorMessage;
    }

    /**
     * Returns the AWS request ID that uniquely identifies the service request
     * the caller made.
     *
     * @return The AWS request ID that uniquely identifies the service request
     *         the caller made.
     */
    public String getRequestId() {
        return requestId;
    }

    /**
     * Returns the name of the service that sent this error response.
     *
     * @return The name of the service that sent this error response.
     */
    public String getServiceName() {
        return serviceName;
    }

    /**
     * Sets the AWS error code represented by this exception.
     *
     * @param errorCode
     *            The AWS error code represented by this exception.
     */
    public void setErrorCode(String errorCode) {
        this.errorCode = errorCode;
    }

    /**
     * Returns the AWS error code represented by this exception.
     *
     * @return The AWS error code represented by this exception.
     */
    public String getErrorCode() {
        return errorCode;
    }

    /**
     * @return the human-readable error message provided by the service
     */
    public String getErrorMessage() {
        return errorMessage;
    }

    /**
     * Sets the HTTP status code that was returned with this service exception.
     *
     * @param statusCode
     *            The HTTP status code that was returned with this service
     *            exception.
     */
    public void setStatusCode(int statusCode) {
        this.statusCode = statusCode;
    }

    /**
     * Returns the HTTP status code that was returned with this service
     * exception.
     *
     * @return The HTTP status code that was returned with this service
     *         exception.
     */
    public int getStatusCode() {
        return statusCode;
    }

    @Override
    public String getMessage() {
        return getErrorMessage()
            + " (Service: " + getServiceName()
            + "; Status Code: " + getStatusCode()
            + "; Error Code: " + getErrorCode()
            + "; Request ID: " + getRequestId() + ")";
    }

}
