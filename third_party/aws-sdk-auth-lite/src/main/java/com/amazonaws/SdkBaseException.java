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
package com.amazonaws;

/**
 *
 * Base class for all exceptions thrown by the SDK.
 * Exception may be a client side exception or an unmarshalled service exception.
 */
public class SdkBaseException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    /**
     * Creates a new SdkBaseException with the specified message, and root
     * cause.
     *
     * @param message
     *            An error message describing why this exception was thrown.
     * @param t
     *            The underlying cause of this exception.
     */
    public SdkBaseException(String message, Throwable t) {
        super(message, t);
    }

    /**
     * Creates a new SdkBaseException with the specified message.
     *
     * @param message
     *            An error message describing why this exception was thrown.
     */
    public SdkBaseException(String message) {
        super(message);
    }

    /**
     * Creates a new SdkBaseException with the root cause.
     *
     * @param t
     *          The underlying cause of this exception.
     */
    public SdkBaseException(Throwable t) {
        super(t);
    }
}
