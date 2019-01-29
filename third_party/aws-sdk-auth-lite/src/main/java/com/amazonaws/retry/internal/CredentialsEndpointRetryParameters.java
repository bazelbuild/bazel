/*
 * Copyright 2011-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *    http://aws.amazon.com/apache2.0
 *
 * This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and
 * limitations under the License.
 */
package com.amazonaws.retry.internal;

import com.amazonaws.annotation.SdkInternalApi;

/**
 * Parameters that are used in {@link CredentialsEndpointRetryPolicy}.
 */
@SdkInternalApi
public class CredentialsEndpointRetryParameters {

    private final Integer statusCode;

    private final Exception exception;

    private CredentialsEndpointRetryParameters(Builder builder) {
        this.statusCode = builder.statusCode;
        this.exception = builder.exception;
    }

    public Integer getStatusCode() {
        return statusCode;
    }

    public Exception getException() {
        return exception;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {

         private final Integer statusCode;

         private final Exception exception;

         private Builder() {
             this.statusCode = null;
             this.exception = null;
         }

         private Builder(Integer statusCode, Exception exception) {
             this.statusCode = statusCode;
             this.exception = exception;
         }

         /**
          * @param statusCode The status code from Http response.
          *
          * @return This object for method chaining.
          */
         public Builder withStatusCode(Integer statusCode) {
             return new Builder(statusCode, this.exception);
         }

         /**
          *
          * @param exception The exception that was thrown.
          * @return This object for method chaining.
          */
         public Builder withException(Exception exception) {
             return new Builder(this.statusCode, exception);
         }

         public CredentialsEndpointRetryParameters build() {
             return new CredentialsEndpointRetryParameters(this);
         }

    }
}
