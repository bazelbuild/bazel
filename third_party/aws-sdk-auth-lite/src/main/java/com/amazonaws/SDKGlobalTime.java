/*
 * Copyright 2015-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
 * Used for clock skew adjustment between the client JVM where the SDK is run,
 * and the server side.
 */
public class SDKGlobalTime {
    /**
     * globalTimeOffset is a time difference in seconds between the running JVM
     * and AWS. Used to globally adjust the client clock skew. Java SDK already
     * provides timeOffset and accessor methods in <code>Request</code> class but
     * those are used per request, whereas this variable will adjust clock skew
     * globally. Java SDK detects clock skew errors and adjusts global clock
     * skew automatically.
     */
    private static volatile int globalTimeOffset;

    /**
     * Sets the global time difference in seconds between the running JVM and
     * AWS. If this value is set then all the subsequent instantiation of an
     * <code>AmazonHttpClient</code> will start using this
     * value to generate timestamps.
     *
     * @param timeOffset
     *            the time difference in seconds between the running JVM and AWS
     */
    public  static void setGlobalTimeOffset(int timeOffset) {
        globalTimeOffset = timeOffset;
    }

    /**
     * Gets the global time difference in seconds between the running JVM and
     * AWS. See <code>Request#getTimeOffset()</code> if global time offset is
     * not set.
     */
    public static int getGlobalTimeOffset() {
        return globalTimeOffset;
    }
}
