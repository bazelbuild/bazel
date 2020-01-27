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
package com.amazonaws.log;

/**
 * Used to delegate internal logging of the signers and core classes to the
 * currently configured default logging framework of the SDK.
 * 
 * @see InternalLogFactory
 */
class InternalLog implements InternalLogApi {
    private final String name;

    InternalLog(String name) {
        this.name = name;
    }

    private InternalLogApi logger() {
        return InternalLogFactory.getFactory().doGetLog(name);
    }

    @Override
    public void debug(Object message) {
        logger().debug(message);
    }

    @Override
    public void debug(Object message, Throwable t) {
        logger().debug(message, t);
    }

    @Override
    public void error(Object message) {
        logger().error(message);
    }

    @Override
    public void error(Object message, Throwable t) {
        logger().error(message, t);
    }

    @Override
    public void fatal(Object message) {
        logger().fatal(message);
    }

    @Override
    public void fatal(Object message, Throwable t) {
        logger().fatal(message, t);
    }

    @Override
    public void info(Object message) {
        logger().info(message);
    }

    @Override
    public void info(Object message, Throwable t) {
        logger().info(message, t);
    }

    @Override
    public boolean isDebugEnabled() {
        return logger().isDebugEnabled();
    }

    @Override
    public boolean isErrorEnabled() {
        return logger().isErrorEnabled();
    }

    @Override
    public boolean isFatalEnabled() {
        return logger().isFatalEnabled();
    }

    @Override
    public boolean isInfoEnabled() {
        return logger().isInfoEnabled();
    }

    @Override
    public boolean isTraceEnabled() {
        return logger().isTraceEnabled();
    }

    @Override
    public boolean isWarnEnabled() {
        return logger().isWarnEnabled();
    }

    @Override
    public void trace(Object message) {
        logger().trace(message);
    }

    @Override
    public void trace(Object message, Throwable t) {
        logger().trace(message, t);
    }

    @Override
    public void warn(Object message) {
        logger().warn(message);
    }

    @Override
    public void warn(Object message, Throwable t) {
        logger().warn(message, t);
    }
}
