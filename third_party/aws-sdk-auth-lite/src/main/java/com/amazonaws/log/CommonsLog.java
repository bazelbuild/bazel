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

import org.apache.commons.logging.Log;

/**
 * Used to delegate internal logging of the signers and core classes to Jakarta
 * Commons Logging.
 */
public class CommonsLog implements com.amazonaws.log.InternalLogApi {

    private final org.apache.commons.logging.Log log;

    CommonsLog(Log log) {
        this.log = log;
    }

    @Override
    public void debug(Object message) {
        log.debug(message);
    }

    @Override
    public void debug(Object message, Throwable t) {
        log.debug(message, t);
    }

    @Override
    public void error(Object message) {
        log.error(message);
    }

    @Override
    public void error(Object message, Throwable t) {
        log.error(message, t);
    }

    @Override
    public void fatal(Object message) {
        log.fatal(message);
    }

    @Override
    public void fatal(Object message, Throwable t) {
        log.fatal(message, t);
    }

    @Override
    public void info(Object message) {
        log.info(message);
    }

    @Override
    public void info(Object message, Throwable t) {
        log.info(message, t);
    }

    @Override
    public boolean isDebugEnabled() {
        return log.isDebugEnabled();
    }

    @Override
    public boolean isErrorEnabled() {
        return log.isErrorEnabled();
    }

    @Override
    public boolean isFatalEnabled() {
        return log.isFatalEnabled();
    }

    @Override
    public boolean isInfoEnabled() {
        return log.isInfoEnabled();
    }

    @Override
    public boolean isTraceEnabled() {
        return log.isTraceEnabled();
    }

    @Override
    public boolean isWarnEnabled() {
        return log.isWarnEnabled();
    }

    @Override
    public void trace(Object message) {
        log.trace(message);
    }

    @Override
    public void trace(Object message, Throwable t) {
        log.trace(message, t);
    }

    @Override
    public void warn(Object message) {
        log.warn(message);
    }

    @Override
    public void warn(Object message, Throwable t) {
        log.warn(message, t);
    }
}
