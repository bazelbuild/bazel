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
 */package com.amazonaws.log;

import java.util.logging.Logger;

/**
 * Internal logging factory for the signers and core classes based on JUL.
 */
public final class JulLogFactory extends InternalLogFactory {
    @Override
    protected InternalLogApi doGetLog(Class<?> clazz) {
        return new JulLog(Logger.getLogger(clazz.getName()));
    }

    @Override
    protected InternalLogApi doGetLog(String name) {
        return new JulLog(Logger.getLogger(name));
    }
}
