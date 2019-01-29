/*
 * Copyright 2012-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with
 * the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package com.amazonaws.annotation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Target;

/**
 * Marker for elements that should only be accessed by the generated clients and not users of the
 * SDK. Do not make breaking changes to these APIs - they won't directly break customers, but
 * they'll break old versions of generated clients.
 * <p>
 * TODO: Write a linter that makes sure generated code only depends on public or
 * {@code @InternalApi} classes.
 */
@Target({ ElementType.PACKAGE, ElementType.TYPE, ElementType.FIELD, ElementType.CONSTRUCTOR, ElementType.METHOD })
public @interface SdkProtectedApi {
}
