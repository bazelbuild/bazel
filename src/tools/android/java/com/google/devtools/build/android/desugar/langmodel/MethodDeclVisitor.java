/*
 * Copyright 2019 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.langmodel;

/** A visitor that directs different operations based on the method types. */
public interface MethodDeclVisitor<R, K extends MethodDeclInfo, P> {

  R visitClassConstructor(K methodDeclInfo, P param);

  R visitClassStaticMethod(K methodDeclInfo, P param);

  R visitClassInstanceMethod(K methodDeclInfo, P param);

  R visitInterfaceStaticMethod(K methodDeclInfo, P param);

  R visitInterfaceInstanceMethod(K methodDeclInfo, P param);
}
