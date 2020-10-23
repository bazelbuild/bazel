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

/** A visitor for all method invocation instructions in combination with method types. */
public interface MethodInstrVisitor<R, K extends MethodKey, P> {

  R visitInvokeVirtual(K methodKey, P param);

  R visitInvokeSpecial(K methodKey, P param);

  R visitConstructorInvokeSpecial(K methodKey, P param);

  R visitInterfaceInvokeSpecial(K methodKey, P param);

  R visitInvokeStatic(K methodKey, P param);

  R visitInterfaceInvokeStatic(K methodKey, P param);

  R visitInvokeInterface(K methodKey, P param);

  R visitInvokeDynamic(K methodKey, P param);
}
