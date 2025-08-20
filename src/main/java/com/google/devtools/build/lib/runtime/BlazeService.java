// Copyright 2025 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.runtime;

/**
 * A marker interface for service interfaces that are used to communicate between Logic Component
 * (LC) and Services Component (SC).
 *
 * <p>A service interface defines a set of interactions between LC and a service implementation. The
 * service implementation then takes care of communicating with the outside world, e.g. remote
 * execution service.
 *
 * <p>SC is a collection of service implementations that are compiled into a separate jar file from
 * LC. They are registered into {@link BlazeServiceRegistry} during server startup. LC can only
 * communicate with SC through the service interface.
 */
public interface BlazeService {}
