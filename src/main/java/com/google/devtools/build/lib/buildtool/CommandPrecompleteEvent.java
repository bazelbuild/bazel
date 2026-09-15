// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

/**
 * This event is fired before at the end of every command before {@link
 * com.google.devtools.build.lib.runtime.BlazeModule#afterCommand()} is called.
 *
 * <p>Its purpose is to give a chance for event bus listeners to do things that need to happen at
 * the end of every command but before {@code afterCommand()}.
 */
public final class CommandPrecompleteEvent {}
