// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.android;

import com.google.common.collect.ImmutableList;

import java.nio.file.Path;


/**
 * And interface for apply modifiers to lists of resource directories.
 *
 * <p>
 * This is a common entry point for resource hacks such as the files deduplication and
 * the resource unpacking.
 */
interface DirectoryModifier {
  public abstract ImmutableList<Path> modify(ImmutableList<Path> directories);
}
