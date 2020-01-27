// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.incrementaldeployment;

/**
 * A dummy class.
 *
 * <p>This class exists because Android L requires that a .dex file be present in a main .apk.
 * We do not want to put any class in there so that we can replace them without reinstalling the
 * complete app, thus, we put this tiny little class in there.
 */
public final class Placeholder {
}
