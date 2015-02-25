// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.packages.AspectFactory;

/**
 * Instantiation of {@link AspectFactory} with the actual types.
 *
 * <p>This is needed because {@link AspectFactory} is needed in the {@code packages} package to
 * do loading phase things properly and to be able to specify them on attributes, but the actual
 * classes are in the {@code view} package, which is not available there.
 */
public interface ConfiguredAspectFactory
    extends AspectFactory<ConfiguredTarget, RuleContext, Aspect> {

}
