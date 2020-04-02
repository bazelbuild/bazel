// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization.autocodec;

import java.lang.annotation.ElementType;
import java.lang.annotation.Target;

/**
 * Applied to a field (which must be static and final). The field is stored as a "constant" allowing
 * for trivial serialization of it as an integer tag (see {@code CodecScanner} and {@code
 * ObjectCodecRegistry}). In order to do that, a trivial associated "RegisteredSingleton" class is
 * generated. Tagging such a field is harmless, and can be done conservatively.
 */
@Target(ElementType.FIELD)
public @interface SerializationConstant {}
