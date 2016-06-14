// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.objc.ObjcProvider.Key;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkSignatureProcessor;

import java.util.Map.Entry;

/**
 * A class that exposes apple rule implementation internals to skylark.
 */
@SkylarkModule(
  name = "apple_common",
  doc = "Functions for skylark to access internals of the apple rule implementations."
)
public class AppleSkylarkCommon {
 
  @VisibleForTesting
  public static final String BAD_KEY_ERROR = "Argument %s not a recognized key or 'providers'.";

  @VisibleForTesting
  public static final String BAD_SET_TYPE_ERROR =
      "Value for key %s must be a set of %s, instead found set of %s.";

  @VisibleForTesting
  public static final String BAD_PROVIDERS_ITER_ERROR =
      "Value for argument 'providers' must be a list of ObjcProvider instances, instead found %s.";

  @VisibleForTesting
  public static final String BAD_PROVIDERS_ELEM_ERROR =
      "Value for argument 'providers' must be a list of ObjcProvider instances, instead found "
          + "iterable with %s.";

  @VisibleForTesting
  public static final String NOT_SET_ERROR = "Value for key %s must be a set, instead found %s.";
  
  @SkylarkCallable(
      name = "apple_toolchain",
      doc = "Utilities for resolving items from the apple toolchain."
  )
  public AppleToolchain getAppleToolchain() {
    return new AppleToolchain();
  }

  @SkylarkSignature(
    name = "new_objc_provider",
    objectType = AppleSkylarkCommon.class,
    returnType = ObjcProvider.class,
    doc = "Creates a new ObjcProvider instance.",
    parameters = {
      @Param(name = "self", type = AppleSkylarkCommon.class, doc = "The apple_common instance."),
      @Param(
        name = "uses_swift",
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = "Whether this provider should enable Swift support."
      )
    },
    extraKeywords =
      @Param(
        name = "kwargs",
        type = SkylarkDict.class,
        defaultValue = "{}",
        doc = "Dictionary of arguments"
      )
  )
  public static final BuiltinFunction NEW_OBJC_PROVIDER =
      new BuiltinFunction("new_objc_provider") {
        @SuppressWarnings("unused")
        // This method is registered statically for skylark, and never called directly.
        public ObjcProvider invoke(
            AppleSkylarkCommon self, Boolean usesSwift, SkylarkDict<String, Object> kwargs) {
          ObjcProvider.Builder resultBuilder = new ObjcProvider.Builder();
          if (usesSwift) {
            resultBuilder.add(ObjcProvider.FLAG, ObjcProvider.Flag.USES_SWIFT);
          }
          for (Entry<String, Object> entry : kwargs.entrySet()) {
            Key<?> key = ObjcProvider.getSkylarkKeyForString(entry.getKey());
            if (key != null) {
              resultBuilder.addElementsFromSkylark(key, entry.getValue());
            } else if (entry.getKey().equals("providers")) {
              resultBuilder.addProvidersFromSkylark(entry.getValue());
            } else {
              throw new IllegalArgumentException(String.format(BAD_KEY_ERROR, entry.getKey()));
            }
          }
          return resultBuilder.build();
        }
      };

  static {
    SkylarkSignatureProcessor.configureSkylarkFunctions(AppleSkylarkCommon.class);
  }
}

