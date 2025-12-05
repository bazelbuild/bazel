// Copyright 2021 The Bazel Authors. All rights reserved.
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

package build.stack.devtools.build.constellate.rendering;

import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleExtensionInfo;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.syntax.Location;

/** Stores information about a Starlark module extension definition. */
public class ModuleExtensionInfoWrapper {
  private final Object identifierObject;
  private final Location location;
  private final ModuleExtensionInfo.Builder moduleExtensionInfo;

  public ModuleExtensionInfoWrapper(
      Object identifierObject,
      Location location,
      ModuleExtensionInfo.Builder moduleExtensionInfo) {
    this.identifierObject = identifierObject;
    this.location = location;
    this.moduleExtensionInfo = moduleExtensionInfo;
  }

  public Object getIdentifierObject() {
    return identifierObject;
  }

  public Location getLocation() {
    return location;
  }

  public ModuleExtensionInfo.Builder getModuleExtensionInfo() {
    return moduleExtensionInfo;
  }
}
