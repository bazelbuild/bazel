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
package com.google.devtools.build.lib.packages;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import java.io.Serializable;
import javax.annotation.Nullable;

/**
 * Generic implementation of {@link InfoInterface}.
 *
 * <p>Natively-defined Info objects should subclass this to be registered as Info objects that may
 * be passed between targets.
 */
public abstract class Info implements Serializable, InfoInterface, SkylarkValue {

  /** The {@link Provider} that describes the type of this instance. */
  protected final Provider provider;

  /**
   * The Skylark location where this provider instance was created.
   *
   * <p>Built-in provider instances may use {@link Location#BUILTIN}.
   */
  @VisibleForSerialization
  protected final Location location;

  protected Info(Provider provider, @Nullable Location location) {
    this.provider = Preconditions.checkNotNull(provider);
    this.location = location == null ? Location.BUILTIN : location;
  }

  /**
   * Returns the Skylark location where this provider instance was created.
   *
   * <p>Builtin provider instances may return {@link Location#BUILTIN}.
   */
  public Location getCreationLoc() {
    return location;
  }

  public Provider getProvider() {
    return provider;
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.append("<instance of provider ");
    printer.append(provider.getPrintableName());
    printer.append(">");
  }
}
