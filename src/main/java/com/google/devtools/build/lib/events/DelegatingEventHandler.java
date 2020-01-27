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

package com.google.devtools.build.lib.events;

import com.google.common.base.Preconditions;

/**
 * An EventHandler which delegates to another EventHandler. Primarily useful as a base class for
 * extending behavior.
 */
public class DelegatingEventHandler implements ExtendedEventHandler {
  protected final ExtendedEventHandler delegate;

  public DelegatingEventHandler(ExtendedEventHandler delegate) {
    super();
    this.delegate = Preconditions.checkNotNull(delegate);
  }

  @Override
  public void handle(Event e) {
    delegate.handle(e);
  }

  @Override
  public void post(ExtendedEventHandler.Postable obj) {
    delegate.post(obj);
  }
}
