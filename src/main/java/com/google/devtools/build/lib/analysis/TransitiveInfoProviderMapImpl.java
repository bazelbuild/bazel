// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.collect.ImmutableSharedKeyMap;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Implementation of {@link TransitiveInfoProvider} that uses {@link ImmutableSharedKeyMap}. For
 * memory efficiency, inheritance is used instead of aggregation as an implementation detail.
 */
public class TransitiveInfoProviderMapImpl extends ImmutableSharedKeyMap<Object, Object>
    implements TransitiveInfoProviderMap {

  @SerializationConstant @VisibleForSerialization
  static final TransitiveInfoProviderMapImpl EMPTY_TRANSITIVE_INFO_PROVIDER_MAP =
      new TransitiveInfoProviderMapImpl(new Object[0], new Object[0]);

  private TransitiveInfoProviderMapImpl(Object[] keys, Object[] values) {
    super(keys, values);
  }

  static TransitiveInfoProviderMapImpl create(Map<Object, Object> map) {
    int count = map.size();
    if (count == 0) {
      return EMPTY_TRANSITIVE_INFO_PROVIDER_MAP;
    }
    Object[] keys = new Object[count];
    Object[] values = new Object[count];
    int i = 0;
    for (Map.Entry<Object, Object> entry : map.entrySet()) {
      keys[i] = entry.getKey();
      values[i] = entry.getValue();
      ++i;
    }
    Preconditions.checkArgument(keys.length == values.length);
    return new TransitiveInfoProviderMapImpl(keys, values);
  }

  @SuppressWarnings("unchecked")
  @Nullable
  @Override
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> providerClass) {
    Class<? extends TransitiveInfoProvider> effectiveClass =
        TransitiveInfoProviderEffectiveClassHelper.get(providerClass);
    return (P) get(effectiveClass);
  }

  @Nullable
  @Override
  public Info get(Provider.Key key) {
    return (Info) super.get(key);
  }

  @Nullable
  @Override
  public Object get(String legacyKey) {
    return super.get(legacyKey);
  }

  @Override
  public int getProviderCount() {
    return size();
  }

  @Override
  public Object getProviderKeyAt(int i) {
    return keyAt(i);
  }

  @Override
  public Object getProviderInstanceAt(int i) {
    return valueAt(i);
  }

  public static ValueSharingCodec valueSharingCodec() {
    return ValueSharingCodec.INSTANCE;
  }

  // TODO: b/359437873 - generate with @AutoCodec.
  private static class ValueSharingCodec
      extends DeferredObjectCodec<TransitiveInfoProviderMapImpl> {
    private static final ValueSharingCodec INSTANCE = new ValueSharingCodec();

    @Override
    public boolean autoRegister() {
      return false;
    }

    @Override
    public Class<TransitiveInfoProviderMapImpl> getEncodedClass() {
      return TransitiveInfoProviderMapImpl.class;
    }

    @Override
    public void serialize(
        SerializationContext context, TransitiveInfoProviderMapImpl obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.putSharedValue(
          obj.getKeys(), /* distinguisher= */ null, DeferredKeysCodec.INSTANCE, codedOut);
      context.serialize(obj.values, codedOut);
    }

    @Override
    public DeferredValue<TransitiveInfoProviderMapImpl> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      var builder = new DeserializationBuilder();
      context.getSharedValue(
          codedIn,
          /* distinguisher= */ null,
          DeferredKeysCodec.INSTANCE,
          builder,
          DeserializationBuilder::setKeys);
      context.deserialize(codedIn, builder, DeserializationBuilder::setValues);
      return builder;
    }

    private static class DeferredKeysCodec extends DeferredObjectCodec<Object[]> {
      private static final DeferredKeysCodec INSTANCE = new DeferredKeysCodec();

      @Override
      public Class<Object[]> getEncodedClass() {
        return Object[].class;
      }

      @Override
      public boolean autoRegister() {
        return false;
      }

      @Override
      public void serialize(SerializationContext context, Object[] obj, CodedOutputStream codedOut)
          throws SerializationException, IOException {
        int length = obj.length;
        codedOut.writeInt32NoTag(length);
        for (int i = 0; i < length; i++) {
          context.serialize(obj[i], codedOut);
        }
      }

      @Override
      public DeferredValue<Object[]> deserializeDeferred(
          AsyncDeserializationContext context, CodedInputStream codedIn)
          throws SerializationException, IOException {
        int length = codedIn.readInt32();
        Object[] values = new Object[length];
        for (int i = 0; i < length; i++) {
          context.deserialize(codedIn, values, new ArrayFieldSetter(i));
        }
        return () -> values;
      }

      private static final class ArrayFieldSetter
          implements AsyncDeserializationContext.FieldSetter<Object[]> {
        private final int index;

        private ArrayFieldSetter(int index) {
          this.index = index;
        }

        @Override
        public void set(Object[] array, Object value) {
          array[index] = value;
        }
      }
    }
  }

  @Keep // used reflectively
  private static final class Codec extends DeferredObjectCodec<TransitiveInfoProviderMapImpl> {
    @Override
    public Class<TransitiveInfoProviderMapImpl> getEncodedClass() {
      return TransitiveInfoProviderMapImpl.class;
    }

    @Override
    public void serialize(
        SerializationContext context, TransitiveInfoProviderMapImpl obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serialize(obj.getKeys(), codedOut);
      context.serialize(obj.values, codedOut);
    }

    @Override
    public DeferredValue<TransitiveInfoProviderMapImpl> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      var builder = new DeserializationBuilder();
      context.deserialize(codedIn, builder, DeserializationBuilder::setKeys);
      context.deserialize(codedIn, builder, DeserializationBuilder::setValues);
      return builder;
    }
  }

  private static final class DeserializationBuilder
      implements DeferredObjectCodec.DeferredValue<TransitiveInfoProviderMapImpl> {
    private Object[] keys;
    private Object[] values;

    @Override
    public TransitiveInfoProviderMapImpl call() {
      return new TransitiveInfoProviderMapImpl(keys, values);
    }

    private static void setKeys(DeserializationBuilder builder, Object value) {
      builder.keys = (Object[]) value;
    }

    private static void setValues(DeserializationBuilder builder, Object value) {
      builder.values = (Object[]) value;
    }
  }
}
