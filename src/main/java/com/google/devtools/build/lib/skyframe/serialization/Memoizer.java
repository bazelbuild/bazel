// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.IdentityHashMap;
import javax.annotation.Nullable;

/**
 * A framework for serializing and deserializing with memo tables. Memoization is useful both for
 * performance and, in the case of cyclic data structures, to help avoid infinite recursion.
 *
 * <p><b>Usage:</b> To add support for a new type of value, create a class that implements {@link
 * MemoizingCodec}. To serialize, instantiate a new {@link Serializer} and call its {@link
 * Serializer#serialize serialize} method with the value and a codec for that value; and similarly
 * to deserialize, instantiate a {@link Deserializer} and call {@link Deserializer#deserialize
 * deserialize}. The memo table lives for the lifetime of the {@code Serializer}/{@code
 * Deserializer} instance. Do not call {@link MemoizingCodec#serializePayload} and {@link
 * MemoizingCodec#deserializePayload} directly, since that will bypass the memoization logic for the
 * top-level value to be encoded or decoded.
 *
 * <p>The memo table associates each value with an integer identifier.
 *
 * <ul>
 *   <li><i>On the sending end:</i> The first time a value is to be serialized, a new id is created
 *       and a mapping for it is added to the table. The id is emitted on the wire alongside the
 *       value's serialized representation. If the same value surfaces again later on, instead of
 *       reserializing it, we just emit a backreference consisting of its id.
 *   <li><i>On the receiving end:</i> Each deserialized value is stored in the memo table along with
 *       the id it was associated with. When a backreference is read, the value in the memo table is
 *       returned instead of deserializing a new copy of that same value.
 * </ul>
 *
 * <p>Cyclic data structures can occur either naturally in complex types, or as a result of a
 * pathological Skylark program. An example of the former is how a user-defined Skylark function
 * implicitly refers to its global frame, which in turn refers to the functions defined in that
 * frame. An example of the latter is a list that the user mutates to contain itself as an element.
 * Such pathological values in Skylark are technically allowed, but they are not useful since
 * Skylark prohibits recursive function calls. They can also expose implementation bugs in code that
 * is not expecting them (b/30310522).
 *
 * <p>Ideally, to handle cyclic data structures, the serializer should add the value to the memo
 * table <em>before</em> actually performing the serialization. For example, to handle the recursive
 * list {@code L = ["A", L]}, the serializer should do the following:
 *
 * <ol>
 *   <li>add a memo mapping from {@code L} to a fresh id, {@code k}
 *   <li>emit {@code k} to the wire
 *   <li>serialize {@code L}, which means it has to
 *       <ol>
 *         <li>write its length, {@code 2}
 *         <li>serialize the string {@code "A"}
 *         <li>serialize {@code L}, but since this matches the entry added in Step 1, it just emits
 *             {@code k} as a backreference
 *       </ol>
 * </ol>
 *
 * The problem is that on the other end of the wire, the deserializer needs to associate a value
 * with the memo entry for k before {@code L} has been fully formed. To solve this, we associate k
 * with a new empty list as the initial value, then allow the deserialization logic to mutate this
 * list to form {@code L}. It is important that this is done by mutating the initial list rather
 * than by replacing it with another list, since each backreference to k creates an actual Java
 * reference to the initial object.
 *
 * <p>However, this strategy does not work for all types of values. There is no way to deserialize
 * the recursive tuple {@code T = ("A", T)}, because tuples are immutable and therefore cannot be
 * instantiated before all of their elements have been. Rather than restrict serialization to only
 * mutable types, or add a special way for deserializers to modify seemingly immutable types, we
 * simply don't memoize immutable types until after they are fully constructed. This means that
 * {@code T} is not serializable. But that's okay, because objects like {@code T} should not even be
 * able to exist (barring a hidden API or reflection). In general, all cycles must go through at
 * least one mutable type of value.
 *
 * <p>Aside from mutability, there is another potential problem: One of the types' constructors may
 * enforce an invariant that is not satisfied at the time the constructor is invoked, even though it
 * may be satisfied once construction of all objects is complete. For instance, suppose a type
 * {@code Foo} has a constructor that takes a non-empty list. Then we would fail to deserialize
 * {@code L = [Foo(L)]}, since {@code L} is initially empty. Such a list could legally be formed by
 * putting other elements in {@code L} before creating {@code Foo}, and then later removing those
 * other elements. But there's no general way for a deserializer to know how to do that. Therefore,
 * it is the caller's responsibility to ensure the following property:
 *
 * <blockquote>
 *
 * For any value that is to be serialized, if the value has children that directly or indirectly
 * contain the value, then the value must be constructible even when those children are in a
 * semi-constructed state.
 *
 * </blockquote>
 *
 * where "semi-constructed state" means any state that can be produced by the codecs for those
 * children. Other serialization systems address this issue by providing multiple hooks for types to
 * setup their invariants, but we keep the API relatively simple.
 *
 * <p>Round-tripping a value through memoized serialization and deserialization is guaranteed to
 * preserve the object graph, i.e., to not duplicate a value. For mutable types, a value can only be
 * serialized and deserialized at most once because it is memoized before recursing over its
 * children. For immutable types, although a value can be serialized multiple times, upon
 * deserialization only one copy is retained in the memo table. This is conceptually similar to how
 * Python's Pickle library <a href="https://github.com/python/cpython/blob/3.6/Lib/pickle.py#L754">
 * handles tuples</a>, although in that case they use an abstract machine whereas we do not.
 *
 * <p>Wire format, as an abstract grammar:
 *
 * <pre>{@code
 * START             -->  NoMemoContent | `NEW_VALUE` MemoContent | `BACKREF` MemoId
 * MemoContent       -->  MemoBeforeContent | MemoAfterContent
 * NoMemoContent     -->  Payload
 * MemoBeforeContent -->  MemoId Payload
 * MemoAfterContent  -->  Payload MemoId
 * MemoId            -->  int32
 * }</pre>
 *
 * where {@code Payload} is the serialized representation of the value. {@code Payload} may itself
 * contain complete memo-aware encodings of the value's children.
 */
// TODO(brandjon): Maybe make this more robust against a pathological cycle of immutable objects, so
// that instead of failing with a stack overflow, we detect the cycle and throw
// SerializationException. This requires just a little extra memo tracking for the MEMOIZE_AFTER
// case.
public class Memoizer {

  /**
   * Constants used in the wire format to signal whether the next bytes are content or a
   * backreference.
   */
  private enum MemoEntry {
    NEW_VALUE,
    BACKREF
  }

  private static ObjectCodec<MemoEntry> memoEntryCodec = new EnumCodec<>(MemoEntry.class);

  private Memoizer() {}

  /** Indicates how a {@link MemoizingCodec} uses the memo table. */
  public enum Strategy {
    /**
     * Indicates that the memo table is not directly used by this codec.
     *
     * <p>Codecs with this strategy will always serialize payloads, never backreferences, even if
     * the same value has been serialized before. This does not apply to other codecs that are
     * delegated to via {@link Serializer#serialize}. Deserialization behaves
     * analogously.
     *
     * <p>This strategy is useful for codecs that write very little data themselves, but that still
     * delegate to other codecs.
     */
    DO_NOT_MEMOIZE,

    /**
     * Indicates that the value is memoized before recursing to its children, so that it is
     * available to form cyclic references from its children. If this strategy is used, {@link
     * MemoizingCodec#makeInitialValue} must be overridden.
     *
     * <p>This should be used for all types where it is feasible to provide an initial value. Any
     * cycle that does not go through at least one {@code MEMOIZE_BEFORE} type of value (e.g., a
     * pathological self-referential tuple) is unserializable.
     */
    MEMOIZE_BEFORE,

    /**
     * Indicates that the value is memoized after recursing to its children, so that it cannot be
     * referred to until after it has been constructed (regardless of whether its children are still
     * under construction).
     *
     * <p>This is typically used for immutable types, since they cannot be created by mutating an
     * initial value.
     */
    MEMOIZE_AFTER
  }

  /**
   * A codec that knows how to serialize/deserialize {@code T} using a memo strategy.
   *
   * <p>Codecs do not interact with memo tables directly; rather, they just define a memo strategy
   * and let {@link Serializer} and {@link Deserializer} take care of the memo logic and metadata. A
   * codec can dispatch to a subcodec to handle components of {@code T} compositionally; this
   * dispatching occurs via the {@code Serializer} or {@code Deserializer} so that the memo table is
   * consulted.
   *
   * <p>Codecs may access additional data from the deserialization context. For instance, Skylark
   * codecs may retrieve a {@link com.google.devtools.build.lib.syntax.Mutability} object to
   * associate with the value they construct. The type of this additional data is checked
   * dynamically, and a mistyping will result in a {@link SerializationException}.
   */
  // It is technically possible to work the type of the additional data into the generic type of
  // MemoizingCodec and Deserializer. But it makes the types ugly, and probably isn't worth the
  // damage to readability.
  public interface MemoizingCodec<T> extends BaseCodec<T> {

    /**
     * Returns the memoization strategy for this codec.
     *
     * <p>If set to {@link Strategy#MEMOIZE_BEFORE}, then {@link #makeInitialValue} must be
     * implemented.
     *
     * <p>Implementations of this method should just return a constant, since the choice of strategy
     * is usually intrinsic to {@code T}.
     */
    Strategy getStrategy();

    /**
     * If {@link #getStrategy} returns {@link Strategy#MEMOIZE_BEFORE}, then this method is called
     * to obtain a new initial / empty instance of {@code T} for deserialization. The returned
     * instance will be populated by {@link #deserializePayload}.
     *
     * <p>If any other strategy is used, this method is ignored. The default implementation throws
     * {@link UnsupportedOperationException}.
     *
     * <p>The passed in deserialization context can provide additional data that is used to help
     * initialize the value. For instance, it may contain a Skylark {@link
     * com.google.devtools.build.lib.syntax.Mutability}.
     */
    default T makeInitialValue(Deserializer deserializer) throws SerializationException {
      throw new UnsupportedOperationException(
          "No initial value was provided for codec class " + getClass().getName());
    }

    /**
     * Serializes an object's payload.
     *
     * <p>To delegate to a subcodec, an implementation should call {@code serializer}'s {@link
     * Serializer#serialize serialize} method with the appropriate piece of {@code obj} and the
     * subcodec. Do not call the subcodec's {@code serializePayload} method directly, as that would
     * bypass the memo table.
     */
    void serializePayload(
        SerializationContext context,
        T obj,
        CodedOutputStream codedOut,
        Serializer serializer)
        throws SerializationException, IOException;

    /**
     * Deserializes an object's payload.
     *
     * <p>To delegate to a subcodec, an implementation should call {@code deserializer}'s {@link
     * Deserializer#deserialize deserialize} method with the appropriate subcodec. Do not call the
     * subcodec's {@code deserializePayload} method directly, as that would bypass the memo table.
     *
     * If this codec uses the {@link Strategy#MEMOIZE_BEFORE} strategy (as determined by {@link
     * #getStrategy}), then the {@code initial} parameter will be a new value of {@code T} that can
     * be mutated into the result. In that case, this function must return {@code initial}. If any
     * other strategy is used, then {@code initial} will be null and this function must instantiate
     * the value to return.
     */
    T deserializePayload(
        DeserializationContext context,
        @Nullable T initial,
        CodedInputStream codedIn,
        Deserializer deserializer)
        throws SerializationException, IOException;
  }

  /** A context for serializing; wraps a memo table. */
  public static class Serializer {
    private final SerializingMemoTable memo = new SerializingMemoTable();

    public <T> void serialize(SerializationContext context, T obj, CodedOutputStream codedOut)
        throws IOException, SerializationException {
      MemoizingCodec<? super T> memoizingCodec =
          context.recordAndMaybeGetMemoizingCodec(obj, codedOut);
      if (memoizingCodec != null) {
        serialize(context, obj, memoizingCodec, codedOut);
      }
    }

    /**
     * Serializes an object using the given codec and current memo table state.
     *
     * @throws SerializationException on a logical error during serialization
     * @throws IOException on {@link IOException} during serialization
     */
    <T> void serialize(
        SerializationContext context,
        T obj,
        MemoizingCodec<? super T> codec,
        CodedOutputStream codedOut)
        throws SerializationException, IOException {
      Strategy strategy = codec.getStrategy();
      if (strategy == Strategy.DO_NOT_MEMOIZE) {
        codec.serializePayload(context, obj, codedOut, this);
      } else {
        Integer id = memo.lookupNullable(obj);
        if (id != null) {
          memoEntryCodec.serialize(context, MemoEntry.BACKREF, codedOut);
          codedOut.writeInt32NoTag(id);
        } else {
          memoEntryCodec.serialize(context, MemoEntry.NEW_VALUE, codedOut);
          serializeMemoContent(context, obj, codec, codedOut, strategy);
        }
      }
    }

    // Corresponds to MemoContent in the abstract grammar.
    private <T> void serializeMemoContent(
        SerializationContext context,
        T obj,
        MemoizingCodec<T> codec,
        CodedOutputStream codedOut,
        Strategy strategy)
        throws SerializationException, IOException {
      switch(strategy) {
        case MEMOIZE_BEFORE: {
          int id = memo.memoize(obj);
          codedOut.writeInt32NoTag(id);
          codec.serializePayload(context, obj, codedOut, this);
          break;
        }
        case MEMOIZE_AFTER: {
          codec.serializePayload(context, obj, codedOut, this);
          // If serializing the children caused the parent object itself to be serialized due to a
          // cycle, then there's now a memo entry for the parent. Don't overwrite it with a new id.
          Integer cylicallyCreatedId = memo.lookupNullable(obj);
          int id = (cylicallyCreatedId != null) ? cylicallyCreatedId : memo.memoize(obj);
          codedOut.writeInt32NoTag(id);
          break;
        }
        default:
          throw new AssertionError("Unreachable (strategy=" + strategy + ")");
      }
    }

    private static class SerializingMemoTable {
      private final IdentityHashMap<Object, Integer> table = new IdentityHashMap<>();

      /**
       * Adds a new value to the memo table and returns its id. The value must not already be
       * present.
       */
      private int memoize(Object value) {
        Preconditions.checkArgument(
            !table.containsKey(value),
            "Tried to memoize object '%s' multiple times", value);
        // Ids count sequentially from 0.
        int newId = table.size();
        table.put(value, newId);
        return newId;
      }

      /**
       * If the value is already memoized, return its on-the-wire id; otherwise return null. Opaque.
       *
       * <p>Beware accidental unboxing of a null result.
       */
      @Nullable
      private Integer lookupNullable(Object value) {
        return table.get(value);
      }
    }
  }

  /** A context for deserializing; wraps a memo table and possibly additional data. */
  public static class Deserializer {
    private final DeserializingMemoTable memo = new DeserializingMemoTable();

    /**
     * Additional data is dynamically typed, and retrieved using a type token.
     *
     * <p>If we need to support multiple kinds of additional data in the future, this could become
     * a mapping.
     */
    @Nullable
    private final Object additionalData;

    public Deserializer() {
      this.additionalData = null;
    }

    @VisibleForTesting
    public Deserializer(Object additionalData) {
      Preconditions.checkNotNull(additionalData);
      this.additionalData = additionalData;
    }

    /**
     * If this {@code Deserializer} was constructed with (non-null) additional data, and if its type
     * satisfies the given type token {@code klass}, returns that additional data.
     *
     * @throws NullPointerException if no additional data is present
     * @throws IllegalArgumentException if the additional data is not an instance of {@code klass}
     */
    <T> T getAdditionalData(Class<T> klass) {
      Preconditions.checkNotNull(additionalData,
          "Codec requires additional data of type %s, but no additional data is defined",
          klass.getName());
      try {
        return klass.cast(additionalData);
      } catch (ClassCastException e) {
        throw new IllegalArgumentException(String.format(
            "Codec requires additional data of type %s, but the available additional data has type "
            + "%s",
            klass.getName(),
            additionalData.getClass().getName()));
      }
    }

    /**
     * For use with generic types where a {@link Class} object is not available at runtime, or where
     * {@link Object} is the desired return type anyway.
     */
    public Object deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws IOException, SerializationException {
      return deserialize(context, codedIn, Object.class);
    }

    public <T> T deserialize(
        DeserializationContext context, CodedInputStream codedIn, Class<T> deserializedClass)
        throws SerializationException, IOException {
      @SuppressWarnings("unchecked")
      MemoizingCodec<? extends T> codec =
          (MemoizingCodec<? extends T>) context.getMemoizingCodecRecordedInInput(codedIn);
      return deserializedClass.cast(deserialize(context, codec, codedIn));
    }

    /**
     * Deserializes an object using the given codec and current memo table state.
     *
     * @throws SerializationException on a logical error during deserialization
     * @throws IOException on {@link IOException} during deserialization
     */
    <T> T deserialize(
        DeserializationContext context, MemoizingCodec<? extends T> codec, CodedInputStream codedIn)
        throws SerializationException, IOException {
      Strategy strategy = codec.getStrategy();
      if (strategy == Strategy.DO_NOT_MEMOIZE) {
        return codec.deserializePayload(context, null, codedIn, this);
      } else {
        MemoEntry memoEntry =
            memoEntryCodec.deserialize(context, codedIn);
        if (memoEntry == MemoEntry.BACKREF) {
          int id = codedIn.readInt32();
          return lookupBackreference(id, codec.getEncodedClass());
        } else if (memoEntry == MemoEntry.NEW_VALUE) {
          switch (strategy) {
            case MEMOIZE_BEFORE:
              return deserializeMemoBeforeContent(context, codec, codedIn);
            case MEMOIZE_AFTER:
              return deserializeMemoAfterContent(context, codec, codedIn);
            default:
              throw new AssertionError("Unreachable (strategy=" + strategy + ")");
          }
        } else {
          throw new AssertionError("Unreachable (memoEntry=" + memoEntry + ")");
        }
      }
    }

    /** Retrieves a memo entry, validating that it exists and has the expected type. */
    private <T> T lookupBackreference(int id, Class<T> expectedType) throws SerializationException {
      Object savedUnchecked = memo.lookup(id);
      if (savedUnchecked == null) {
        throw new SerializationException(
            "Found backreference to non-existent memo id (" + id + ")");
      }
      try {
        return expectedType.cast(savedUnchecked);
      } catch (ClassCastException e) {
        throw new SerializationException(
            "Backreference (to memo id " + id + ") has type "
                + savedUnchecked.getClass().getName()
                + " but expected type " + expectedType.getName());
      }
    }

    // Corresponds to MemoBeforeContent in the abstract grammar.
    private <T> T deserializeMemoBeforeContent(
        DeserializationContext context,
        MemoizingCodec<T> codec,
        CodedInputStream codedIn)
        throws SerializationException, IOException {
      int id = codedIn.readInt32();
      T initial = codec.makeInitialValue(this);
      memo.memoize(id, initial);
      T value = codec.deserializePayload(context, initial, codedIn, this);
      if (value != initial) {
        // This indicates a bug in the particular codec subclass.
        throw new SerializationException("doDeserialize did not return the initial instance");
      }
      return value;
    }

    // Corresponds to MemoAfterContent in the abstract grammar.
    private <T> T deserializeMemoAfterContent(
        DeserializationContext context,
        MemoizingCodec<T> codec,
        CodedInputStream codedIn)
        throws SerializationException, IOException {
      T value = codec.deserializePayload(context, null, codedIn, this);
      int id = codedIn.readInt32();
      // If deserializing the children caused the parent object itself to be deserialized due to
      // a cycle, then there's now a memo entry for the parent. Reuse that object, discarding
      // the one we were trying to construct here, so as to avoid creating duplicate objects in
      // the object graph.
      Object cyclicallyCreatedObject = memo.lookup(id);
      if (cyclicallyCreatedObject != null) {
        Class<T> expectedType = codec.getEncodedClass();
        try {
          return expectedType.cast(cyclicallyCreatedObject);
        } catch (ClassCastException e) {
          // This indicates either some kind of type reification mismatch, or a memo id
          // collision.
          throw new SerializationException(
              "While trying to deserialize a value of type " + expectedType.getName()
                  + " marked with memo id " + id + ", another value having that id was "
                  + "recursively created; but that value had unexpected type "
                  + cyclicallyCreatedObject.getClass().getName());
        }
      } else {
        memo.memoize(id, value);
        return value;
      }
    }

    private static class DeserializingMemoTable {

      private HashMap<Integer, Object> table = new HashMap<>();

      /**
       * Adds a new id -> object maplet to the memo table. The value must not already be present.
       */
      private void memoize(int id, Object value) {
        Preconditions.checkNotNull(value);
        Object prev = table.put(id, value);
        Preconditions.checkArgument(
            prev == null,
            "Tried to memoize id %s to object '%s', when it is already memoized to object '%s'",
            id, value, prev);
      }

      /** If the id has been memoized, returns its corresponding object. Otherwise returns null. */
      @Nullable
      private Object lookup(int id) {
        return table.get(id);
      }
    }
  }

  /**
   * An adaptor for treating an {@link ObjectCodec} as a {@link MemoizingCodec} that does not use
   * memoization.
   */
  static class ObjectCodecAdaptor<T> implements MemoizingCodec<T> {

    private final ObjectCodec<T> codec;

    ObjectCodecAdaptor(ObjectCodec<T> codec) {
      this.codec = codec;
    }

    @Override
    public Class<T> getEncodedClass() {
      return codec.getEncodedClass();
    }

    @Override
    public Strategy getStrategy() {
      return Strategy.DO_NOT_MEMOIZE;
    }

    @Override
    public void serializePayload(
        SerializationContext context,
        T obj,
        CodedOutputStream codedOut,
        Serializer serializer)
        throws SerializationException, IOException {
      codec.serialize(context, obj, codedOut);
    }

    @Override
    public T deserializePayload(DeserializationContext context,
        T initial,
        CodedInputStream codedIn,
        Deserializer deserializer)
        throws SerializationException, IOException {
      return codec.deserialize(context, codedIn);
    }
  }

  /** An adapter for {@link ObjectCodec} that allows the values to be memoized after creation. */
  static class MemoizingAfterObjectCodecAdapter<T> extends ObjectCodecAdaptor<T> {
    MemoizingAfterObjectCodecAdapter(ObjectCodec<T> codec) {
      super(codec);
    }

    @Override
    public Strategy getStrategy() {
      return Strategy.MEMOIZE_AFTER;
    }
  }

  /**
   * An adaptor for treating a {@link MemoizingCodec} as an {@link ObjectCodec}. By default, each
   * call to {@link ObjectCodec#serialize} or {@link ObjectCodec#deserialize} uses a fresh memo
   * table.
   *
   * <p>A subclass can override the {@link #getDeserializer} hook to add additional context
   * information, e.g. a Skylark {@link com.google.devtools.build.lib.syntax.Mutability} to share
   * among all deserialized Skylark values. Note that if a {@code Deserializer} is reused across
   * multiple calls to this hook, this adaptor may become stateful.
   */
  static class MemoizingCodecAdaptor<T> implements ObjectCodec<T> {

    private final MemoizingCodec<T> codec;

    MemoizingCodecAdaptor(MemoizingCodec<T> codec) {
      this.codec = codec;
    }

    /** Provides the context used in {@link #deserialize}. */
    Deserializer getDeserializer() {
      return new Deserializer();
    }

    @Override
    public Class<T> getEncodedClass() {
      return codec.getEncodedClass();
    }

    @Override
    public void serialize(SerializationContext context, T obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      new Serializer().serialize(context, obj, codec, codedOut);
    }

    @Override
    public T deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      return getDeserializer().deserialize(context, codec, codedIn);
    }
  }
}
