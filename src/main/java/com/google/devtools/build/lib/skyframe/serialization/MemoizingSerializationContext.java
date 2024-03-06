// Copyright 2024 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedOutputStream;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import it.unimi.dsi.fastutil.objects.Reference2IntOpenHashMap;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.HashSet;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * {@link SerializationContext} with memoization tables.
 *
 * <p>Memoization is useful both for performance and, in the case of cyclic data structures, to help
 * avoid infinite recursion.
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
 * pathological Starlark program. An example of the former is how a user-defined Starlark function
 * implicitly refers to its global frame, which in turn refers to the functions defined in that
 * frame. An example of the latter is a list that the user mutates to contain itself as an element.
 * Such pathological values in Starlark are technically allowed, but they are not useful since
 * Starlark prohibits recursive function calls. They can also expose implementation bugs in code
 * that is not expecting them (b/30310522).
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
abstract class MemoizingSerializationContext extends SerializationContext {
  private final Reference2IntOpenHashMap<Object> table = new Reference2IntOpenHashMap<>();

  /** Table for types memoized using values equality, currently only {@link String}. */
  private final Object2IntOpenHashMap<Object> valuesTable = new Object2IntOpenHashMap<>();

  private final Set<Class<?>> explicitlyAllowedClasses = new HashSet<>();

  @VisibleForTesting // private
  static MemoizingSerializationContext createForTesting(
      ObjectCodecRegistry codecRegistry, ImmutableClassToInstanceMap<Object> dependencies) {
    return new MemoizingSerializationContextImpl(codecRegistry, dependencies);
  }

  MemoizingSerializationContext(
      ObjectCodecRegistry codecRegistry, ImmutableClassToInstanceMap<Object> dependencies) {
    super(codecRegistry, dependencies);
    table.defaultReturnValue(-1);
    valuesTable.defaultReturnValue(-1);
  }

  static byte[] serializeToBytes(
      ObjectCodecRegistry codecRegistry,
      ImmutableClassToInstanceMap<Object> dependencies,
      @Nullable Object subject,
      int outputCapacity,
      int bufferCapacity)
      throws SerializationException {
    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream(outputCapacity);
    serializeToStream(codecRegistry, dependencies, subject, bytesOut, bufferCapacity);
    return bytesOut.toByteArray();
  }

  static ByteString serializeToByteString(
      ObjectCodecRegistry codecRegistry,
      ImmutableClassToInstanceMap<Object> dependencies,
      @Nullable Object subject,
      int outputCapacity,
      int bufferCapacity)
      throws SerializationException {
    ByteString.Output bytesOut = ByteString.newOutput(outputCapacity);
    serializeToStream(codecRegistry, dependencies, subject, bytesOut, bufferCapacity);
    return bytesOut.toByteString();
  }

  @Override
  public final void addExplicitlyAllowedClass(Class<?> allowedClass) {
    explicitlyAllowedClasses.add(allowedClass);
  }

  @Override
  public final <T> void checkClassExplicitlyAllowed(Class<T> allowedClass, T objectForDebugging)
      throws SerializationException {
    if (!explicitlyAllowedClasses.contains(allowedClass)) {
      throw new SerializationException(
          allowedClass
              + " not explicitly allowed (allowed classes were: "
              + explicitlyAllowedClasses
              + ") and object is "
              + objectForDebugging);
    }
  }

  @Override
  final void serializeWithCodec(ObjectCodec<Object> codec, Object obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    switch (codec.getStrategy()) {
      case MEMOIZE_BEFORE:
        {
          int id = memoize(obj);
          codedOut.writeInt32NoTag(id);
          codec.serialize(this, obj, codedOut);
          break;
        }
      case MEMOIZE_AFTER:
        {
          codec.serialize(this, obj, codedOut);
          // If serializing the children caused the parent object itself to be serialized due to a
          // cycle, then there's now a memo entry for the parent. Don't overwrite it with a new
          // id.
          int cylicallyCreatedId = getMemoizedIndex(obj);
          int id = (cylicallyCreatedId != -1) ? cylicallyCreatedId : memoize(obj);
          codedOut.writeInt32NoTag(id);
          break;
        }
    }
  }

  @Override
  final boolean writeBackReferenceIfMemoized(Object obj, CodedOutputStream codedOut)
      throws IOException {
    int memoizedIndex = getMemoizedIndex(obj);
    if (memoizedIndex == -1) {
      return false;
    }
    // Subtracts 1 so it will be negative and not collide with null.
    codedOut.writeSInt32NoTag(-memoizedIndex - 1);
    return true;
  }

  @Override
  public final boolean isMemoizing() {
    return true;
  }

  /** If the value is already memoized, return its on-the-wire id; otherwise returns {@code -1}. */
  private int getMemoizedIndex(Object value) {
    if (value instanceof String) {
      return valuesTable.getInt(value);
    }
    return table.getInt(value);
  }

  /**
   * Adds a new value to the memo table and returns its id.
   *
   * <p>{@code value} must not already be present.
   */
  private int memoize(Object value) {
    Preconditions.checkArgument(
        getMemoizedIndex(value) == -1, "Tried to memoize object '%s' multiple times", value);
    // Ids count sequentially from 0.
    int newId = table.size() + valuesTable.size();
    if (value instanceof String) {
      valuesTable.put(value, newId);
    } else {
      table.put(value, newId);
    }
    return newId;
  }

  private static void serializeToStream(
      ObjectCodecRegistry codecRegistry,
      ImmutableClassToInstanceMap<Object> dependencies,
      @Nullable Object subject,
      OutputStream output,
      int bufferCapacity)
      throws SerializationException {
    CodedOutputStream codedOut = CodedOutputStream.newInstance(output, bufferCapacity);
    try {
      new MemoizingSerializationContextImpl(codecRegistry, dependencies)
          .serialize(subject, codedOut);
      codedOut.flush();
    } catch (IOException e) {
      throw new SerializationException("Failed to serialize " + subject, e);
    }
  }

  /**
   * This mainly exists to restrict use of {@link MemoizingSerializationContext}'s constructor.
   *
   * <p>It's also slightly cleaner for {@link SharedValueSerializationContext} to not inherit the
   * implementation of {@link #getFreshContext}.
   */
  private static final class MemoizingSerializationContextImpl
      extends MemoizingSerializationContext {
    private MemoizingSerializationContextImpl(
        ObjectCodecRegistry codecRegistry, ImmutableClassToInstanceMap<Object> dependencies) {
      super(codecRegistry, dependencies);
    }

    @Override
    public MemoizingSerializationContext getFreshContext() {
      return new MemoizingSerializationContextImpl(getCodecRegistry(), getDependencies());
    }
  }
}
