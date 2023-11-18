Project: /_project.yaml
Book: /_book.yaml

# A Guide to Skyframe `StateMachine`s

{% include "_buttons.html" %}

## Overview

A Skyframe `StateMachine` is a *deconstructed* function-object that resides on
the heap. It supports flexible and evaluation without redundancy[^1] when
required values are not immediately available but computed asynchronously. The
`StateMachine` cannot tie up a thread resource while waiting, but instead has to
be suspended and resumed. The deconstruction thus exposes explicit re-entry
points so that prior computations can be skipped.

`StateMachine`s can be used to express sequences, branching, structured logical
concurrency and are tailored specifically for Skyframe interaction.
`StateMachine`s can be composed into larger `StateMachine`s and share
sub-`StateMachine`s. Concurrency is always hierarchical by construction and
purely logical. Every concurrent subtask runs in the single shared parent
SkyFunction thread.

## Introduction

This section briefly motivates and introduces `StateMachine`s, found in the
[`java.com.google.devtools.build.skyframe.state`](https://github.com/bazelbuild/bazel/tree/master/src/main/java/com/google/devtools/build/skyframe/state)
package.

### A brief introduction to Skyframe restarts

Skyframe is a framework that performs parallel evaluation of dependency graphs.
Each node in the graph corresponds with the evaluation of a SkyFunction with a
SkyKey specifying its parameters and SkyValue specifying its result. The
computational model is such that a SkyFunction may lookup SkyValues by SkyKey,
triggering recursive, parallel evaluation of additional SkyFunctions. Instead of
blocking, which would tie up a thread, when a requested SkyValue is not yet
ready because some subgraph of computation is incomplete, the requesting
SkyFunction observes a `null` `getValue` response and should return `null`
instead of a SkyValue, signaling that it is incomplete due to missing inputs.
Skyframe *restarts* the SkyFunctions when all previously requested SkyValues
become available.

Before the introduction of `SkyKeyComputeState`, the traditional way of handling
a restart was to fully rerun the computation. Although this has quadratic
complexity, functions written this way eventually complete because each rerun,
fewer lookups return `null`. With `SkyKeyComputeState` it is possible to
associate hand-specified check-point data with a SkyFunction, saving significant
recomputation.

`StateMachine`s are objects that live inside `SkyKeyComputeState` and eliminate
virtually all recomputation when a SkyFunction restarts (assuming that
`SkyKeyComputeState` does not fall out of cache) by exposing suspend and resume
execution hooks.

### Stateful computations inside `SkyKeyComputeState` {:#stateful-computations}

From an object-oriented design standpoint, it makes sense to consider storing
computational objects inside `SkyKeyComputeState` instead of pure data values.
In *Java*, the bare minimum description of a behavior carrying object is a
*functional interface* and it turns out to be sufficient. A `StateMachine` has
the following, curiously recursive, definition[^2].

```
@FunctionalInterface
public interface StateMachine {
  StateMachine step(Tasks tasks) throws InterruptedException;
}
```

The `Tasks` interface is analogous to `SkyFunction.Environment` but it is
designed for asynchrony and adds support for logically concurrent subtasks[^3].

The return value of `step` is another `StateMachine`, allowing the specification
of a sequence of steps, inductively. `step` returns `DONE` when the
`StateMachine` is done. For example:

```
class HelloWorld implements StateMachine {
  @Override
  public StateMachine step(Tasks tasks) {
    System.out.println("hello");
    return this::step2;  // The next step is HelloWorld.step2.
  }

  private StateMachine step2(Tasks tasks) {
     System.out.println("world");
     // DONE is special value defined in the `StateMachine` interface signaling
     // that the computation is done.
     return DONE;
  }
}
```

describes a `StateMachine` with the following output.

```
hello
world
```

Note that the method reference `this::step2` is also a `StateMachine` due to
`step2` satisfying `StateMachine`'s functional interface definition. Method
references are the most common way to specify the next state in a
`StateMachine`.

![Suspending and resuming](/contribute/images/suspend-resume.svg)

Intuitively, breaking a computation down into `StateMachine` steps, instead of a
monolithic function, provides the hooks needed to *suspend* and *resume* a
computation. When `StateMachine.step` returns, there is an explicit *suspension*
point. The continuation specified by the returned `StateMachine` value is an
explicit *resume* point. Recomputation can thus be avoided because the
computation can be picked up exactly where it left off.

### Callbacks, continuations and asynchronous computation

In technical terms, a `StateMachine` serves as a *continuation*, determining the
subsequent computation to be executed. Instead of blocking, a `StateMachine` can
voluntarily *suspend* by returning from the `step` function, which transfers
control back to a [`Driver`](#drivers-and-bridging) instance. The `Driver` can
then switch to a ready `StateMachine` or relinquish control back to Skyframe.

Traditionally, *callbacks* and *continuations* are conflated into one concept.
However, `StateMachine`s maintain a distinction between the two.

*   *Callback* - describes where to store the result of an asynchronous
    computation.
*   *Continuation* - specifies the next execution state.

Callbacks are required when invoking an asynchronous operation, which means that
the actual operation doesn't occur immediately upon calling the method, as in
the case of a SkyValue lookup. Callbacks should be kept as simple as possible.

Caution: A common pitfall of callbacks is that the asynchronous computation must
ensure the callback is called by the end of every reachable path. It's possible
to overlook some branches and the compiler doesn't give warnings about this.

*Continuations* are the `StateMachine` return values of `StateMachine`s and
encapsulate the complex execution that follows once all asynchronous
computations resolve. This structured approach helps to keep the complexity of
callbacks manageable.

## Tasks

The `Tasks` interface provides `StateMachine`s with an API to lookup SkyValues
by SkyKey and to schedule concurrent subtasks.

```
interface Tasks {
  void enqueue(StateMachine subtask);

  void lookUp(SkyKey key, Consumer<SkyValue> sink);

  <E extends Exception>
  void lookUp(SkyKey key, Class<E> exceptionClass, ValueOrExceptionSink<E> sink);

  // lookUp overloads for 2 and 3 exception types exist, but are elided here.
}
```

Tip: When any state uses the `Tasks` interface to perform lookups or create
subtasks, those lookups and subtasks will complete before the next state begins.

Tip: (Corollary) If subtasks are complex `StateMachine`s or recursively create
subtasks, they all *transitively* complete before the next state begins.

### SkyValue lookups {:#skyvalue-lookups}

`StateMachine`s use `Tasks.lookUp` overloads to look up SkyValues. They are
analogous to `SkyFunction.Environment.getValue` and
`SkyFunction.Environment.getValueOrThrow` and have similar exception handling
semantics. The implementation does not immediately perform the lookup, but
instead, batches[^4] as many lookups as possible before doing so. The value
might not be immediately available, for example, requiring a Skyframe restart,
so the caller specifies what to do with the resulting value using a callback.

The `StateMachine` processor ([`Driver`s and bridging to
SkyFrame](#drivers-and-bridging)) guarantees that the value is available before
the next state begins. An example follows.

```
class DoesLookup implements StateMachine, Consumer<SkyValue> {
  private Value value;

  @Override
  public StateMachine step(Tasks tasks) {
    tasks.lookUp(new Key(), (Consumer<SkyValue>) this);
    return this::processValue;
  }

  // The `lookUp` call in `step` causes this to be called before `processValue`.
  @Override  // Implementation of Consumer<SkyValue>.
  public void accept(SkyValue value) {
    this.value = (Value)value;
  }

  private StateMachine processValue(Tasks tasks) {
    System.out.println(value);  // Prints the string representation of `value`.
    return DONE;
  }
}
```

In the above example, the first step does a lookup for `new Key()`, passing
`this` as the consumer. That is possible because `DoesLookup` implements
`Consumer<SkyValue>`.

Tip: When passing `this` as a value sink, it's helpful to readers to upcast it
to the receiver type to narrow down the purpose of passing `this`. The example
passes `(Consumer<SkyValue>) this`.

By contract, before the next state `DoesLookup.processValue` begins, all the
lookups of `DoesLookup.step` are complete. Therefore `value` is available when
it is accessed in `processValue`.

### Subtasks

`Tasks.enqueue` requests the execution of logically concurrent subtasks.
Subtasks are also `StateMachine`s and can do anything regular `StateMachine`s
can do, including recursively creating more subtasks or looking up SkyValues.
Much like `lookUp`, the state machine driver ensures that all subtasks are
complete before proceeding to the next step. An example follows.

```
class Subtasks implements StateMachine {
  private int i = 0;

  @Override
  public StateMachine step(Tasks tasks) {
    tasks.enqueue(new Subtask1());
    tasks.enqueue(new Subtask2());
    // The next step is Subtasks.processResults. It won't be called until both
    // Subtask1 and Subtask 2 are complete.
    return this::processResults;
  }

  private StateMachine processResults(Tasks tasks) {
    System.out.println(i);  // Prints "3".
    return DONE;  // Subtasks is done.
  }

  private class Subtask1 implements StateMachine {
    @Override
    public StateMachine step(Tasks tasks) {
      i += 1;
      return DONE;  // Subtask1 is done.
    }
  }

  private class Subtask2 implements StateMachine {
    @Override
    public StateMachine step(Tasks tasks) {
      i += 2;
      return DONE;  // Subtask2 is done.
    }
  }
}
```

Though `Subtask1` and `Subtask2` are logically concurrent, everything runs in a
single thread so the "concurrent" update of `i` does not need any
synchronization.

### Structured concurrency {:#structured-concurrency}

Since every `lookUp` and `enqueue` must resolve before advancing to the next
state, it means that concurrency is naturally limited to tree-structures. It's
possible to create hierarchical[^5] concurrency as shown in the following
example.

![Structured Concurrency](/contribute/images/structured-concurrency.svg)

It's hard to tell from the *UML* that the concurrency structure forms a tree.
There's an [alternate view](#concurrency-tree-diagram) that better shows the
tree structure.

![Unstructured Concurrency](/contribute/images/unstructured-concurrency.svg)

Structured concurrency is much easier to reason about.

## Composition and control flow patterns

This section presents examples for how multiple `StateMachine`s can be composed
and solutions to certain control flow problems.

### Sequential states

This is the most common and straightforward control flow pattern. An example of
this is shown in [Stateful computations inside
`SkyKeyComputeState`](#stateful-computations).

### Branching

Branching states in `StateMachine`s can be achieved by returning different
values using regular *Java* control flow, as shown in the following example.

```
class Branch implements StateMachine {
  @Override
  public StateMachine step(Tasks tasks) {
    // Returns different state machines, depending on condition.
    if (shouldUseA()) {
      return this::performA;
    }
    return this::performB;
  }
  …
}
```

It’s very common for certain branches to return `DONE`, for early completion.

### Advanced sequential composition

Since the `StateMachine` control structure is memoryless, sharing `StateMachine`
definitions as subtasks can sometimes be awkward. Let *M<sub>1</sub>* and
*M<sub>2</sub>* be `StateMachine` instances that share a `StateMachine`, *S*,
with *M<sub>1</sub>* and *M<sub>2</sub>* being the sequences *&lt;A, S, B>* and
*&lt;X, S, Y>* respectively. The problem is that *S* doesn’t know whether to
continue to *B* or *Y* after it completes and `StateMachine`s don't quite keep a
call stack. This section reviews some techniques for achieving this.

#### `StateMachine` as terminal sequence element

This doesn’t solve the initial problem posed. It only demonstrates sequential
composition when the shared `StateMachine` is terminal in the sequence.

```
// S is the shared state machine.
class S implements StateMachine { … }

class M1 implements StateMachine {
  @Override
  public StateMachine step(Tasks tasks) {
    performA();
    return new S();
  }
}

class M2 implements StateMachine {
  @Override
  public StateMachine step(Tasks tasks) {
    performX();
    return new S();
  }
}
```

This works even if *S* is itself a complex state machine.

#### Subtask for sequential composition

Since enqueued subtasks are guaranteed to complete before the next state, it’s
sometimes possible to slightly abuse[^6] the subtask mechanism.

```
class M1 implements StateMachine {
  @Override
  public StateMachine step(Tasks tasks) {
    performA();
    // S starts after `step` returns and by contract must complete before `doB`
    // begins. It is effectively sequential, inducing the sequence < A, S, B >.
    tasks.enqueue(new S());
    return this::doB;
  }

  private StateMachine doB(Tasks tasks) {
    performB();
    return DONE;
  }
}

class M2 implements StateMachine {
  @Override
  public StateMachine step(Tasks tasks) {
    performX();
    // Similarly, this induces the sequence < X, S, Y>.
    tasks.enqueue(new S());
    return this::doY;
  }

  private StateMachine doY(Tasks tasks) {
    performY();
    return DONE;
  }
}
```

#### `runAfter` injection {:#runafter-injection}

Sometimes, abusing `Tasks.enqueue` is impossible because there are other
parallel subtasks or `Tasks.lookUp` calls that must be completed before *S*
executes. In this case, injecting a `runAfter` parameter into *S* can be used to
inform *S* of what to do next.

```
class S implements StateMachine {
  // Specifies what to run after S completes.
  private final StateMachine runAfter;

  @Override
  public StateMachine step(Tasks tasks) {
    … // Performs some computations.
    return this::processResults;
  }

  @Nullable
  private StateMachine processResults(Tasks tasks) {
    … // Does some additional processing.

    // Executes the state machine defined by `runAfter` after S completes.
    return runAfter;
  }
}

class M1 implements StateMachine {
  @Override
  public StateMachine step(Tasks tasks) {
    performA();
    // Passes `this::doB` as the `runAfter` parameter of S, resulting in the
    // sequence < A, S, B >.
    return new S(/* runAfter= */ this::doB);
  }

  private StateMachine doB(Tasks tasks) {
    performB();
    return DONE;
  }
}

class M2 implements StateMachine {
  @Override
  public StateMachine step(Tasks tasks) {
    performX();
    // Passes `this::doY` as the `runAfter` parameter of S, resulting in the
    // sequence < X, S, Y >.
    return new S(/* runAfter= */ this::doY);
  }

  private StateMachine doY(Tasks tasks) {
    performY();
    return DONE;
  }
}
```

This approach is cleaner than abusing subtasks. However, applying this too
liberally, for example, by nesting multiple `StateMachine`s with `runAfter`, is
the road to [Callback Hell](#callback-hell). It’s better to break up sequential
`runAfter`s with ordinary sequential states instead.

```
  return new S(/* runAfter= */ new T(/* runAfter= */ this::nextStep))
```

can be replaced with the following.

```
  private StateMachine step1(Tasks tasks) {
     doStep1();
     return new S(/* runAfter= */ this::intermediateStep);
  }

  private StateMachine intermediateStep(Tasks tasks) {
    return new T(/* runAfter= */ this::nextStep);
  }
```

Note: It's possible to pass `DONE` as the `runAfter` parameter when there's
nothing to run afterwards.

Tip: When using `runAfter`, always annotate the parameter with `/* runAfter= */`
to let the reader know the meaning at the callsite.

#### *Forbidden* alternative: `runAfterUnlessError`

In an earlier draft, we had considered a `runAfterUnlessError` that would abort
early on errors. This was motivated by the fact that errors often end up getting
checked twice, once by the `StateMachine` that has a `runAfter` reference and
once by the `runAfter` machine itself.

After some deliberation, we decided that uniformity of the code is more
important than deduplicating the error checking. It would be confusing if the
`runAfter` mechanism did not work in a consistent manner with the
`tasks.enqueue` mechanism, which always requires error checking.

Warning: When using `runAfter`, the machine that has the injected `runAfter`
should invoke it unconditionally at completion, even on error, for consistency.

### Direct delegation

Each time there is a formal state transition, the main `Driver` loop advances.
As per contract, advancing states means that all previously enqueued SkyValue
lookups and subtasks resolve before the next state executes. Sometimes the logic
of a delegate `StateMachine` makes a phase advance unnecessary or
counterproductive. For example, if the first `step` of the delegate performs
SkyKey lookups that could be parallelized with lookups of the delegating state
then a phase advance would make them sequential. It could make more sense to
perform direct delegation, as shown in the example below.

```
class Parent implements StateMachine {
  @Override
  public StateMachine step(Tasks tasks ) {
    tasks.lookUp(new Key1(), this);
    // Directly delegates to `Delegate`.
    //
    // The (valid) alternative:
    //   return new Delegate(this::afterDelegation);
    // would cause `Delegate.step` to execute after `step` completes which would
    // cause lookups of `Key1` and `Key2` to be sequential instead of parallel.
    return new Delegate(this::afterDelegation).step(tasks);
  }

  private StateMachine afterDelegation(Tasks tasks) {
    …
  }
}

class Delegate implements StateMachine {
  private final StateMachine runAfter;

  Delegate(StateMachine runAfter) {
    this.runAfter = runAfter;
  }

  @Override
  public StateMachine step(Tasks tasks) {
    tasks.lookUp(new Key2(), this);
    return …;
  }

  // Rest of implementation.
  …

  private StateMachine complete(Tasks tasks) {
    …
    return runAfter;
  }
}
```

## Data flow

The focus of the previous discussion has been on managing control flow. This
section describes the propagation of data values.

### Implementing `Tasks.lookUp` callbacks

There’s an example of implementing a `Tasks.lookUp` callback in [SkyValue
lookups](#skyvalue-lookups). This section provides rationale and suggests
approaches for handling multiple SkyValues.

#### `Tasks.lookUp` callbacks {:#tasks-lookup-callbacks}

The `Tasks.lookUp` method takes a callback, `sink`, as a parameter.

```
  void lookUp(SkyKey key, Consumer<SkyValue> sink);
```

The idiomatic approach would be to use a *Java* lambda to implement this:

```
  tasks.lookUp(key, value -> myValue = (MyValueClass)value);
```

with `myValue` being a member variable of the `StateMachine` instance doing the
lookup. However, the lambda requires an extra memory allocation compared to
implementing the `Consumer<SkyValue>` interface in the `StateMachine`
implementation. The lambda is still useful when there are multiple lookups that
would be ambiguous.

Note: Bikeshed warning. There is a noticeable difference of approximately 1%
end-to-end CPU usage when implementing callbacks systematically in
`StateMachine` implementations compared to using lambdas, which makes this
recommendation debatable. To avoid unnecessary debates, it is advised to leave
the decision up to the individual implementing the solution.

There are also error handling overloads of `Tasks.lookUp`, that are analogous to
`SkyFunction.Environment.getValueOrThrow`.

```
  <E extends Exception> void lookUp(
      SkyKey key, Class<E> exceptionClass, ValueOrExceptionSink<E> sink);

  interface ValueOrExceptionSink<E extends Exception> {
    void acceptValueOrException(@Nullable SkyValue value, @Nullable E exception);
  }
```

An example implementation is shown below.

```
class PerformLookupWithError extends StateMachine, ValueOrExceptionSink<MyException> {
  private MyValue value;
  private MyException error;

  @Override
  public StateMachine step(Tasks tasks) {
    tasks.lookUp(new MyKey(), MyException.class, ValueOrExceptionSink<MyException>) this);
    return this::processResult;
  }

  @Override
  public acceptValueOrException(@Nullable SkyValue value, @Nullable MyException exception) {
    if (value != null) {
      this.value = (MyValue)value;
      return;
    }
    if (exception != null) {
      this.error = exception;
      return;
    }
    throw new IllegalArgumentException("Both parameters were unexpectedly null.");
  }

  private StateMachine processResult(Tasks tasks) {
    if (exception != null) {
      // Handles the error.
      …
      return DONE;
    }
    // Processes `value`, which is non-null.
    …
  }
}
```

As with lookups without error handling, having the `StateMachine` class directly
implement the callback saves a memory allocation for the lamba.

[Error handling](#error-handling) provides a bit more detail, but essentially,
there's not much difference between the propagation of errors and normal values.

#### Consuming multiple SkyValues

Multiple SkyValue lookups are often required. An approach that works much of the
time is to switch on the type of SkyValue. The following is an example that has
been simplified from prototype production code.

```
  @Nullable
  private StateMachine fetchConfigurationAndPackage(Tasks tasks) {
    var configurationKey = configuredTarget.getConfigurationKey();
    if (configurationKey != null) {
      tasks.lookUp(configurationKey, (Consumer<SkyValue>) this);
    }

    var packageId = configuredTarget.getLabel().getPackageIdentifier();
    tasks.lookUp(PackageValue.key(packageId), (Consumer<SkyValue>) this);

    return this::constructResult;
  }

  @Override  // Implementation of `Consumer<SkyValue>`.
  public void accept(SkyValue value) {
    if (value instanceof BuildConfigurationValue) {
      this.configurationValue = (BuildConfigurationValue) value;
      return;
    }
    if (value instanceof PackageValue) {
      this.pkg = ((PackageValue) value).getPackage();
      return;
    }
    throw new IllegalArgumentException("unexpected value: " + value);
  }
```

The `Consumer<SkyValue>` callback implementation can be shared unambiguously
because the value types are different. When that’s not the case, falling back to
lambda-based implementations or full inner-class instances that implement the
appropriate callbacks is viable.

### Propagating values between `StateMachine`s {:#propagating-values}

So far, this document has only explained how to arrange work in a subtask, but
subtasks also need to report a values back to the caller. Since subtasks are
logically asynchronous, their results are communicated back to the caller using
a *callback*. To make this work, the subtask defines a sink interface that is
injected via its constructor.

```
class BarProducer implements StateMachine {
  // Callers of BarProducer implement the following interface to accept its
  // results. Exactly one of the two methods will be called by the time
  // BarProducer completes.
  interface ResultSink {
    void acceptBarValue(Bar value);
    void acceptBarError(BarException exception);
  }

  private final ResultSink sink;

  BarProducer(ResultSink sink) {
     this.sink = sink;
  }

  … // StateMachine steps that end with this::complete.

  private StateMachine complete(Tasks tasks) {
    if (hasError()) {
      sink.acceptBarError(getError());
      return DONE;
    }
    sink.acceptBarValue(getValue());
    return DONE;
  }
}
```

Tip: It would be tempting to use the more concise signature void `accept(Bar
value)` rather than the stuttery `void acceptBarValue(Bar value)` above.
However, `Consumer<SkyValue>` is a common overload of `void accept(Bar value)`,
so doing this often leads to violations of the [Overloads: never
split](https://google.github.io/styleguide/javaguide.html#s3.4.2-ordering-class-contents)
style-guide rule.

Tip: Using a custom `ResultSink` type instead of a generic one from
`java.util.function` makes it easy to find implementations in the code base,
improving readability.

A caller `StateMachine` would then look like the following.

```
class Caller implements StateMachine, BarProducer.ResultSink {
  interface ResultSink {
    void acceptCallerValue(Bar value);
    void acceptCallerError(BarException error);
  }

  private final ResultSink sink;

  private Bar value;

  Caller(ResultSink sink) {
    this.sink = sink;
  }

  @Override
  @Nullable
  public StateMachine step(Tasks tasks) {
    tasks.enqueue(new BarProducer((BarProducer.ResultSink) this));
    return this::processResult;
  }

  @Override
  public void acceptBarValue(Bar value) {
    this.value = value;
  }

  @Override
  public void acceptBarError(BarException error) {
    sink.acceptCallerError(error);
  }

  private StateMachine processResult(Tasks tasks) {
    // Since all enqueued subtasks resolve before `processResult` starts, one of
    // the `BarResultSink` callbacks must have been called by this point.
    if (value == null) {
      return DONE;  // There was a previously reported error.
    }
    var finalResult = computeResult(value);
    sink.acceptCallerValue(finalResult);
    return DONE;
  }
}
```

The preceding example demonstrates a few things. `Caller` has to propagate its
results back and defines its own `Caller.ResultSink`. `Caller` implements the
`BarProducer.ResultSink` callbacks. Upon resumption, `processResult` checks if
`value` is null to determine if an error occurred. This is a common behavior
pattern after accepting output from either a subtask or SkyValue lookup.

Note that the implementation of `acceptBarError` eagerly forwards the result to
the `Caller.ResultSink`, as required by [Error bubbling](#error-bubbling).

Alternatives for top-level `StateMachine`s are described in [`Driver`s and
bridging to SkyFunctions](#drivers-and-bridging).

### Error handling {:#error-handling}

There's a couple of examples of error handling already in [`Tasks.lookUp`
callbacks](#tasks-lookup-callbacks) and [Propagating values between
`StateMachines`](#propagating-values). Exceptions, other than
`InterruptedException` are not thrown, but instead passed around through
callbacks as values. Such callbacks often have exclusive-or semantics, with
exactly one of a value or error being passed.

The next section describes a a subtle, but important interaction with Skyframe
error handling.

#### Error bubbling (--nokeep\_going) {:#error-bubbling}

Warning: Errors need to be eagerly propagated all the way back to the
SkyFunction for error bubbling to function correctly.

During error bubbling, a SkyFunction may be restarted even if not all requested
SkyValues are available. In such cases, the subsequent state will never be
reached due to the `Tasks` API contract. However, the `StateMachine` should
still propagate the exception.

Since propagation must occur regardless of whether the next state is reached,
the error handling callback must perform this task. For an inner `StateMachine`,
this is achieved by invoking the parent callback.

At the top-level `StateMachine`, which interfaces with the SkyFunction, this can
be done by calling the `setException` method of `ValueOrExceptionProducer`.
`ValueOrExceptionProducer.tryProduceValue` will then throw the exception, even
if there are missing SkyValues.

If a `Driver` is being utilized directly, it is essential to check for
propagated errors from the SkyFunction, even if the machine has not finished
processing.

### Event Handling {:#event-handling}

For SkyFunctions that need to emit events, a `StoredEventHandler` is injected
into SkyKeyComputeState and further injected into `StateMachine`s that require
them. Historically, the `StoredEventHandler` was needed due to Skyframe dropping
certain events unless they are replayed but this was subsequently fixed.
`StoredEventHandler` injection is preserved because it simplifies the
implementation of events emitted from error handling callbacks.

## `Driver`s and bridging to SkyFunctions {:#drivers-and-bridging}

A `Driver` is responsible for managing the execution of `StateMachine`s,
beginning with a specified root `StateMachine`. As `StateMachine`s can
recursively enqueue subtask `StateMachine`s, a single `Driver` can manage
numerous subtasks. These subtasks create a tree structure, a result of
[Structured concurrency](#structured-concurrency). The `Driver` batches SkyValue
lookups across subtasks for improved efficiency.

There are a number of classes built around the `Driver`, with the following API.

```
public final class Driver {
  public Driver(StateMachine root);
  public boolean drive(SkyFunction.Environment env) throws InterruptedException;
}
```

`Driver` takes a single root `StateMachine` as a parameter. Calling
`Driver.drive` executes the `StateMachine` as far as it can go without a
Skyframe restart. It returns true when the `StateMachine` completes and false
otherwise, indicating that not all values were available.

`Driver` maintains the concurrent state of the `StateMachine` and it is well
suited for embedding in `SkyKeyComputeState`.

### Directly instantiating `Driver`

`StateMachine` implementations conventionally communicate their results via
callbacks. It's possible to directly instantiate a `Driver` as shown in the
following example.

The `Driver` is embedded in the `SkyKeyComputeState` implementation along with
an implementation of the corresponding `ResultSink` to be defined a bit further
down. At the top level, the `State` object is an appropriate receiver for the
result of the computation as it is guaranteed to outlive `Driver`.

```
class State implements SkyKeyComputeState, ResultProducer.ResultSink {
  // The `Driver` instance, containing the full tree of all `StateMachine`
  // states. Responsible for calling `StateMachine.step` implementations when
  // asynchronous values are available and performing batched SkyFrame lookups.
  //
  // Non-null while `result` is being computed.
  private Driver resultProducer;

  // Variable for storing the result of the `StateMachine`
  //
  // Will be non-null after the computation completes.
  //
  private ResultType result;

  // Implements `ResultProducer.ResultSink`.
  //
  // `ResultProducer` propagates its final value through a callback that is
  // implemented here.
  @Override
  public void acceptResult(ResultType result) {
    this.result = result;
  }
}
```

The code below sketches the `ResultProducer`.

```
class ResultProducer implements StateMachine {
  interface ResultSink {
    void acceptResult(ResultType value);
  }

  private final Parameters parameters;
  private final ResultSink sink;

  … // Other internal state.

  ResultProducer(Parameters parameters, ResultSink sink) {
    this.parameters = parameters;
    this.sink = sink;
  }

  @Override
  public StateMachine step(Tasks tasks) {
    …  // Implementation.
    return this::complete;
  }

  private StateMachine complete(Tasks tasks) {
    sink.acceptResult(getResult());
    return DONE;
  }
}
```

Then the code for lazily computing the result could look like the following.

```
@Nullable
private Result computeResult(State state, Skyfunction.Environment env)
    throws InterruptedException {
  if (state.result != null) {
    return state.result;
  }
  if (state.resultProducer == null) {
    state.resultProducer = new Driver(new ResultProducer(
      new Parameters(), (ResultProducer.ResultSink)state));
  }
  if (state.resultProducer.drive(env)) {
    // Clears the `Driver` instance as it is no longer needed.
    state.resultProducer = null;
  }
  return state.result;
}
```

### Embedding `Driver` {:#embedding-driver}

If the `StateMachine` produces a value and raises no exceptions, embedding
`Driver` is another possible implementation, as shown in the following example.

```
class ResultProducer implements StateMachine {
  private final Parameters parameters;
  private final Driver driver;

  private ResultType result;

  ResultProducer(Parameters parameters) {
    this.parameters = parameters;
    this.driver = new Driver(this);
  }

  @Nullable  // Null when a Skyframe restart is needed.
  public ResultType tryProduceValue( SkyFunction.Environment env)
      throws InterruptedException {
    if (!driver.drive(env)) {
      return null;
    }
    return result;
  }

  @Override
  public StateMachine step(Tasks tasks) {
    …  // Implementation.
}
```

The SkyFunction may have code that looks like the following (where `State` is
the function specific type of `SkyKeyComputeState`).

```
@Nullable  // Null when a Skyframe restart is needed.
Result computeResult(SkyFunction.Environment env, State state)
    throws InterruptedException {
  if (state.result != null) {
    return state.result;
  }
  if (state.resultProducer == null) {
    state.resultProducer = new ResultProducer(new Parameters());
  }
  var result = state.resultProducer.tryProduceValue(env);
  if (result == null) {
    return null;
  }
  state.resultProducer = null;
  return state.result = result;
}
```

Embedding `Driver` in the `StateMachine` implementation is a better fit for
Skyframe's synchronous coding style.

### StateMachines that may produce exceptions

Otherwise, there are `SkyKeyComputeState`-embeddable `ValueOrExceptionProducer`
and `ValueOrException2Producer` classes that have synchronous APIs to match
synchronous SkyFunction code.

The `ValueOrExceptionProducer` abstract class includes the following methods.

```
public abstract class ValueOrExceptionProducer<V, E extends Exception>
    implements StateMachine {
  @Nullable
  public final V tryProduceValue(Environment env)
      throws InterruptedException, E {
    …  // Implementation.
  }

  protected final void setValue(V value)  {  … // Implementation. }
  protected final void setException(E exception) {  … // Implementation. }
}
```

It includes an embedded `Driver` instance and closely resembles the
`ResultProducer` class in [Embedding driver](#embedding-driver) and interfaces
with the SkyFunction in a similar manner. Instead of defining a `ResultSink`,
implementations call `setValue` or `setException` when either of those occur.
When both occur, the exception takes priority. The `tryProduceValue` method
bridges the asynchronous callback code to synchronous code and throws an
exception when one is set.

As previously noted, during error bubbling, it's possible for an error to occur
even if the machine is not yet done because not all inputs are available. To
accommodate this, `tryProduceValue` throws any set exceptions, even before the
machine is done.

## Epilogue: Eventually removing callbacks

`StateMachine`s are a highly efficient, but boilerplate intensive way to perform
asynchronous computation. Continuations (particularly in the form of `Runnable`s
passed to `ListenableFuture`) are widespread in certain parts of *Bazel* code,
but aren't prevalent in analysis SkyFunctions. Analysis is mostly CPU bound and
there are no efficient asynchronous APIs for disk I/O. Eventually, it would be
good to optimize away callbacks as they have a learning curve and impede
readability.

One of the most promising alternatives is *Java* virtual threads. Instead of
having to write callbacks, everything is replaced with synchronous, blocking
calls. This is possible because tying up a virtual thread resource, unlike a
platform thread, is supposed to be cheap. However, even with virtual threads,
replacing simple synchronous operations with thread creation and synchronization
primitives is too expensive. We performed a migration from `StateMachine`s to
*Java* virtual threads and they were orders of magnitude slower, leading to
almost a 3x increase in end-to-end analysis latency. Since virtual threads are
still a preview feature, it's possible that this migration can be performed at a
later date when performance improves.

Another approach to consider is waiting for *Loom* coroutines, if they ever
become available. The advantage here is that it might be possible to reduce
synchronization overhead by using cooperative multitasking.

If all else fails, low-level bytecode rewriting could also be a viable
alternative. With enough optimization, it might be possible to achieve
performance that approaches hand-written callback code.

## Appendix

### Callback Hell {:#callback-hell}

Callback hell is an infamous problem in asynchronous code that uses callbacks.
It stems from the fact that the continuation for a subsequent step is nested
within the previous step. If there are many steps, this nesting can be extremely
deep. If coupled with control flow the code becomes unmanageable.

```
class CallbackHell implements StateMachine {
  @Override
  public StateMachine step(Tasks task) {
    doA();
    return (t, l) -> {
      doB();
      return (t1, l2) -> {
        doC();
        return DONE;
      };
    };
  }
}
```

One of the advantages of nested implementations is that the stack frame of the
outer step can be preserved. In *Java*, captured lambda variables must be
effectively final so using such variables can be cumbersome. Deep nesting is
avoided by returning method references as continuations instead of lambdas as
shown as follows.

```
class CallbackHellAvoided implements StateMachine {
  @Override
  public StateMachine step(Tasks task) {
    doA();
    return this::step2;
  }

  private StateMachine step2(Tasks tasks) {
    doB();
    return this::step3;
  }

  private StateMachine step3(Tasks tasks) {
    doC();
    return DONE;
  }
}
```

Callback hell may also occur if the [`runAfter` injection](#runafter-injection)
pattern is used too densely, but this can be avoided by interspersing injections
with sequential steps.

#### Example: Chained SkyValue lookups {:#chained-skyvalue-lookups}

It is often the case that the application logic requires dependent chains of
SkyValue lookups, for example, if a second SkyKey depends on the first SkyValue.
Thinking about this naively, this would result in a complex, deeply nested
callback structure.

```
private ValueType1 value1;
private ValueType2 value2;

private StateMachine step1(...) {
  tasks.lookUp(key1, (Consumer<SkyValue>) this);  // key1 has type KeyType1.
  return this::step2;
}

@Override
public void accept(SkyValue value) {
  this.value1 = (ValueType1) value;
}

private StateMachine step2(...) {
  KeyType2 key2 = computeKey(value1);
  tasks.lookup(key2, this::acceptValueType2);
  return this::step3;
}

private void acceptValueType2(SkyValue value) {
  this.value2 = (ValueType2) value;
}
```

However, since continuations are specified as method references, the code looks
procedural across state transitions: `step2` follows `step1`. Note that here, a
lambda is used to assign `value2`. This makes the ordering of the code match the
ordering of the computation from top-to-bottom.

### Miscellaneous Tips

#### Readability: Execution Ordering

To improve readability, strive to keep the `StateMachine.step` implementations
in execution order and callback implementations immediately following where they
are passed in the code. This isn't always possible where the control flow
branches. Additional comments might be helpful in such cases.

In [Example: Chained SkyValue lookups](#chained-skyvalue-lookups), an
intermediate method reference is created to achieve this. This trades a small
amount of performance for readability, which is likely worthwhile here.

#### Generational Hypothesis

Medium-lived *Java* objects break the generational hypothesis of the *Java*
garbage collector, which is designed to handle objects that live for a very
short time or objects that live forever. By definition, objects in
`SkyKeyComputeState` violate this hypothesis. Such objects, containing the
constructed tree of all still-running `StateMachine`s, rooted at `Driver` have
an intermediate lifespan as they suspend, waiting for asynchronous computations
to complete.

It seems less bad in JDK19, but when using `StateMachine`s, it's sometimes
possible to observe an increase in GC time, even with dramatic decreases in
actual garbage generated. Since `StateMachine`s have an intermediate lifespan
they could be promoted to old gen, causing it to fill up more quickly, thus
necessitating more expensive major or full GCs to clean up.

The initial precaution is to minimize the use of `StateMachine` variables, but
it is not always feasible, for example, if a value is needed across multiple
states. Where it is possible, local stack `step` variables are young generation
variables and efficiently GC'd.

For `StateMachine` variables, breaking things down into subtasks and following
the recommended pattern for [Propagating values between
`StateMachine`s](#propagating-values) is also helpful. Observe that when
following the pattern, only child `StateMachine`s have references to parent
`StateMachine`s and not vice versa. This means that as children complete and
update the parents using result callbacks, the children naturally fall out of
scope and become eligible for GC.

Finally, in some cases, a `StateMachine` variable is needed in earlier states
but not in later states. It can be beneficial to null out references of large
objects once it is known that they are no longer needed.

#### Naming states

When naming a method, it's usually possible to name a method for the behavior
that happens within that method. It's less clear how to do this in
`StateMachine`s because there is no stack. For example, suppose method `foo`
calls a sub-method `bar`. In a `StateMachine`, this could be translated into the
state sequence `foo`, followed by `bar`. `foo` no longer includes the behavior
`bar`. As a result, method names for states tend to be narrower in scope,
potentially reflecting local behavior.

### Concurrency tree diagram {:#concurrency-tree-diagram}

The following is an alternative view of the diagram in [Structured
concurrency](#structured-concurrency) that better depicts the tree structure.
The blocks form a small tree.

![Structured Concurrency 3D](/contribute/images/structured-concurrency-3d.svg)

[^1]: In contrast to Skyframe's convention of restarting from the beginning when
 values are not available.
[^2]: Note that `step` is permitted to throw `InterruptedException`, but the
 examples omit this. There are a few low methods in *Bazel* code that throw
 this exception and it propagates up to the `Driver`, to be described later,
 that runs the `StateMachine`. It's fine to not declare it to be thrown when
 unneeded.
[^3]: Concurrent subtasks were motivated by the `ConfiguredTargetFunction` which
 performs *independent* work for each dependency. Instead of manipulating
 complex data structures that process all the dependencies at once,
 introducing inefficiencies, each dependency has its own independent
 `StateMachine`.
[^4]: Multiple `tasks.lookUp` calls within a single step are batched together.
 Additional batching can be created by lookups occurring within concurrent
 subtasks.
[^5]: This is conceptually similar to Java’s structured concurrency
 [jeps/428](https://openjdk.org/jeps/428).
[^6]: Doing this is similar to spawning a thread and joining it to achieve
 sequential composition.