## Analysis Producers

Analysis is driven by `StateMachine`[^documentation] implementations that
memoize computations across Skyframe restarts, and enable extremely efficient
concurrency. Concurrency is crucial in this context, as analysis involves
applying a series of processing steps to multiple dependencies. By employing
concurrency, these steps can proceed independently rather than being delayed by
stragglers.

`StateMachines` serve as composable building blocks, and this document
illustrates how the analysis producers fit together.

### `DependencyContextProducer`

The `DependencyContextProducer` is used by `AspectFunction`. It first computes
the unloaded toolchain contexts, which as a side effect, computes
`PlatformInfo`, a prerequisite of `ConfigConditionsProducer`.

```none
                   +---------------------------------+
                   |    DependencyContextProducer    |
                   +---------------------------------+
                                  ^         ^
                                  |         |
                                  |         |
+-----------------------------------+     +---------------------------------+
| UnloadedToolchainContextsProducer |     |    ConfigConditionsProducer     |
+-----------------------------------+     +---------------------------------+
                                            ^
                                            |
                                            |
                                          +---------------------------------+
                                          | ConfiguredTargetAndDataProducer |
                                          +---------------------------------+
```

### `DependencyContextProducerWithCompatibilityCheck`

The `DependencyContextProducerWithCompatibilityCheck` is used by the
`ConfiguredTargetFunction`. Here, a check for incompatibility needs to happen
before the unloaded toolchain contexts are computed to avoid failing the build
due to incompatibility. So instead of obtaining `PlatformInfo` as a side effect
of computing the unloaded toolchain contexts, it is first computed directly
using `PlatformInfoProducer`. The `PlatformInfo` is then used to compute
`ConfigConditions` via the `ConfigConditionsProducer` and the resulting
`ConfigConditions` are used in the compatibility check. Only after the check
passes does it finally compute the unloaded toolchain contexts.

```none
+-----------------------------------++----------------------------+
| UnloadedToolchainContextsProducer || IncompatibleTargetProducer |
+-----------------------------------++----------------------------+
  |                                    |
  |                                    |
  v                                    v
+-----------------------------------------------------------------+     +----------------------+
|         DependencyContextProducerWithCompatibilityCheck         | <-- | PlatformInfoProducer |
+-----------------------------------------------------------------+     +----------------------+
  ^
  |
  |
+-----------------------------------+
|     ConfigConditionsProducer      |
+-----------------------------------+
  ^
  |
  |
+-----------------------------------+
|  ConfiguredTargetAndDataProducer  |
+-----------------------------------+
```

[^documentation]: TODO(b/261521010): add link when documentation has been
    checked in.
