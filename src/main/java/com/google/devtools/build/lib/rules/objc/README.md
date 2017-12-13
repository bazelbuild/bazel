# The Apple Rule Implementations:
The packages `devtools/build/lib/rules/objc` and
`devtools/build/lib/rules/apple` implement the objc and ios Bazel rules.

## Interfacing from Skylark

Information exchange between skylark rules and native objc_* or ios_* rules
occurs by three mechanisms:

1) **`AppleToolchain:`**

`AppleToolchain.java` houses constants and static methods for use in rule
implementations.  It is accessed in skylark through the global `apple_common`
namespace:

```
def __impl(ctx):
    platform_dir = apple_common.apple_toolchain().platform_dir('iphoneos')
    sdk_dir = apple_common.apple_toolchain().sdk_dir()
```

2) **`AppleConfiguration` and `ObjcConfiguration`**:

In Bazel, configuration fragments are used as containers for invocation-specific
build information (that is, information that cannot always be derived strictly
from BUILD files).  The contents of these configurations can be inspected by
looking at `rules/objc/ObjcConfiguration.java` and
`rules/apple/AppleConfiguration.java`.  To access a configuration fragment from
skylark, the fragment must be declared in the rule definition:

```
def __impl(ctx):
    cpu = ctx.fragments.apple.ios_cpu()
my_rule = rule(
  implementation = __impl
  fragments = ['apple']
)
```

3) **`ObjcProvider`**:

The ObjcProvider maps "keys" to NestedSet instances, where "keys" are singleton
objects defined in ObjcProvider that identify a category of transitive
information to be communicated between targets in a dependency chain.

Native objc/ios rules export ObjcProvider instances, which are made available
to skylark dependants:

```
def __impl(ctx):
    dep = ctx.attr.deps[0]
    objc_provider = dep.objc
```

The provider can be queried by accessing fields that correspond to ObjcProvider
keys.

```
    libraries = objc_provider.library  # A SkylarkNestedSet of Artifacts
```

A skylark rule that is intended to be a dependency of native objc rules should
export an ObjcProvider itself.  An ObjcProvider is constructed using a
constructor exposed on the apple_common namespace.

```
def __impl(ctx):
    define = 'some_define'
    objc_provider = apple_common.new_objc_provider(define=define)
    return struct(objc = objc_provider)
```

Arguments to `new_objc_provider` should correspond to ObjcProvider keys, and
values should be skylark sets that should be added to the provider. Other
instances of ObjcProvider can also be used in provider construction.

```
def __impl(ctx):
    dep = ctx.attr.deps[0]
    define = 'some_define'
    objc_provider = apple_common.new_objc_provider(providers=[dep.objc], define=define)
    return struct(objc = objc_provider)
```
