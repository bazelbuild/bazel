# Standard constraint_setting and constraint_values to be used in platforms.

package(
    default_visibility = ["//visibility:public"],
)

# These match values in //src/main/java/com/google/build/lib/util:CPU.java
constraint_setting(name = "cpu")

constraint_value(
    name = "x86_32",
    constraint_setting = ":cpu",
)

constraint_value(
    name = "x86_64",
    constraint_setting = ":cpu",
)

constraint_value(
    name = "ppc",
    constraint_setting = ":cpu",
)

constraint_value(
    name = "arm",
    constraint_setting = ":cpu",
)

constraint_value(
    name = "x390x",
    constraint_setting = ":cpu",
)

# These match values in //src/main/java/com/google/build/lib/util:OS.java
constraint_setting(name = "os")

constraint_value(
    name = "osx",
    constraint_setting = ":os",
)

constraint_value(
    name = "freebsd",
    constraint_setting = ":os",
)

constraint_value(
    name = "linux",
    constraint_setting = ":os",
)

constraint_value(
    name = "windows",
    constraint_setting = ":os",
)

# A default platform referring to the host system. This only exists for
# internal build configurations, and so can't be accessed by other packages.
platform(
    name = "host_platform",
    host_cpu_constraints = [
        ":x86_32",
        ":x86_64",
        ":ppc",
        ":arm",
        ":x390x",
    ],
    host_os_constraints = [
        ":osx",
        ":freebsd",
        ":linux",
        ":windows",
    ],
    host_platform = True,
)
