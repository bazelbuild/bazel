**ProGuard** is a free Java class file shrinker, optimizer, obfuscator, and
preverifier. It detects and removes unused classes, fields, methods, and
attributes. It optimizes bytecode and removes unused instructions. It renames
the remaining classes, fields, and methods using short meaningless names. The
resulting applications and libraries are smaller, faster, and a bit better
hardened against reverse engineering.

Typical applications:

- Reducing the size of Android apps for faster downloads, shorter startup
  times, and smaller memory footprints.
- Optimizing code for better performance on mobile devices.

**ProGuard**'s main advantage compared to other Java obfuscators is probably
its compact template-based configuration. A few intuitive command line options
or a simple configuration file are usually sufficient. The user manual
explains all available options and shows examples of this powerful
configuration style.

**ProGuard** is fast. It only takes seconds to process programs and libraries
of several megabytes. The results section presents actual figures for a number
of applications.

**ProGuard** is a command-line tool with an optional graphical user interface.
It also comes with plugins for Ant, for Gradle, and for the JME Wireless
Toolkit. It is already part of Google's Android SDK, where it can be enabled
with a simple flag.

**ProGuard** provides basic protection against reverse engineering and
tampering, with basic name obfuscation.
[**DexGuard**](http://www.guardsquare.com/dexguard), its specialized
commercial extension for Android, focuses further on the protection of apps,
additionally optimizing, obfuscating and encrypting strings, classes,
resources, resource files, asset files, and native libraries. Professional
developers should definitely consider it for security-sensitive apps.

