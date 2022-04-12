**ProGuard** can be seamlessly integrated in Oracle's Wireless Toolkit (WTK)
for Java Micro Edition (JME).

The WTK already comes with a plug-in for ProGuard. Alternatively, ProGuard
offers its own plug-in. This latter implementation is recommended, as it more
up to date and it solves some problems. It is also somewhat more efficient,
invoking the ProGuard engine directly, instead of writing out a configuration
file and running ProGuard in a separate virtual machine.

In order to integrate this plug-in in the toolkit, you'll have to put the
following lines in the file {j2mewtk.dir}`/wtklib/Linux/ktools.properties` or
{j2mewtk.dir}`\wtklib\Windows\ktools.properties` (whichever is applicable).

    obfuscator.runner.class.name: proguard.wtk.ProGuardObfuscator
    obfuscator.runner.classpath: /usr/local/java/proguard/lib/proguard.jar

Please make sure the class path is set correctly for your system.

Once ProGuard has been set up, you can apply it to your projects as part of
the build process. The build process is started from the WTK menu bar:

**Project &rarr; Package &rarr; Create Obfuscated Package**

This option will compile, shrink, obfuscate, verify, and install your midlets
for testing.

Should you ever need to customize your ProGuard configuration for the JME WTK,
you can adapt the configuration file `proguard/wtk/default.pro` that's inside
the `proguard.jar`.
