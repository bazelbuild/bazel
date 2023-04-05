Project: /_project.yaml
Book: /_book.yaml

# Why a Build System?

{% include "_buttons.html" %}

This page discusses what build systems are, what they do, why you should use a
build system, and why compilers and build scripts aren't the best choice as your
organization starts to scale. It's intended for developers who don't have much
experience with a build system.

## What is a build system?

Fundamentally, all build systems have a straightforward purpose: they transform
the source code written by engineers into executable binaries that can be read
by machines. Build systems aren't just for human-authored code; they also allow
machines to create builds automatically, whether for testing or for releases to
production. In an organization with thousands of engineers, it's common that
most builds are triggered automatically rather than directly by engineers.

### Can't I just use a compiler?

The need for a build system might not be immediately obvious. Most engineers
don't use a build system while learning to code: most start by invoking tools
like `gcc` or `javac` directly from the command line, or the equivalent in an
integrated development environment (IDE). As long as all the source code is in
the same directory, a command like this works fine:

```posix-terminal
javac *.java
```

This instructs the Java compiler to take every Java source file in the current
directory and turn it into a binary class file. In the simplest case, this is
all you need.

However, as soon as code expands, the complications begin. `javac` is smart
enough to look in subdirectories of the current directory to find code to
import. But it has no way of finding code stored in _other parts_ of the
filesystem (perhaps a library shared by several projects). It also only knows
how to build Java code. Large systems often involve different pieces written in
a variety of programming languages with webs of dependencies among those pieces,
meaning no compiler for a single language can possibly build the entire system.

Once you're dealing with code from multiple languages or multiple compilation
units, building code is no longer a one-step process. Now you must evaluate what
your code depends on and build those pieces in the proper order, possibly using
a different set of tools for each piece. If any dependencies change, you must
repeat this process to avoid depending on stale binaries. For a codebase of even
moderate size, this process quickly becomes tedious and error-prone.

The compiler also doesn’t know anything about how to handle external
dependencies, such as third-party `JAR` files in Java. Without a build system,
you could manage this by downloading the dependency from the internet, sticking
it in a `lib` folder on the hard drive, and configuring the compiler to read
libraries from that directory. Over time, it's difficult to maintain the
updates, versions, and source of these external dependencies.

### What about shell scripts?

Suppose that your hobby project starts out simple enough that you can build it
using just a compiler, but you begin running into some of the problems described
previously. Maybe you still don’t think you need a build system and can automate
away the tedious parts using some simple shell scripts that take care of
building things in the correct order. This helps out for a while, but pretty
soon you start running into even more problems:

*   It becomes tedious. As your system grows more complex, you begin spending
    almost as much time working on your build scripts as on real code. Debugging
    shell scripts is painful, with more and more hacks being layered on top of
    one another.

*   It’s slow. To make sure you weren’t accidentally relying on stale libraries,
    you have your build script build every dependency in order every time you
    run it. You think about adding some logic to detect which parts need to be
    rebuilt, but that sounds awfully complex and error prone for a script. Or
    you think about specifying which parts need to be rebuilt each time, but
    then you’re back to square one.

*   Good news: it’s time for a release! Better go figure out all the arguments
    you need to pass to the jar command to make your final build. And remember
    how to upload it and push it out to the central repository. And build and
    push the documentation updates, and send out a notification to users. Hmm,
    maybe this calls for another script...

*   Disaster! Your hard drive crashes, and now you need to recreate your entire
    system. You were smart enough to keep all of your source files in version
    control, but what about those libraries you downloaded? Can you find them
    all again and make sure they were the same version as when you first
    downloaded them? Your scripts probably depended on particular tools being
    installed in particular places—can you restore that same environment so that
    the scripts work again? What about all those environment variables you set a
    long time ago to get the compiler working just right and then forgot about?

*   Despite the problems, your project is successful enough that you’re able to
    begin hiring more engineers. Now you realize that it doesn’t take a disaster
    for the previous problems to arise—you need to go through the same painful
    bootstrapping process every time a new developer joins your team. And
    despite your best efforts, there are still small differences in each
    person’s system. Frequently, what works on one person’s machine doesn’t work
    on another’s, and each time it takes a few hours of debugging tool paths or
    library versions to figure out where the difference is.

*   You decide that you need to automate your build system. In theory, this is
    as simple as getting a new computer and setting it up to run your build
    script every night using cron. You still need to go through the painful
    setup process, but now you don’t have the benefit of a human brain being
    able to detect and resolve minor problems. Now, every morning when you get
    in, you see that last night’s build failed because yesterday a developer
    made a change that worked on their system but didn’t work on the automated
    build system. Each time it’s a simple fix, but it happens so often that you
    end up spending a lot of time each day discovering and applying these simple
    fixes.

*   Builds become slower and slower as the project grows. One day, while waiting
    for a build to complete, you gaze mournfully at the idle desktop of your
    coworker, who is on vacation, and wish there were a way to take advantage of
    all that wasted computational power.

You’ve run into a classic problem of scale. For a single developer working on at
most a couple hundred lines of code for at most a week or two (which might have
been the entire experience thus far of a junior developer who just graduated
university), a compiler is all you need. Scripts can maybe take you a little bit
farther. But as soon as you need to coordinate across multiple developers and
their machines, even a perfect build script isn’t enough because it becomes very
difficult to account for the minor differences in those machines. At this point,
this simple approach breaks down and it’s time to invest in a real build system.
