/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2017 Eric Lafortune @ GuardSquare
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard;

import proguard.classfile.*;
import proguard.classfile.attribute.visitor.AllAttributeVisitor;
import proguard.classfile.editor.*;
import proguard.classfile.visitor.*;
import proguard.obfuscate.Obfuscator;
import proguard.optimize.Optimizer;
import proguard.optimize.peephole.LineNumberLinearizer;
import proguard.preverify.*;
import proguard.shrink.Shrinker;

import java.io.*;

/**
 * Tool for shrinking, optimizing, obfuscating, and preverifying Java classes.
 *
 * @author Eric Lafortune
 */
public class ProGuard
{
    public static final String VERSION = "ProGuard, version 5.3.3";

    private final Configuration configuration;
    private       ClassPool     programClassPool = new ClassPool();
    private final ClassPool     libraryClassPool = new ClassPool();


    /**
     * Creates a new ProGuard object to process jars as specified by the given
     * configuration.
     */
    public ProGuard(Configuration configuration)
    {
        this.configuration = configuration;
    }


    /**
     * Performs all subsequent ProGuard operations.
     */
    public void execute() throws IOException
    {
        System.out.println(VERSION);

        GPL.check();

        if (configuration.printConfiguration != null)
        {
            printConfiguration();
        }

        new ConfigurationChecker(configuration).check();

        if (configuration.programJars != null     &&
            configuration.programJars.hasOutput() &&
            new UpToDateChecker(configuration).check())
        {
            return;
        }

        readInput();

        if (configuration.shrink    ||
            configuration.optimize  ||
            configuration.obfuscate ||
            configuration.preverify)
        {
            clearPreverification();
        }

        if (configuration.printSeeds != null ||
            configuration.shrink    ||
            configuration.optimize  ||
            configuration.obfuscate ||
            configuration.preverify)
        {
            initialize();
        }

        if (configuration.targetClassVersion != 0)
        {
            target();
        }

        if (configuration.printSeeds != null)
        {
            printSeeds();
        }

        if (configuration.shrink)
        {
            shrink();
        }

        if (configuration.preverify)
        {
            inlineSubroutines();
        }

        if (configuration.optimize)
        {
            for (int optimizationPass = 0;
                 optimizationPass < configuration.optimizationPasses;
                 optimizationPass++)
            {
                if (!optimize())
                {
                    // Stop optimizing if the code doesn't improve any further.
                    break;
                }

                // Shrink again, if we may.
                if (configuration.shrink)
                {
                    // Don't print any usage this time around.
                    configuration.printUsage       = null;
                    configuration.whyAreYouKeeping = null;

                    shrink();
                }
            }
        }

        if (configuration.optimize)
        {
            linearizeLineNumbers();
        }

        if (configuration.obfuscate)
        {
            obfuscate();
        }

        if (configuration.optimize)
        {
            trimLineNumbers();
        }

        if (configuration.preverify)
        {
            preverify();
        }

        if (configuration.shrink    ||
            configuration.optimize  ||
            configuration.obfuscate ||
            configuration.preverify)
        {
            sortClassElements();
        }

        if (configuration.programJars.hasOutput())
        {
            writeOutput();
        }

        if (configuration.dump != null)
        {
            dump();
        }
    }


    /**
     * Prints out the configuration that ProGuard is using.
     */
    private void printConfiguration() throws IOException
    {
        if (configuration.verbose)
        {
            System.out.println("Printing configuration to [" + fileName(configuration.printConfiguration) + "]...");
        }

        PrintStream ps = createPrintStream(configuration.printConfiguration);
        try
        {
            new ConfigurationWriter(ps).write(configuration);
        }
        finally
        {
            closePrintStream(ps);
        }
    }


    /**
     * Reads the input class files.
     */
    private void readInput() throws IOException
    {
        if (configuration.verbose)
        {
            System.out.println("Reading input...");
        }

        // Fill the program class pool and the library class pool.
        new InputReader(configuration).execute(programClassPool, libraryClassPool);
    }


    /**
     * Initializes the cross-references between all classes, performs some
     * basic checks, and shrinks the library class pool.
     */
    private void initialize() throws IOException
    {
        if (configuration.verbose)
        {
            System.out.println("Initializing...");
        }

        new Initializer(configuration).execute(programClassPool, libraryClassPool);
    }


    /**
     * Sets that target versions of the program classes.
     */
    private void target() throws IOException
    {
        if (configuration.verbose)
        {
            System.out.println("Setting target versions...");
        }

        new Targeter(configuration).execute(programClassPool);
    }


    /**
     * Prints out classes and class members that are used as seeds in the
     * shrinking and obfuscation steps.
     */
    private void printSeeds() throws IOException
    {
        if (configuration.verbose)
        {
            System.out.println("Printing kept classes, fields, and methods...");
        }

        PrintStream ps = createPrintStream(configuration.printSeeds);
        try
        {
            new SeedPrinter(ps).write(configuration, programClassPool, libraryClassPool);
        }
        finally
        {
            closePrintStream(ps);
        }
    }


    /**
     * Performs the shrinking step.
     */
    private void shrink() throws IOException
    {
        if (configuration.verbose)
        {
            System.out.println("Shrinking...");

            // We'll print out some explanation, if requested.
            if (configuration.whyAreYouKeeping != null)
            {
                System.out.println("Explaining why classes and class members are being kept...");
            }

            // We'll print out the usage, if requested.
            if (configuration.printUsage != null)
            {
                System.out.println("Printing usage to [" + fileName(configuration.printUsage) + "]...");
            }
        }

        // Perform the actual shrinking.
        programClassPool =
            new Shrinker(configuration).execute(programClassPool, libraryClassPool);
    }


    /**
     * Performs the subroutine inlining step.
     */
    private void inlineSubroutines()
    {
        if (configuration.verbose)
        {
            System.out.println("Inlining subroutines...");
        }

        // Perform the actual inlining.
        new SubroutineInliner(configuration).execute(programClassPool);
    }


    /**
     * Performs the optimization step.
     */
    private boolean optimize() throws IOException
    {
        if (configuration.verbose)
        {
            System.out.println("Optimizing...");
        }

        // Perform the actual optimization.
        return new Optimizer(configuration).execute(programClassPool, libraryClassPool);
    }


    /**
     * Performs the obfuscation step.
     */
    private void obfuscate() throws IOException
    {
        if (configuration.verbose)
        {
            System.out.println("Obfuscating...");

            // We'll apply a mapping, if requested.
            if (configuration.applyMapping != null)
            {
                System.out.println("Applying mapping [" + fileName(configuration.applyMapping) + "]");
            }

            // We'll print out the mapping, if requested.
            if (configuration.printMapping != null)
            {
                System.out.println("Printing mapping to [" + fileName(configuration.printMapping) + "]...");
            }
        }

        // Perform the actual obfuscation.
        new Obfuscator(configuration).execute(programClassPool, libraryClassPool);
    }


    /**
     * Disambiguates the line numbers of all program classes, after
     * optimizations like method inlining and class merging.
     */
    private void linearizeLineNumbers()
    {
        programClassPool.classesAccept(new LineNumberLinearizer());
    }


    /**
     * Trims the line number table attributes of all program classes.
     */
    private void trimLineNumbers()
    {
        programClassPool.classesAccept(new AllAttributeVisitor(true,
                                       new LineNumberTableAttributeTrimmer()));
    }


    /**
     * Clears any JSE preverification information from the program classes.
     */
    private void clearPreverification()
    {
        programClassPool.classesAccept(
            new ClassVersionFilter(ClassConstants.CLASS_VERSION_1_6,
            new AllMethodVisitor(
            new AllAttributeVisitor(
            new NamedAttributeDeleter(ClassConstants.ATTR_StackMapTable)))));
    }


    /**
     * Performs the preverification step.
     */
    private void preverify()
    {
        if (configuration.verbose)
        {
            System.out.println("Preverifying...");
        }

        // Perform the actual preverification.
        new Preverifier(configuration).execute(programClassPool);
    }


    /**
     * Sorts the elements of all program classes.
     */
    private void sortClassElements()
    {
        programClassPool.classesAccept(new ClassElementSorter());
    }


    /**
     * Writes the output class files.
     */
    private void writeOutput() throws IOException
    {
        if (configuration.verbose)
        {
            System.out.println("Writing output...");
        }

        // Write out the program class pool.
        new OutputWriter(configuration).execute(programClassPool);
    }


    /**
     * Prints out the contents of the program classes.
     */
    private void dump() throws IOException
    {
        if (configuration.verbose)
        {
            System.out.println("Printing classes to [" + fileName(configuration.dump) + "]...");
        }

        PrintStream ps = createPrintStream(configuration.dump);
        try
        {
            programClassPool.classesAccept(new ClassPrinter(ps));
        }
        finally
        {
            closePrintStream(ps);
        }
    }


    /**
     * Returns a print stream for the given file, or the standard output if
     * the file name is empty.
     */
    private PrintStream createPrintStream(File file)
    throws FileNotFoundException
    {
        return file == Configuration.STD_OUT ? System.out :
            new PrintStream(
            new BufferedOutputStream(
            new FileOutputStream(file)));
    }


    /**
     * Closes the given print stream, or closes it if is the standard output.
     * @param printStream
     */
    private void closePrintStream(PrintStream printStream)
    {
        if (printStream == System.out)
        {
            printStream.flush();
        }
        else
        {
            printStream.close();
        }
    }


    /**
     * Returns the canonical file name for the given file, or "standard output"
     * if the file name is empty.
     */
    private String fileName(File file)
    {
        if (file == Configuration.STD_OUT)
        {
            return "standard output";
        }
        else
        {
            try
            {
                return file.getCanonicalPath();
            }
            catch (IOException ex)
            {
                return file.getPath();
            }
        }
    }


    /**
     * The main method for ProGuard.
     */
    public static void main(String[] args)
    {
        if (args.length == 0)
        {
            System.out.println(VERSION);
            System.out.println("Usage: java proguard.ProGuard [options ...]");
            System.exit(1);
        }

        // Create the default options.
        Configuration configuration = new Configuration();

        try
        {
            // Parse the options specified in the command line arguments.
            ConfigurationParser parser = new ConfigurationParser(args,
                                                                 System.getProperties());
            try
            {
                parser.parse(configuration);
            }
            finally
            {
                parser.close();
            }

            // Execute ProGuard with these options.
            new ProGuard(configuration).execute();
        }
        catch (Exception ex)
        {
            if (configuration.verbose)
            {
                // Print a verbose stack trace.
                ex.printStackTrace();
            }
            else
            {
                // Print just the stack trace message.
                System.err.println("Error: "+ex.getMessage());
            }

            System.exit(1);
        }

        System.exit(0);
    }
}
