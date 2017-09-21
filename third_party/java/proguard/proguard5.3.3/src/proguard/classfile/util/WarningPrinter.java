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
package proguard.classfile.util;

import proguard.util.*;

import java.io.PrintStream;
import java.util.List;

/**
 * This class prints out and counts warnings.
 *
 * @author Eric Lafortune
 */
public class WarningPrinter
{
    private final PrintStream   printStream;
    private final StringMatcher classFilter;
    private int                 warningCount;


    /**
     * Creates a new WarningPrinter that prints to the System.err print stream.
     */
    public WarningPrinter()
    {
        this(System.err);
    }


    /**
     * Creates a new WarningPrinter that prints to the given print stream.
     */
    public WarningPrinter(PrintStream printStream)
    {
        this.printStream = printStream;
        this.classFilter = null;
    }


    /**
     * Creates a new WarningPrinter that prints to the given print stream,
     * except if the names of any involved classes matches the given filter.
     */
    public WarningPrinter(PrintStream printStream, List classFilter)
    {
        this.printStream = printStream;
        this.classFilter = classFilter == null ? null :
            new ListParser(new ClassNameParser()).parse(classFilter);
    }


    /**
     * Prints out the given warning and increments the warning count, if
     * the given class name passes the class name filter.
     */
    public void print(String className, String warning)
    {
        if (accepts(className))
        {
            print(warning);
        }
    }


    /**
     * Returns whether the given class name passes the class name filter.
     */
    public boolean accepts(String className)
    {
        return classFilter == null ||
            !classFilter.matches(className);
    }


    /**
     * Prints out the given warning and increments the warning count, if
     * the given class names pass the class name filter.
     */
    public void print(String className1, String className2, String warning)
    {
        if (accepts(className1, className2))
        {
            print(warning);
        }
    }


    /**
     * Returns whether the given class names pass the class name filter.
     */
    public boolean accepts(String className1, String className2)
    {
        return classFilter == null ||
            !(classFilter.matches(className1) ||
              classFilter.matches(className2));
    }


    /**
     * Prints out the given warning and increments the warning count.
     */
    private void print(String warning)
    {
        printStream.println(warning);

        warningCount++;
    }


    /**
     * Returns the number of warnings printed so far.
     */
    public int getWarningCount()
    {
        return warningCount;
    }
}
