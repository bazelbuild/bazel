/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
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

import java.io.*;


/**
 * A <code>WordReader</code> that returns words from an argument list.
 * Single arguments are split into individual words if necessary.
 *
 * @author Eric Lafortune
 */
public class ArgumentWordReader extends WordReader
{
    private final String[] arguments;

    private int index = 0;


//    /**
//     * Creates a new ArgumentWordReader for the given arguments.
//     */
//    public ArgumentWordReader(String[] arguments)
//    {
//        this(arguments, null);
//    }
//
//
    /**
     * Creates a new ArgumentWordReader for the given arguments, with the
     * given base directory.
     */
    public ArgumentWordReader(String[] arguments, File baseDir)
    {
        super(baseDir);

        this.arguments = arguments;
    }


    // Implementations for WordReader.

    protected String nextLine() throws IOException
    {
        return index < arguments.length ?
            arguments[index++] :
            null;
    }


    protected String lineLocationDescription()
    {
        return "argument number " + index;
    }


    /**
     * Test application that prints out the individual words of
     * the argument list.
     */
    public static void main(String[] args) {

        try
        {
            WordReader reader = new ArgumentWordReader(args, null);

            try
            {
                while (true)
                {
                    String word = reader.nextWord(false, false);
                    if (word == null)
                        System.exit(-1);

                    System.err.println("["+word+"]");
                }
            }
            catch (Exception ex)
            {
                ex.printStackTrace();
            }
            finally
            {
                reader.close();
            }
        }
        catch (IOException ex)
        {
            ex.printStackTrace();
        }
    }
}
